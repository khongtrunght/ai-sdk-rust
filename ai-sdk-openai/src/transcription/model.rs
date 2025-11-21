use ai_sdk_provider::*;
use ai_sdk_provider_utils::merge_headers_reqwest;
use async_trait::async_trait;
use reqwest::{multipart, Client};
use serde::Deserialize;
use std::collections::HashMap;
use std::time::SystemTime;

use crate::{
    openai_config::{OpenAIConfig, OpenAIUrlOptions},
    transcription::{map_language, OpenAITranscriptionProviderOptions},
};

#[derive(Deserialize, Debug)]
struct TranscriptionApiResponse {
    text: String,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    duration: Option<f64>,
    #[serde(default)]
    segments: Option<Vec<Segment>>,
    #[serde(default)]
    words: Option<Vec<Word>>,
}

#[derive(Deserialize, Debug)]
struct Segment {
    text: String,
    start: f64,
    end: f64,
}

#[derive(Deserialize, Debug)]
struct Word {
    word: String,
    start: f64,
    end: f64,
}

/// OpenAI implementation of transcription model.
pub struct OpenAITranscriptionModel {
    model_id: String,
    client: Client,
    config: OpenAIConfig,
}

impl OpenAITranscriptionModel {
    /// Creates a new transcription model with the specified model ID and API key.
    pub fn new(model_id: impl Into<String>, config: impl Into<OpenAIConfig>) -> Self {
        Self {
            model_id: model_id.into(),
            client: Client::new(),
            config: config.into(),
        }
    }

    // Maps a media type to its corresponding file extension.
    // It was originally introduced to set a filename for audio file uploads
    //
    // @param media_type The media type to map.
    // @returns The corresponding file extension
    // @see https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/MIME_types/Common_types
    fn get_file_extension(&self, media_type: &str) -> String {
        let lowered = media_type.to_lowercase();
        let subtype = lowered.split('/').nth(1).unwrap_or_default();
        match subtype {
            "mpeg" => "mp3".to_string(),
            "x-wav" => "wav".to_string(),
            "opus" => "ogg".to_string(),
            "mp4" | "x-m4a" => "m4a".to_string(),
            _ => subtype.to_string(),
        }
    }

    fn supports_verbose_json(&self) -> bool {
        // gpt-4o-transcribe and gpt-4o-mini-transcribe don't support verbose_json, use json instead
        // https://platform.openai.com/docs/api-reference/audio/createTranscription#audio_createtranscription-response_format
        !matches!(
            self.model_id.as_str(),
            "gpt-4o-transcribe" | "gpt-4o-mini-transcribe"
        )
    }
}

#[async_trait]
impl TranscriptionModel for OpenAITranscriptionModel {
    fn provider(&self) -> &str {
        "openai"
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn do_generate(
        &self,
        options: TranscriptionOptions,
    ) -> Result<TranscriptionResponse, Box<dyn std::error::Error + Send + Sync>> {
        let warnings = Vec::new();

        // Parse provider-specific options
        let openai_options =
            OpenAITranscriptionProviderOptions::from(options.provider_options.clone());

        // Convert audio to bytes
        let audio_bytes = match options.audio {
            AudioInput::Binary(bytes) => bytes,
            AudioInput::Base64(b64) => {
                // Decode base64 string
                use base64::Engine;
                base64::engine::general_purpose::STANDARD
                    .decode(&b64)
                    .map_err(|e| format!("Failed to decode base64 audio: {}", e))?
            }
        };

        let url = (self.config.url)(OpenAIUrlOptions {
            path: "/audio/transcriptions".to_string(),
            model_id: self.model_id.clone(),
        });

        // Get file extension from media type
        let extension = self.get_file_extension(&options.media_type);
        let filename = format!("audio.{extension}");

        // Create multipart form
        let audio_part = multipart::Part::bytes(audio_bytes)
            .file_name(filename.clone())
            .mime_str(&options.media_type)?;

        let mut form = multipart::Form::new()
            .text("model", self.model_id.clone())
            .part("file", audio_part);

        // Add provider-specific options to form
        // https://platform.openai.com/docs/api-reference/audio/createTranscription#audio_createtranscription-response_format
        // Prefer verbose_json to get segments for models that support it
        let response_format = if self.supports_verbose_json() {
            "verbose_json"
        } else {
            "json"
        };
        form = form.text("response_format", response_format);

        // Add language if provided
        if let Some(language) = &openai_options.language {
            form = form.text("language", language.clone());
        }

        // Add prompt if provided
        if let Some(prompt) = &openai_options.prompt {
            form = form.text("prompt", prompt.clone());
        }

        // Add temperature if provided
        if let Some(temperature) = openai_options.temperature {
            form = form.text("temperature", temperature.to_string());
        }

        // Add timestamp granularities if provided, otherwise use default for verbose_json
        if let Some(granularities) = &openai_options.timestamp_granularities {
            for granularity in granularities {
                form = form.text("timestamp_granularities[]", granularity.clone());
            }
        } else if self.supports_verbose_json() {
            // Default to segment granularity for verbose_json
            form = form.text("timestamp_granularities[]", "segment");
        }

        // Send request with merged headers
        let response = self
            .client
            .post(&url)
            .multipart(form)
            .headers(merge_headers_reqwest(
                (self.config.headers)(),
                options.headers.as_ref(),
            ))
            .send()
            .await?;

        let status = response.status();
        let response_headers: HashMap<String, String> = response
            .headers()
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
            .collect();

        if !status.is_success() {
            let error_text = response.text().await?;
            return Err(format!("API error {}: {}", status, error_text).into());
        }

        let api_response: TranscriptionApiResponse = response.json().await?;

        // Convert segments or words to our format
        let segments = if let Some(segs) = api_response.segments {
            segs.into_iter()
                .map(|s| TranscriptionSegment {
                    text: s.text,
                    start_second: s.start,
                    end_second: s.end,
                })
                .collect()
        } else if let Some(words) = api_response.words {
            // If no segments but we have words, use words as segments
            words
                .into_iter()
                .map(|w| TranscriptionSegment {
                    text: w.word,
                    start_second: w.start,
                    end_second: w.end,
                })
                .collect()
        } else {
            Vec::new()
        };

        // Map language name to ISO-639-1 code
        let language = api_response
            .language
            .as_ref()
            .and_then(|lang| map_language(lang));

        Ok(TranscriptionResponse {
            text: api_response.text,
            segments,
            language,
            duration_in_seconds: api_response.duration,
            warnings,
            request: None,
            response: transcription_model::ResponseInfo {
                timestamp: SystemTime::now(),
                model_id: self.model_id.clone(),
                headers: Some(response_headers),
                body: None,
            },
            provider_metadata: None,
        })
    }
}
