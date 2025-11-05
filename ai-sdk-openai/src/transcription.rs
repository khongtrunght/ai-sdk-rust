use ai_sdk_provider::*;
use async_trait::async_trait;
use reqwest::{multipart, Client};
use serde::Deserialize;
use std::collections::HashMap;
use std::time::SystemTime;

pub struct OpenAITranscriptionModel {
    model_id: String,
    api_key: String,
    client: Client,
    base_url: String,
}

impl OpenAITranscriptionModel {
    pub fn new(model_id: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            api_key: api_key.into(),
            client: Client::new(),
            base_url: "https://api.openai.com/v1".into(),
        }
    }

    fn get_file_extension(&self, media_type: &str) -> &str {
        // Map common IANA media types to file extensions
        match media_type {
            "audio/mpeg" | "audio/mp3" => "mp3",
            "audio/wav" | "audio/wave" | "audio/x-wav" => "wav",
            "audio/mp4" | "audio/m4a" => "m4a",
            "audio/webm" => "webm",
            "audio/ogg" => "ogg",
            "audio/flac" => "flac",
            _ => "mp3", // Default fallback
        }
    }

    fn supports_verbose_json(&self) -> bool {
        // gpt-4o models don't support verbose_json, use json instead
        !self.model_id.starts_with("gpt-4o-")
    }
}

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

// Language mapping from OpenAI's full language names to ISO-639-1 codes
// https://platform.openai.com/docs/guides/speech-to-text#supported-languages
fn map_language(language: &str) -> Option<String> {
    let lang_code = match language.to_lowercase().as_str() {
        "afrikaans" => "af",
        "arabic" => "ar",
        "armenian" => "hy",
        "azerbaijani" => "az",
        "belarusian" => "be",
        "bosnian" => "bs",
        "bulgarian" => "bg",
        "catalan" => "ca",
        "chinese" => "zh",
        "croatian" => "hr",
        "czech" => "cs",
        "danish" => "da",
        "dutch" => "nl",
        "english" => "en",
        "estonian" => "et",
        "finnish" => "fi",
        "french" => "fr",
        "galician" => "gl",
        "german" => "de",
        "greek" => "el",
        "hebrew" => "he",
        "hindi" => "hi",
        "hungarian" => "hu",
        "icelandic" => "is",
        "indonesian" => "id",
        "italian" => "it",
        "japanese" => "ja",
        "kannada" => "kn",
        "kazakh" => "kk",
        "korean" => "ko",
        "latvian" => "lv",
        "lithuanian" => "lt",
        "macedonian" => "mk",
        "malay" => "ms",
        "marathi" => "mr",
        "maori" => "mi",
        "nepali" => "ne",
        "norwegian" => "no",
        "persian" => "fa",
        "polish" => "pl",
        "portuguese" => "pt",
        "romanian" => "ro",
        "russian" => "ru",
        "serbian" => "sr",
        "slovak" => "sk",
        "slovenian" => "sl",
        "spanish" => "es",
        "swahili" => "sw",
        "swedish" => "sv",
        "tagalog" => "tl",
        "tamil" => "ta",
        "thai" => "th",
        "turkish" => "tr",
        "ukrainian" => "uk",
        "urdu" => "ur",
        "vietnamese" => "vi",
        "welsh" => "cy",
        _ => return None,
    };
    Some(lang_code.to_string())
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

        let url = format!("{}/audio/transcriptions", self.base_url);

        // Get file extension from media type
        let extension = self.get_file_extension(&options.media_type);
        let filename = format!("audio.{}", extension);

        // Create multipart form
        let audio_part = multipart::Part::bytes(audio_bytes)
            .file_name(filename.clone())
            .mime_str(&options.media_type)?;

        let mut form = multipart::Form::new()
            .text("model", self.model_id.clone())
            .part("file", audio_part);

        // Determine response format based on model
        let response_format = if self.supports_verbose_json() {
            "verbose_json"
        } else {
            "json"
        };
        form = form.text("response_format", response_format);

        // Add timestamp granularities for verbose_json
        if self.supports_verbose_json() {
            form = form.text("timestamp_granularities[]", "segment");
        }

        // Add custom headers if provided
        let mut request = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .multipart(form);

        if let Some(headers) = &options.headers {
            for (key, value) in headers {
                request = request.header(key, value);
            }
        }

        let response = request.send().await?;

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
