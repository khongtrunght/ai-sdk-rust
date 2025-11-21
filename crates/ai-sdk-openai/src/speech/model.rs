use ai_sdk_provider::*;
use ai_sdk_provider_utils::merge_headers_reqwest;
use async_trait::async_trait;
use reqwest::Client;
use serde::Serialize;
use std::collections::HashMap;
use std::time::SystemTime;

use crate::openai_config::{OpenAIConfig, OpenAIUrlOptions};
use crate::speech::OpenAISpeechProviderOptions;

#[derive(Serialize)]
struct SpeechRequest {
    model: String,
    input: String,
    voice: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    speed: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
}

/// OpenAI implementation of speech model.
pub struct OpenAISpeechModel {
    model_id: String,
    config: OpenAIConfig,
    client: Client,
}

impl OpenAISpeechModel {
    /// Creates a new speech model with the specified model ID and configuration.
    ///
    /// # Arguments
    /// * `model_id` - The model ID (e.g., "tts-1", "tts-1-hd")
    /// * `config` - OpenAI configuration
    ///
    /// # Example
    /// ```rust,ignore
    /// use ai_sdk_openai::{OpenAIConfig, speech::{OpenAISpeechModel, TTS_1}};
    ///
    /// let api_key = std::env::var("OPENAI_API_KEY").unwrap();
    /// let config = OpenAIConfig::new(
    ///     "openai",
    ///     |opts| format!("https://api.openai.com/v1{}", opts.path),
    ///     move || {
    ///         let mut headers = std::collections::HashMap::new();
    ///         headers.insert("Authorization".to_string(), format!("Bearer {}", api_key));
    ///         headers
    ///     }
    /// );
    ///
    /// let model = OpenAISpeechModel::new(TTS_1, config);
    /// ```
    pub fn new(model_id: impl Into<String>, config: impl Into<OpenAIConfig>) -> Self {
        Self {
            model_id: model_id.into(),
            config: config.into(),
            client: Client::new(),
        }
    }
}

#[async_trait]
impl SpeechModel for OpenAISpeechModel {
    fn provider(&self) -> &str {
        &self.config.provider
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn do_generate(
        &self,
        options: SpeechGenerateOptions,
    ) -> Result<SpeechGenerateResponse, Box<dyn std::error::Error + Send + Sync>> {
        let mut warnings = Vec::new();

        // Parse provider-specific options
        let openai_options = OpenAISpeechProviderOptions::from(options.provider_options.clone());

        // Validate output format
        let output_format = if let Some(fmt) = &options.output_format {
            if ["mp3", "opus", "aac", "flac", "wav", "pcm"].contains(&fmt.as_str()) {
                Some(fmt.clone())
            } else {
                warnings.push(SpeechCallWarning::UnsupportedSetting {
                    setting: "outputFormat".into(),
                    details: Some(format!(
                        "Unsupported output format: {}. Using mp3 instead.",
                        fmt
                    )),
                });
                Some("mp3".into())
            }
        } else {
            Some("mp3".into())
        };

        // Language is not supported by OpenAI speech models
        if options.language.is_some() {
            warnings.push(SpeechCallWarning::UnsupportedSetting {
                setting: "language".into(),
                details: Some(
                    "OpenAI speech models do not support language selection. Language parameter was ignored.".into()
                ),
            });
        }

        // Build request URL using config
        let url = (self.config.url)(OpenAIUrlOptions {
            model_id: self.model_id.clone(),
            path: "/audio/speech".to_string(),
        });

        // Merge provider options with general options
        // Provider options take precedence
        let speed = openai_options.speed.or(options.speed);
        let instructions = openai_options.instructions.or(options.instructions);

        // Build request body
        let request_body = SpeechRequest {
            model: self.model_id.clone(),
            input: options.text,
            voice: options.voice.unwrap_or_else(|| "alloy".into()),
            response_format: output_format,
            speed,
            instructions,
        };

        let request_body_json = serde_json::to_string(&request_body)?;

        // Send request with merged headers
        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .headers(merge_headers_reqwest(
                (self.config.headers)(),
                options.headers.as_ref(),
            ))
            .json(&request_body)
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

        // Get audio as binary
        let audio_bytes = response.bytes().await?.to_vec();

        Ok(SpeechGenerateResponse {
            audio: AudioData::Binary(audio_bytes),
            warnings,
            request: Some(speech_model::RequestInfo {
                body: Some(request_body_json),
            }),
            response: speech_model::ResponseInfo {
                timestamp: SystemTime::now(),
                model_id: self.model_id.clone(),
                headers: Some(response_headers),
                body: None, // Audio is in the audio field, not body
            },
            provider_metadata: None,
        })
    }
}
