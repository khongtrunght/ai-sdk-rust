use ai_sdk_provider::*;
use async_trait::async_trait;
use reqwest::Client;
use serde::Serialize;
use std::collections::HashMap;
use std::time::SystemTime;

/// OpenAI implementation of speech model.
pub struct OpenAISpeechModel {
    model_id: String,
    api_key: String,
    client: Client,
    base_url: String,
}

impl OpenAISpeechModel {
    /// Creates a new speech model with the specified model ID and API key.
    pub fn new(model_id: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            api_key: api_key.into(),
            client: Client::new(),
            base_url: "https://api.openai.com/v1".into(),
        }
    }

    /// Configures a custom base URL for the API endpoint.
    ///
    /// This is primarily useful for testing with mock servers.
    ///
    /// # Arguments
    /// * `base_url` - Custom base URL (e.g., "http://localhost:8080")
    ///
    /// # Example
    /// ```rust
    /// # use ai_sdk_openai::OpenAISpeechModel;
    /// let model = OpenAISpeechModel::new("tts-1", "api-key")
    ///     .with_base_url("http://localhost:8080");
    /// ```
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }
}

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

#[async_trait]
impl SpeechModel for OpenAISpeechModel {
    fn provider(&self) -> &str {
        "openai"
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn do_generate(
        &self,
        options: SpeechGenerateOptions,
    ) -> Result<SpeechGenerateResponse, Box<dyn std::error::Error + Send + Sync>> {
        let mut warnings = Vec::new();

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

        let url = format!("{}/audio/speech", self.base_url);

        // Build request body
        let request_body = SpeechRequest {
            model: self.model_id.clone(),
            input: options.text,
            voice: options.voice.unwrap_or_else(|| "alloy".into()),
            response_format: output_format,
            speed: options.speed,
            instructions: options.instructions,
        };

        let request_body_json = serde_json::to_string(&request_body)?;

        let mut request = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body);

        // Add custom headers if provided
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
