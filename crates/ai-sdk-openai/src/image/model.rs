use ai_sdk_provider::*;
use ai_sdk_provider_utils::merge_headers_reqwest;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::openai_config::{OpenAIConfig, OpenAIUrlOptions};

/// OpenAI implementation of image model.
pub struct OpenAIImageModel {
    model_id: String,
    client: Client,
    config: OpenAIConfig,
}

impl OpenAIImageModel {
    /// Creates a new image model with the specified model ID and API key.
    pub fn new(model_id: impl Into<String>, config: impl Into<OpenAIConfig>) -> Self {
        Self {
            model_id: model_id.into(),
            client: Client::new(),
            config: config.into(),
        }
    }

    /// Check if this model has a default response format
    fn has_default_response_format(&self) -> bool {
        matches!(self.model_id.as_str(), "gpt-image-1" | "gpt-image-1-mini")
    }
}

#[derive(Serialize)]
struct ImageRequest {
    model: String,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<String>,
}

#[derive(Deserialize, Debug)]
struct ImageApiResponse {
    data: Vec<ImageResponseData>,
}

#[derive(Deserialize, Debug)]
struct ImageResponseData {
    b64_json: String,
    #[serde(default)]
    revised_prompt: Option<String>,
}

#[async_trait]
impl ImageModel for OpenAIImageModel {
    fn provider(&self) -> &str {
        "openai"
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn max_images_per_call(&self) -> Option<usize> {
        match self.model_id.as_str() {
            "dall-e-3" => Some(1),
            "dall-e-2" | "gpt-image-1" | "gpt-image-1-mini" => Some(10),
            _ => Some(1),
        }
    }

    async fn do_generate(
        &self,
        options: ImageGenerateOptions,
    ) -> Result<ImageGenerateResponse, Box<dyn std::error::Error + Send + Sync>> {
        let mut warnings = Vec::new();

        // Check unsupported settings
        if options.aspect_ratio.is_some() {
            warnings.push(ImageCallWarning::UnsupportedSetting {
                setting: "aspectRatio".into(),
                details: Some(
                    "This model does not support aspect ratio. Use `size` instead.".into(),
                ),
            });
        }

        if options.seed.is_some() {
            warnings.push(ImageCallWarning::UnsupportedSetting {
                setting: "seed".into(),
                details: None,
            });
        }

        // let url = format!("{}/images/generations", (self.config.url)());
        let url = (self.config.url)(OpenAIUrlOptions {
            model_id: self.model_id.clone(),
            path: "/images/generations".into(),
        });

        // Build request body
        let request_body = ImageRequest {
            model: self.model_id.clone(),
            prompt: options.prompt,
            n: options.n,
            size: options.size,
            response_format: if !self.has_default_response_format() {
                Some("b64_json".into())
            } else {
                None
            },
        };

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

        let api_response: ImageApiResponse = response.json().await?;

        // Build provider metadata
        let mut openai_metadata = JsonObject::new();
        let images_metadata: Vec<_> = api_response
            .data
            .iter()
            .map(|d| {
                d.revised_prompt.as_ref().map(|p| {
                    let mut map = JsonObject::new();
                    map.insert("revisedPrompt".to_string(), JsonValue::String(p.clone()));
                    JsonValue::Object(map)
                })
            })
            .map(|opt| opt.unwrap_or(JsonValue::Null))
            .collect();

        openai_metadata.insert("images".to_string(), JsonValue::Array(images_metadata));

        let mut provider_metadata = HashMap::new();
        provider_metadata.insert("openai".to_string(), openai_metadata);

        Ok(ImageGenerateResponse {
            images: api_response
                .data
                .into_iter()
                .map(|d| ImageData::Base64(d.b64_json))
                .collect(),
            warnings,
            provider_metadata: Some(ImageProviderMetadata {
                metadata: provider_metadata,
            }),
            response: image_model::ResponseInfo {
                timestamp: std::time::SystemTime::now(),
                model_id: self.model_id.clone(),
                headers: Some(response_headers),
            },
        })
    }
}
