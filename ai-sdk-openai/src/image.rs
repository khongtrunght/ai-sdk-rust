use ai_sdk_provider::*;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub struct OpenAIImageModel {
    model_id: String,
    api_key: String,
    client: Client,
    base_url: String,
}

impl OpenAIImageModel {
    pub fn new(model_id: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            api_key: api_key.into(),
            client: Client::new(),
            base_url: "https://api.openai.com/v1".into(),
        }
    }

    /// Check if this model has a default response format
    fn has_default_response_format(&self) -> bool {
        matches!(
            self.model_id.as_str(),
            "gpt-image-1" | "gpt-image-1-mini"
        )
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

#[derive(Deserialize)]
struct ImageApiResponse {
    data: Vec<ImageResponseData>,
}

#[derive(Deserialize)]
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

        let url = format!("{}/images/generations", self.base_url);

        // Build request body
        let request_body = ImageRequest {
            model: self.model_id.clone(),
            prompt: options.prompt,
            n: if options.n > 0 { Some(options.n) } else { None },
            size: options.size,
            response_format: if !self.has_default_response_format() {
                Some("b64_json".into())
            } else {
                None
            },
        };

        // Add provider-specific options from provider_options
        // (In the full implementation, we would merge OpenAI-specific options here)

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
