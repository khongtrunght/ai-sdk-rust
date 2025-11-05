use ai_sdk_provider::*;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

pub struct OpenAIEmbeddingModel {
    model_id: String,
    api_key: String,
    client: Client,
    base_url: String,
}

impl OpenAIEmbeddingModel {
    pub fn new(model_id: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            api_key: api_key.into(),
            client: Client::new(),
            base_url: "https://api.openai.com/v1".into(),
        }
    }
}

#[derive(Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
    encoding_format: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
}

#[derive(Deserialize)]
struct EmbeddingApiResponse {
    data: Vec<EmbeddingData>,
    usage: Option<UsageInfo>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct UsageInfo {
    prompt_tokens: u32,
}

#[async_trait]
impl EmbeddingModel<String> for OpenAIEmbeddingModel {
    fn provider(&self) -> &str {
        "openai"
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn max_embeddings_per_call(&self) -> Option<usize> {
        Some(2048)
    }

    async fn supports_parallel_calls(&self) -> bool {
        true
    }

    async fn do_embed(
        &self,
        options: EmbedOptions<String>,
    ) -> Result<EmbedResponse, Box<dyn std::error::Error + Send + Sync>> {
        // Check max embeddings limit
        if let Some(max) = self.max_embeddings_per_call().await {
            if options.values.len() > max {
                return Err(format!(
                    "Too many embeddings: {} exceeds max of {}",
                    options.values.len(),
                    max
                )
                .into());
            }
        }

        let url = format!("{}/embeddings", self.base_url);

        // Extract dimensions from provider options if present
        let dimensions = options
            .provider_options
            .as_ref()
            .and_then(|opts| opts.get("openai"))
            .and_then(|openai_opts| openai_opts.get("dimensions"))
            .and_then(|d| match d {
                JsonValue::Number(n) => n.as_u64().map(|n| n as u32),
                _ => None,
            });

        let user = options
            .provider_options
            .as_ref()
            .and_then(|opts| opts.get("openai"))
            .and_then(|openai_opts| openai_opts.get("user"))
            .and_then(|u| match u {
                JsonValue::String(s) => Some(s.clone()),
                _ => None,
            });

        let request_body = EmbeddingRequest {
            model: self.model_id.clone(),
            input: options.values,
            encoding_format: "float".into(),
            dimensions,
            user,
        };

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
        let response_headers: std::collections::HashMap<String, String> = response
            .headers()
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
            .collect();

        if !status.is_success() {
            let error_text = response.text().await?;
            return Err(format!("API error {}: {}", status, error_text).into());
        }

        let response_body = response.text().await?;
        let api_response: EmbeddingApiResponse = serde_json::from_str(&response_body)?;

        Ok(EmbedResponse {
            embeddings: api_response
                .data
                .into_iter()
                .map(|d| d.embedding)
                .collect(),
            usage: api_response.usage.map(|u| EmbeddingUsage {
                tokens: u.prompt_tokens,
            }),
            provider_metadata: None,
            response: Some(embedding_model::ResponseInfo {
                headers: Some(response_headers),
                body: serde_json::from_str(&response_body).ok(),
            }),
        })
    }
}
