use crate::{
    embedding::OpenAIEmbeddingProviderOptions,
    openai_config::{OpenAIConfig, OpenAIUrlOptions},
};
use ai_sdk_provider::*;
use ai_sdk_provider_utils::merge_headers_reqwest;
use async_trait::async_trait;
use reqwest::Client;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
    encoding_format: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct EmbeddingApiResponse {
    data: Vec<EmbeddingData>,
    usage: Option<UsageInfo>,
}

#[derive(Serialize, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
struct UsageInfo {
    prompt_tokens: u32,
}

/// OpenAI implementation of embedding model.
pub struct OpenAIEmbeddingModel {
    model_id: String,

    client: Client,

    config: OpenAIConfig,
}

impl OpenAIEmbeddingModel {
    const MAX_EMBEDDINGS_PER_CALL: usize = 2048;
    const SUPPORTS_PARALLEL_CALLS: bool = true;
    /// Creates a new embedding model with the specified model ID and API key.
    pub fn new(model_id: impl Into<String>, config: impl Into<OpenAIConfig>) -> Self {
        Self {
            model_id: model_id.into(),
            client: Client::new(),
            config: config.into(),
        }
    }
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
        Some(Self::MAX_EMBEDDINGS_PER_CALL)
    }

    async fn supports_parallel_calls(&self) -> bool {
        Self::SUPPORTS_PARALLEL_CALLS
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

        let url = (self.config.url)(OpenAIUrlOptions {
            model_id: self.model_id.clone(),
            path: "/embeddings".into(),
        });

        let OpenAIEmbeddingProviderOptions { dimensions, user }: OpenAIEmbeddingProviderOptions =
            options.provider_options.into();

        let request_body = EmbeddingRequest {
            model: self.model_id.clone(),
            input: options.values,
            encoding_format: "float".into(),
            dimensions,
            user,
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
            embeddings: api_response.data.into_iter().map(|d| d.embedding).collect(),
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
