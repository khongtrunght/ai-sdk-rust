use super::*;
use crate::{SharedHeaders, SharedProviderMetadata};
use async_trait::async_trait;
use std::collections::HashMap;
use tokio_stream::Stream;

/// Main language model trait (v3 specification)
#[async_trait]
pub trait LanguageModel: Send + Sync {
    /// Specification version (always "v3")
    fn specification_version(&self) -> &str {
        "v3"
    }

    /// Provider ID (e.g., "openai", "anthropic")
    fn provider(&self) -> &str;

    /// Provider-specific model ID (e.g., "gpt-4")
    fn model_id(&self) -> &str;

    /// Supported URL patterns by media type
    ///
    /// Returns a map where:
    /// - Keys are media type patterns (e.g., "image/*", "application/pdf")
    /// - Values are regex patterns for supported URLs
    async fn supported_urls(&self) -> HashMap<String, Vec<String>> {
        HashMap::new()
    }

    /// Generate a language model output (non-streaming)
    async fn do_generate(
        &self,
        options: CallOptions,
    ) -> Result<GenerateResponse, Box<dyn std::error::Error + Send + Sync>>;

    /// Generate a language model output (streaming)
    async fn do_stream(
        &self,
        options: CallOptions,
    ) -> Result<StreamResponse, Box<dyn std::error::Error + Send + Sync + 'static>>;
}

/// Response from do_generate
#[derive(Debug, Clone)]
pub struct GenerateResponse {
    pub content: Vec<Content>,
    pub finish_reason: FinishReason,
    pub usage: Usage,
    pub provider_metadata: Option<SharedProviderMetadata>,
    pub request: Option<RequestInfo>,
    pub response: Option<ResponseInfo>,
    pub warnings: Vec<CallWarning>,
}

/// Response from do_stream
pub struct StreamResponse {
    pub stream: std::pin::Pin<Box<dyn Stream<Item = Result<StreamPart, StreamError>> + Send>>,
    pub request: Option<RequestInfo>,
    pub response: Option<ResponseInfo>,
}

#[derive(Debug, Clone)]
pub struct RequestInfo {
    pub body: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct ResponseInfo {
    pub headers: Option<SharedHeaders>,
    pub body: Option<serde_json::Value>,
    pub id: Option<String>,
    pub timestamp: Option<String>,
    pub model_id: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum StreamError {
    #[error("Stream error: {0}")]
    Other(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyModel;

    #[async_trait]
    impl LanguageModel for DummyModel {
        fn provider(&self) -> &str {
            "test"
        }
        fn model_id(&self) -> &str {
            "dummy"
        }

        async fn do_generate(
            &self,
            _opts: CallOptions,
        ) -> Result<GenerateResponse, Box<dyn std::error::Error + Send + Sync>> {
            unimplemented!()
        }

        async fn do_stream(
            &self,
            _opts: CallOptions,
        ) -> Result<StreamResponse, Box<dyn std::error::Error + Send + Sync>> {
            unimplemented!()
        }
    }

    #[test]
    fn test_trait_implementation() {
        let model = DummyModel;
        assert_eq!(model.provider(), "test");
        assert_eq!(model.specification_version(), "v3");
    }
}
