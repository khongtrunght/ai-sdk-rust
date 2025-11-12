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
    /// Generated content parts
    pub content: Vec<Content>,
    /// Reason why generation finished
    pub finish_reason: FinishReason,
    /// Token usage information
    pub usage: Usage,
    /// Provider-specific metadata
    pub provider_metadata: Option<SharedProviderMetadata>,
    /// Information about the request
    pub request: Option<RequestInfo>,
    /// Information about the response
    pub response: Option<ResponseInfo>,
    /// Warnings about the call
    pub warnings: Vec<CallWarning>,
}

/// Response from do_stream
pub struct StreamResponse {
    /// Stream of response parts
    pub stream: std::pin::Pin<Box<dyn Stream<Item = Result<StreamPart, StreamError>> + Send>>,
    /// Information about the request
    pub request: Option<RequestInfo>,
    /// Information about the response
    pub response: Option<ResponseInfo>,
}

/// Information about the request sent to the provider
#[derive(Debug, Clone)]
pub struct RequestInfo {
    /// Request body sent to the provider
    pub body: Option<serde_json::Value>,
}

/// Information about the response from the provider
#[derive(Debug, Clone)]
pub struct ResponseInfo {
    /// Response headers from the provider
    pub headers: Option<SharedHeaders>,
    /// Response body from the provider
    pub body: Option<serde_json::Value>,
    /// Provider's response ID
    pub id: Option<String>,
    /// Response timestamp
    pub timestamp: Option<String>,
    /// Actual model used (may differ from requested)
    pub model_id: Option<String>,
}

/// Error during streaming
#[derive(Debug, thiserror::Error)]
pub enum StreamError {
    /// Other streaming error
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
