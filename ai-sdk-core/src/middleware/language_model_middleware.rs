use ai_sdk_provider::language_model::{
    CallOptions, GenerateResponse, LanguageModel, StreamResponse,
};
use async_trait::async_trait;
use futures::future::BoxFuture;
use std::collections::HashMap;
use std::sync::Arc;

/// Language model middleware trait
///
/// Middlewares can transform parameters, wrap calls, and override model metadata.
/// They use the "onion" pattern: first middleware transforms input first, wraps output last.
#[async_trait]
pub trait LanguageModelMiddleware: Send + Sync {
    /// Specification version (always "v3")
    fn specification_version(&self) -> &str {
        "v3"
    }

    /// Override the provider name
    fn override_provider(&self, _model: &dyn LanguageModel) -> Option<String> {
        None
    }

    /// Override the model ID
    fn override_model_id(&self, _model: &dyn LanguageModel) -> Option<String> {
        None
    }

    /// Override supported URLs
    async fn override_supported_urls(
        &self,
        _model: &dyn LanguageModel,
    ) -> Option<HashMap<String, Vec<String>>> {
        None
    }

    /// Transform parameters before model call
    ///
    /// This is called before the actual model call, allowing the middleware to
    /// modify the call options (e.g., add default values, transform messages).
    async fn transform_params(
        &self,
        _call_type: CallType,
        params: CallOptions,
        _model: &dyn LanguageModel,
    ) -> Result<CallOptions, Box<dyn std::error::Error + Send + Sync>> {
        Ok(params)
    }

    /// Wrap the generate operation
    ///
    /// This allows the middleware to intercept the call, modify the result,
    /// or even call a different method entirely.
    async fn wrap_generate(
        &self,
        do_generate: GenerateFn,
        _do_stream: StreamFn,
        _params: &CallOptions,
        _model: &dyn LanguageModel,
    ) -> Result<GenerateResponse, Box<dyn std::error::Error + Send + Sync>> {
        do_generate().await
    }

    /// Wrap the stream operation
    ///
    /// This allows the middleware to intercept the call, modify the stream,
    /// or even call a different method entirely.
    async fn wrap_stream(
        &self,
        _do_generate: GenerateFn,
        do_stream: StreamFn,
        _params: &CallOptions,
        _model: &dyn LanguageModel,
    ) -> Result<StreamResponse, Box<dyn std::error::Error + Send + Sync + 'static>> {
        do_stream().await
    }
}

/// Type of model call (generate or stream)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallType {
    /// Non-streaming generation
    Generate,
    /// Streaming generation
    Stream,
}

/// Function type for do_generate calls
pub type GenerateFn = Arc<
    dyn Fn()
            -> BoxFuture<'static, Result<GenerateResponse, Box<dyn std::error::Error + Send + Sync>>>
        + Send
        + Sync,
>;

/// Function type for do_stream calls
pub type StreamFn = Arc<
    dyn Fn() -> BoxFuture<
            'static,
            Result<StreamResponse, Box<dyn std::error::Error + Send + Sync + 'static>>,
        > + Send
        + Sync,
>;

#[cfg(test)]
mod tests {
    use super::*;

    struct TestMiddleware;

    #[async_trait]
    impl LanguageModelMiddleware for TestMiddleware {}

    #[test]
    fn test_specification_version() {
        let middleware = TestMiddleware;
        assert_eq!(middleware.specification_version(), "v3");
    }
}
