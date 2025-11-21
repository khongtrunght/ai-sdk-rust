use super::language_model_middleware::{CallType, GenerateFn, LanguageModelMiddleware, StreamFn};
use ai_sdk_provider::language_model::{
    CallOptions, GenerateResponse, LanguageModel, StreamResponse,
};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

/// Wraps a language model with a chain of middlewares
///
/// Middlewares are applied in reverse order:
/// - First middleware in the list transforms params first
/// - Last middleware in the list wraps the call first (innermost)
///
/// This creates an "onion" pattern where the first middleware controls the outermost layer.
pub fn wrap_language_model(
    model: Box<dyn LanguageModel>,
    middlewares: Vec<Box<dyn LanguageModelMiddleware>>,
) -> Box<dyn LanguageModel> {
    if middlewares.is_empty() {
        return model;
    }

    // Reverse middlewares so first transforms input first
    let reversed: Vec<_> = middlewares.into_iter().rev().collect();

    // Fold/reduce to apply each middleware
    reversed.into_iter().fold(model, |wrapped, middleware| {
        Box::new(WrappedLanguageModel::new(wrapped, middleware))
    })
}

/// A language model wrapped with a middleware
struct WrappedLanguageModel {
    inner: Arc<Box<dyn LanguageModel>>,
    middleware: Arc<Box<dyn LanguageModelMiddleware>>,
    // Cache overridden values to avoid lifetime issues
    cached_provider: Option<String>,
    cached_model_id: Option<String>,
}

impl WrappedLanguageModel {
    /// Create a new wrapped model, evaluating overrides once at construction time
    fn new(inner: Box<dyn LanguageModel>, middleware: Box<dyn LanguageModelMiddleware>) -> Self {
        let cached_provider = middleware.override_provider(&*inner);
        let cached_model_id = middleware.override_model_id(&*inner);

        Self {
            inner: Arc::new(inner),
            middleware: Arc::new(middleware),
            cached_provider,
            cached_model_id,
        }
    }
}

#[async_trait]
impl LanguageModel for WrappedLanguageModel {
    fn specification_version(&self) -> &str {
        "v3"
    }

    fn provider(&self) -> &str {
        self.cached_provider
            .as_deref()
            .unwrap_or_else(|| self.inner.provider())
    }

    fn model_id(&self) -> &str {
        self.cached_model_id
            .as_deref()
            .unwrap_or_else(|| self.inner.model_id())
    }

    async fn supported_urls(&self) -> HashMap<String, Vec<String>> {
        self.middleware
            .override_supported_urls(self.inner.as_ref().as_ref())
            .await
            .unwrap_or_else(|| {
                // Need to call the async function, but we're in an async context
                // so we can't directly await here. Instead, create a future and await it.
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current()
                        .block_on(async { self.inner.supported_urls().await })
                })
            })
    }

    async fn do_generate(
        &self,
        options: CallOptions,
    ) -> Result<GenerateResponse, Box<dyn std::error::Error + Send + Sync>> {
        // Transform params
        let transformed = self
            .middleware
            .transform_params(CallType::Generate, options, self.inner.as_ref().as_ref())
            .await?;

        // Create closures for do_generate and do_stream
        let inner = Arc::clone(&self.inner);
        let transformed_for_generate = transformed.clone();

        let do_generate: GenerateFn = Arc::new(move || {
            let inner = Arc::clone(&inner);
            let opts = transformed_for_generate.clone();
            Box::pin(async move { inner.do_generate(opts).await })
        });

        let inner = Arc::clone(&self.inner);
        let transformed_for_stream = transformed.clone();

        let do_stream: StreamFn = Arc::new(move || {
            let inner = Arc::clone(&inner);
            let opts = transformed_for_stream.clone();
            Box::pin(async move { inner.do_stream(opts).await })
        });

        // Call middleware wrapper
        self.middleware
            .wrap_generate(
                do_generate,
                do_stream,
                &transformed,
                self.inner.as_ref().as_ref(),
            )
            .await
    }

    async fn do_stream(
        &self,
        options: CallOptions,
    ) -> Result<StreamResponse, Box<dyn std::error::Error + Send + Sync + 'static>> {
        // Transform params
        let transformed = self
            .middleware
            .transform_params(CallType::Stream, options, self.inner.as_ref().as_ref())
            .await?;

        // Create closures for do_generate and do_stream
        let inner = Arc::clone(&self.inner);
        let transformed_for_generate = transformed.clone();

        let do_generate: GenerateFn = Arc::new(move || {
            let inner = Arc::clone(&inner);
            let opts = transformed_for_generate.clone();
            Box::pin(async move { inner.do_generate(opts).await })
        });

        let inner = Arc::clone(&self.inner);
        let transformed_for_stream = transformed.clone();

        let do_stream: StreamFn = Arc::new(move || {
            let inner = Arc::clone(&inner);
            let opts = transformed_for_stream.clone();
            Box::pin(async move { inner.do_stream(opts).await })
        });

        // Call middleware wrapper
        self.middleware
            .wrap_stream(
                do_generate,
                do_stream,
                &transformed,
                self.inner.as_ref().as_ref(),
            )
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_sdk_provider::language_model::{Content, FinishReason, TextPart, Usage};

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
            Ok(GenerateResponse {
                content: vec![Content::Text(TextPart {
                    text: "test".to_string(),
                    provider_metadata: None,
                })],
                finish_reason: FinishReason::Stop,
                usage: Usage::default(),
                provider_metadata: None,
                request: None,
                response: None,
                warnings: vec![],
            })
        }

        async fn do_stream(
            &self,
            _opts: CallOptions,
        ) -> Result<StreamResponse, Box<dyn std::error::Error + Send + Sync + 'static>> {
            unimplemented!()
        }
    }

    struct TestMiddleware;

    #[async_trait]
    impl LanguageModelMiddleware for TestMiddleware {
        fn override_provider(&self, _model: &dyn LanguageModel) -> Option<String> {
            Some("overridden".to_string())
        }
    }

    #[tokio::test]
    async fn test_wrap_empty_middlewares() {
        let model = Box::new(DummyModel);
        let wrapped = wrap_language_model(model, vec![]);
        assert_eq!(wrapped.provider(), "test");
    }

    #[tokio::test]
    async fn test_wrap_with_middleware() {
        let model = Box::new(DummyModel);
        let wrapped = wrap_language_model(model, vec![Box::new(TestMiddleware)]);
        assert_eq!(wrapped.provider(), "overridden");
    }

    #[tokio::test]
    async fn test_do_generate_passthrough() {
        let model = Box::new(DummyModel);
        let wrapped = wrap_language_model(model, vec![Box::new(TestMiddleware)]);

        let result = wrapped.do_generate(CallOptions::default()).await.unwrap();

        match &result.content[0] {
            Content::Text(text_part) => assert_eq!(text_part.text, "test"),
            _ => panic!("Expected text content"),
        }
    }
}
