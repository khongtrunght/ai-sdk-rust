//! Single value embedding API

use crate::error::EmbedError;
use crate::retry::RetryPolicy;
use ai_sdk_provider::{EmbedOptions, EmbeddingModel, EmbeddingUsage};
use std::sync::Arc;

/// Builder for single value embedding
///
/// # Example
///
/// ```rust,ignore
/// use ai_sdk_core::embed;
/// use ai_sdk_openai::openai_embedding;
///
/// let result = embed()
///     .model(openai_embedding("text-embedding-3-small").api_key(api_key))
///     .value("Hello, world!".to_string())
///     .execute()
///     .await?;
///
/// println!("Embedding dimensions: {}", result.embedding().len());
/// ```
pub struct EmbedBuilder<VALUE = String>
where
    VALUE: Send + Sync + Clone,
{
    model: Option<Arc<dyn EmbeddingModel<VALUE>>>,
    value: Option<VALUE>,
    retry_policy: RetryPolicy,
}

impl<VALUE> EmbedBuilder<VALUE>
where
    VALUE: Send + Sync + Clone + 'static,
{
    /// Create a new embed builder
    pub fn new() -> Self {
        Self {
            model: None,
            value: None,
            retry_policy: RetryPolicy::default(),
        }
    }

    /// Set the embedding model to use
    pub fn model<M: EmbeddingModel<VALUE> + 'static>(mut self, model: M) -> Self {
        self.model = Some(Arc::new(model));
        self
    }

    /// Set the value to embed
    pub fn value(mut self, value: VALUE) -> Self {
        self.value = Some(value);
        self
    }

    /// Set custom retry policy
    pub fn retry_policy(mut self, retry_policy: RetryPolicy) -> Self {
        self.retry_policy = retry_policy;
        self
    }

    /// Execute the embedding
    pub async fn execute(self) -> Result<EmbedResult<VALUE>, EmbedError> {
        let model = self.model.ok_or(EmbedError::MissingModel)?;
        let value = self.value.ok_or(EmbedError::MissingValue)?;

        // Call model with retry
        let response = self
            .retry_policy
            .retry(|| {
                let options = EmbedOptions {
                    values: vec![value.clone()],
                    provider_options: None,
                    headers: None,
                };
                async { model.do_embed(options).await }
            })
            .await?;

        // Extract first embedding
        let embedding = response
            .embeddings
            .into_iter()
            .next()
            .ok_or(EmbedError::EmptyResponse)?;

        Ok(EmbedResult {
            value,
            embedding,
            usage: response.usage,
        })
    }
}

impl<VALUE> Default for EmbedBuilder<VALUE>
where
    VALUE: Send + Sync + Clone + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Result of embedding a single value
#[derive(Debug, Clone)]
pub struct EmbedResult<VALUE> {
    /// The original value that was embedded
    pub value: VALUE,
    /// The embedding vector
    pub embedding: Vec<f32>,
    /// Token usage information
    pub usage: Option<EmbeddingUsage>,
}

impl<VALUE> EmbedResult<VALUE> {
    /// Get the embedding vector
    pub fn embedding(&self) -> &[f32] {
        &self.embedding
    }

    /// Get the original value
    pub fn value(&self) -> &VALUE {
        &self.value
    }

    /// Get token usage
    pub fn usage(&self) -> Option<&EmbeddingUsage> {
        self.usage.as_ref()
    }
}

/// Entry point function for embedding a single value
///
/// # Example
///
/// ```rust,ignore
/// use ai_sdk_core::embed;
/// use ai_sdk_openai::openai_embedding;
///
/// let result = embed()
///     .model(openai_embedding("text-embedding-3-small").api_key(api_key))
///     .value("Hello, world!".to_string())
///     .execute()
///     .await?;
/// ```
pub fn embed<VALUE>() -> EmbedBuilder<VALUE>
where
    VALUE: Send + Sync + Clone + 'static,
{
    EmbedBuilder::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_builder_defaults() {
        let builder = embed::<String>();
        assert!(builder.model.is_none());
        assert!(builder.value.is_none());
    }

    #[tokio::test]
    async fn test_embed_missing_model() {
        let result = embed::<String>().value("test".to_string()).execute().await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), EmbedError::MissingModel));
    }

    #[tokio::test]
    async fn test_embed_missing_value() {
        use ai_sdk_provider::EmbedResponse;
        use async_trait::async_trait;

        struct DummyModel;
        #[async_trait]
        impl EmbeddingModel<String> for DummyModel {
            fn provider(&self) -> &str {
                "test"
            }
            fn model_id(&self) -> &str {
                "dummy"
            }
            async fn max_embeddings_per_call(&self) -> Option<usize> {
                Some(100)
            }
            async fn supports_parallel_calls(&self) -> bool {
                true
            }
            async fn do_embed(
                &self,
                _options: EmbedOptions<String>,
            ) -> Result<EmbedResponse, Box<dyn std::error::Error + Send + Sync>> {
                Ok(EmbedResponse {
                    embeddings: vec![vec![0.1, 0.2, 0.3]],
                    usage: Some(EmbeddingUsage { tokens: 10 }),
                    provider_metadata: None,
                    response: None,
                })
            }
        }

        let result = embed::<String>().model(DummyModel).execute().await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), EmbedError::MissingValue));
    }

    #[tokio::test]
    async fn test_embed_success() {
        use ai_sdk_provider::EmbedResponse;
        use async_trait::async_trait;

        struct DummyModel;
        #[async_trait]
        impl EmbeddingModel<String> for DummyModel {
            fn provider(&self) -> &str {
                "test"
            }
            fn model_id(&self) -> &str {
                "dummy"
            }
            async fn max_embeddings_per_call(&self) -> Option<usize> {
                Some(100)
            }
            async fn supports_parallel_calls(&self) -> bool {
                true
            }
            async fn do_embed(
                &self,
                options: EmbedOptions<String>,
            ) -> Result<EmbedResponse, Box<dyn std::error::Error + Send + Sync>> {
                assert_eq!(options.values.len(), 1);
                assert_eq!(options.values[0], "test value");
                Ok(EmbedResponse {
                    embeddings: vec![vec![0.1, 0.2, 0.3]],
                    usage: Some(EmbeddingUsage { tokens: 10 }),
                    provider_metadata: None,
                    response: None,
                })
            }
        }

        let result = embed()
            .model(DummyModel)
            .value("test value".to_string())
            .execute()
            .await
            .unwrap();

        assert_eq!(result.value(), "test value");
        assert_eq!(result.embedding().len(), 3);
        assert_eq!(result.embedding(), &[0.1, 0.2, 0.3]);
        assert_eq!(result.usage().unwrap().tokens, 10);
    }
}
