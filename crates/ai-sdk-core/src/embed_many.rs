//! Multiple values embedding API with automatic batching

use crate::error::EmbedError;
use crate::retry::RetryPolicy;
use ai_sdk_provider::{EmbedOptions, EmbeddingModel, EmbeddingUsage};
use futures::stream::{self, StreamExt};
use std::sync::Arc;

/// Builder for embedding multiple values
///
/// Automatically handles batching and parallel execution for efficient
/// embedding of large datasets.
///
/// # Example
///
/// ```rust,ignore
/// use ai_sdk_core::embed_many;
/// use ai_sdk_openai::openai_embedding;
///
/// let texts = vec![
///     "Machine learning is a subset of AI".to_string(),
///     "Rust is a systems programming language".to_string(),
///     "Neural networks are inspired by the brain".to_string(),
/// ];
///
/// let result = embed_many()
///     .model(openai_embedding("text-embedding-3-small").api_key(api_key))
///     .values(texts)
///     .max_parallel_calls(5)
///     .execute()
///     .await?;
///
/// println!("Embedded {} texts", result.embeddings().len());
/// ```
pub struct EmbedManyBuilder<VALUE = String>
where
    VALUE: Send + Sync + Clone,
{
    model: Option<Arc<dyn EmbeddingModel<VALUE>>>,
    values: Vec<VALUE>,
    max_parallel_calls: usize,
    retry_policy: RetryPolicy,
}

impl<VALUE> EmbedManyBuilder<VALUE>
where
    VALUE: Send + Sync + Clone + 'static,
{
    /// Create a new embed_many builder
    pub fn new() -> Self {
        Self {
            model: None,
            values: Vec::new(),
            max_parallel_calls: usize::MAX,
            retry_policy: RetryPolicy::default(),
        }
    }

    /// Set the embedding model
    pub fn model<M: EmbeddingModel<VALUE> + 'static>(mut self, model: M) -> Self {
        self.model = Some(Arc::new(model));
        self
    }

    /// Set the values to embed
    pub fn values(mut self, values: Vec<VALUE>) -> Self {
        self.values = values;
        self
    }

    /// Set maximum number of parallel API calls (default: unlimited)
    pub fn max_parallel_calls(mut self, max: usize) -> Self {
        self.max_parallel_calls = max;
        self
    }

    /// Set custom retry policy
    pub fn retry_policy(mut self, retry_policy: RetryPolicy) -> Self {
        self.retry_policy = retry_policy;
        self
    }

    /// Execute the embedding
    pub async fn execute(self) -> Result<EmbedManyResult<VALUE>, EmbedError> {
        let model = self.model.clone().ok_or(EmbedError::MissingModel)?;

        if self.values.is_empty() {
            return Ok(EmbedManyResult {
                values: Vec::new(),
                embeddings: Vec::new(),
                total_usage: EmbeddingUsage { tokens: 0 },
            });
        }

        // Get model capabilities
        let max_embeddings_per_call = model.max_embeddings_per_call().await;
        let supports_parallel = model.supports_parallel_calls().await;

        // Determine batching strategy
        let result = if let Some(max_per_call) = max_embeddings_per_call {
            self.embed_with_batching(model, max_per_call, supports_parallel)
                .await?
        } else {
            self.embed_single_call(model).await?
        };

        Ok(result)
    }

    /// Embed all values in a single call
    async fn embed_single_call(
        &self,
        model: Arc<dyn EmbeddingModel<VALUE>>,
    ) -> Result<EmbedManyResult<VALUE>, EmbedError> {
        let response = self
            .retry_policy
            .retry(|| {
                let options = EmbedOptions {
                    values: self.values.clone(),
                    provider_options: None,
                    headers: None,
                };
                async { model.do_embed(options).await }
            })
            .await?;

        Ok(EmbedManyResult {
            values: self.values.clone(),
            embeddings: response.embeddings,
            total_usage: response.usage.unwrap_or(EmbeddingUsage { tokens: 0 }),
        })
    }

    /// Embed with batching and optional parallel execution
    async fn embed_with_batching(
        &self,
        model: Arc<dyn EmbeddingModel<VALUE>>,
        max_per_call: usize,
        supports_parallel: bool,
    ) -> Result<EmbedManyResult<VALUE>, EmbedError> {
        // Split into batches
        let batches: Vec<Vec<VALUE>> = self
            .values
            .chunks(max_per_call)
            .map(|chunk| chunk.to_vec())
            .collect();

        let mut all_embeddings = Vec::new();
        let mut total_usage = EmbeddingUsage { tokens: 0 };

        if supports_parallel && self.max_parallel_calls > 1 {
            // Parallel execution
            let max_concurrent = self.max_parallel_calls.min(batches.len());

            let results = stream::iter(batches)
                .map(|batch| {
                    let model = model.clone();
                    let retry_policy = self.retry_policy.clone();
                    async move {
                        retry_policy
                            .retry(|| {
                                let options = EmbedOptions {
                                    values: batch.clone(),
                                    provider_options: None,
                                    headers: None,
                                };
                                async { model.do_embed(options).await }
                            })
                            .await
                    }
                })
                .buffer_unordered(max_concurrent)
                .collect::<Vec<_>>()
                .await;

            // Aggregate results
            for result in results {
                let response = result?;
                all_embeddings.extend(response.embeddings);
                if let Some(usage) = response.usage {
                    total_usage.tokens += usage.tokens;
                }
            }
        } else {
            // Sequential execution
            for batch in batches {
                let response = self
                    .retry_policy
                    .retry(|| {
                        let options = EmbedOptions {
                            values: batch.clone(),
                            provider_options: None,
                            headers: None,
                        };
                        async { model.do_embed(options).await }
                    })
                    .await?;

                all_embeddings.extend(response.embeddings);
                if let Some(usage) = response.usage {
                    total_usage.tokens += usage.tokens;
                }
            }
        }

        Ok(EmbedManyResult {
            values: self.values.clone(),
            embeddings: all_embeddings,
            total_usage,
        })
    }
}

impl<VALUE> Default for EmbedManyBuilder<VALUE>
where
    VALUE: Send + Sync + Clone + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Result of embedding multiple values
#[derive(Debug, Clone)]
pub struct EmbedManyResult<VALUE> {
    /// The original values that were embedded
    pub values: Vec<VALUE>,
    /// The embedding vectors (in the same order as values)
    pub embeddings: Vec<Vec<f32>>,
    /// Total token usage across all API calls
    pub total_usage: EmbeddingUsage,
}

impl<VALUE> EmbedManyResult<VALUE> {
    /// Get all embeddings
    pub fn embeddings(&self) -> &[Vec<f32>] {
        &self.embeddings
    }

    /// Get embedding at index
    pub fn embedding(&self, index: usize) -> Option<&[f32]> {
        self.embeddings.get(index).map(|e| e.as_slice())
    }

    /// Get original values
    pub fn values(&self) -> &[VALUE] {
        &self.values
    }

    /// Get total token usage
    pub fn usage(&self) -> &EmbeddingUsage {
        &self.total_usage
    }

    /// Iterate over (value, embedding) pairs
    pub fn iter(&self) -> impl Iterator<Item = (&VALUE, &[f32])> {
        self.values
            .iter()
            .zip(self.embeddings.iter().map(|e| e.as_slice()))
    }
}

/// Entry point function for embedding multiple values
///
/// # Example
///
/// ```rust,ignore
/// use ai_sdk_core::embed_many;
/// use ai_sdk_openai::openai_embedding;
///
/// let result = embed_many()
///     .model(openai_embedding("text-embedding-3-small").api_key(api_key))
///     .values(vec!["text1".to_string(), "text2".to_string()])
///     .max_parallel_calls(5)
///     .execute()
///     .await?;
/// ```
pub fn embed_many<VALUE>() -> EmbedManyBuilder<VALUE>
where
    VALUE: Send + Sync + Clone + 'static,
{
    EmbedManyBuilder::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_sdk_provider::EmbedResponse;
    use async_trait::async_trait;

    struct DummyModel {
        max_per_call: Option<usize>,
        supports_parallel: bool,
    }

    #[async_trait]
    impl EmbeddingModel<String> for DummyModel {
        fn provider(&self) -> &str {
            "test"
        }
        fn model_id(&self) -> &str {
            "dummy"
        }
        async fn max_embeddings_per_call(&self) -> Option<usize> {
            self.max_per_call
        }
        async fn supports_parallel_calls(&self) -> bool {
            self.supports_parallel
        }
        async fn do_embed(
            &self,
            options: EmbedOptions<String>,
        ) -> Result<EmbedResponse, Box<dyn std::error::Error + Send + Sync>> {
            let embeddings = options
                .values
                .iter()
                .enumerate()
                .map(|(i, _)| vec![i as f32, (i + 1) as f32])
                .collect();

            Ok(EmbedResponse {
                embeddings,
                usage: Some(EmbeddingUsage {
                    tokens: options.values.len() as u32,
                }),
                provider_metadata: None,
                response: None,
            })
        }
    }

    #[test]
    fn test_embed_many_builder_defaults() {
        let builder = embed_many::<String>();
        assert!(builder.model.is_none());
        assert_eq!(builder.values.len(), 0);
        assert_eq!(builder.max_parallel_calls, usize::MAX);
    }

    #[tokio::test]
    async fn test_embed_many_empty_values() {
        let model = DummyModel {
            max_per_call: Some(10),
            supports_parallel: true,
        };

        let result = embed_many()
            .model(model)
            .values(Vec::<String>::new())
            .execute()
            .await
            .unwrap();

        assert_eq!(result.values().len(), 0);
        assert_eq!(result.embeddings().len(), 0);
        assert_eq!(result.usage().tokens, 0);
    }

    #[tokio::test]
    async fn test_embed_many_single_call() {
        let model = DummyModel {
            max_per_call: None, // No limit - single call
            supports_parallel: true,
        };

        let values = vec![
            "text1".to_string(),
            "text2".to_string(),
            "text3".to_string(),
        ];

        let result = embed_many()
            .model(model)
            .values(values.clone())
            .execute()
            .await
            .unwrap();

        assert_eq!(result.values().len(), 3);
        assert_eq!(result.embeddings().len(), 3);
        assert_eq!(result.embedding(0).unwrap(), &[0.0, 1.0]);
        assert_eq!(result.embedding(1).unwrap(), &[1.0, 2.0]);
        assert_eq!(result.embedding(2).unwrap(), &[2.0, 3.0]);
        assert_eq!(result.usage().tokens, 3);
    }

    #[tokio::test]
    async fn test_embed_many_batched() {
        let model = DummyModel {
            max_per_call: Some(2), // 2 embeddings per call
            supports_parallel: false,
        };

        let values = vec![
            "text1".to_string(),
            "text2".to_string(),
            "text3".to_string(),
            "text4".to_string(),
            "text5".to_string(),
        ];

        let result = embed_many()
            .model(model)
            .values(values.clone())
            .execute()
            .await
            .unwrap();

        assert_eq!(result.values().len(), 5);
        assert_eq!(result.embeddings().len(), 5);
        // Total tokens: 2 + 2 + 1 = 5 (batches of 2, 2, 1)
        assert_eq!(result.usage().tokens, 5);
    }

    #[tokio::test]
    async fn test_embed_many_parallel() {
        let model = DummyModel {
            max_per_call: Some(2),
            supports_parallel: true,
        };

        let values: Vec<String> = (0..10).map(|i| format!("text{}", i)).collect();

        let result = embed_many()
            .model(model)
            .values(values.clone())
            .max_parallel_calls(3)
            .execute()
            .await
            .unwrap();

        assert_eq!(result.values().len(), 10);
        assert_eq!(result.embeddings().len(), 10);
        assert_eq!(result.usage().tokens, 10);
    }

    #[tokio::test]
    async fn test_embed_many_iter() {
        let model = DummyModel {
            max_per_call: None,
            supports_parallel: true,
        };

        let values = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        let result = embed_many()
            .model(model)
            .values(values.clone())
            .execute()
            .await
            .unwrap();

        let pairs: Vec<(&String, &[f32])> = result.iter().collect();
        assert_eq!(pairs.len(), 3);
        assert_eq!(pairs[0].0, "a");
        assert_eq!(pairs[0].1, &[0.0, 1.0]);
        assert_eq!(pairs[1].0, "b");
        assert_eq!(pairs[1].1, &[1.0, 2.0]);
        assert_eq!(pairs[2].0, "c");
        assert_eq!(pairs[2].1, &[2.0, 3.0]);
    }
}
