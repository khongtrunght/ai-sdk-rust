use super::*;
use async_trait::async_trait;

/// Main embedding model trait (v3 specification)
///
/// VALUE is the type of the values that the model can embed.
/// This will allow us to go beyond text embeddings in the future,
/// e.g. to support image embeddings.
#[async_trait]
pub trait EmbeddingModel<VALUE>: Send + Sync
where
    VALUE: Send + Sync,
{
    /// Specification version (always "v3")
    fn specification_version(&self) -> &str {
        "v3"
    }

    /// Provider ID (e.g., "openai")
    fn provider(&self) -> &str;

    /// Provider-specific model ID (e.g., "text-embedding-3-small")
    fn model_id(&self) -> &str;

    /// Limit of how many embeddings can be generated in a single API call.
    ///
    /// Returns None for models that do not have a limit.
    async fn max_embeddings_per_call(&self) -> Option<usize>;

    /// True if the model can handle multiple embedding calls in parallel.
    async fn supports_parallel_calls(&self) -> bool;

    /// Generates a list of embeddings for the given input values.
    ///
    /// Naming: "do" prefix to prevent accidental direct usage of the method by the user.
    async fn do_embed(
        &self,
        options: EmbedOptions<VALUE>,
    ) -> Result<EmbedResponse, Box<dyn std::error::Error + Send + Sync>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyEmbeddingModel;

    #[async_trait]
    impl EmbeddingModel<String> for DummyEmbeddingModel {
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

    #[tokio::test]
    async fn test_embedding_model_trait() {
        let model = DummyEmbeddingModel;
        assert_eq!(model.provider(), "test");
        assert_eq!(model.model_id(), "dummy");
        assert_eq!(model.specification_version(), "v3");
        assert_eq!(model.max_embeddings_per_call().await, Some(100));
        assert!(model.supports_parallel_calls().await);
    }
}
