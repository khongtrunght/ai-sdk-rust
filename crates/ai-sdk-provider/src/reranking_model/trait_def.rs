use super::{RerankOptions, RerankResponse};
use async_trait::async_trait;

/// Specification for a reranking model that implements the reranking model interface version 3.
#[async_trait]
pub trait RerankingModel: Send + Sync {
    /// The reranking model must specify which reranking model interface version it implements.
    /// This returns "v3" for all implementations of this trait.
    fn specification_version(&self) -> &str {
        "v3"
    }

    /// Provider ID (e.g., "cohere", "anthropic")
    fn provider(&self) -> &str;

    /// Provider-specific model ID
    fn model_id(&self) -> &str;

    /// Rerank a list of documents using the query.
    ///
    /// # Arguments
    ///
    /// * `options` - The reranking options including documents, query, and optional top-N limit
    ///
    /// # Returns
    ///
    /// A `RerankResponse` containing the ranked list of documents sorted by relevance score
    async fn do_rerank(
        &self,
        options: RerankOptions,
    ) -> Result<RerankResponse, Box<dyn std::error::Error + Send + Sync>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reranking_model::{Documents, RankingItem};

    struct TestRerankingModel;

    #[async_trait]
    impl RerankingModel for TestRerankingModel {
        fn provider(&self) -> &str {
            "test"
        }

        fn model_id(&self) -> &str {
            "test-model"
        }

        async fn do_rerank(
            &self,
            _options: RerankOptions,
        ) -> Result<RerankResponse, Box<dyn std::error::Error + Send + Sync>> {
            Ok(RerankResponse {
                ranking: vec![
                    RankingItem {
                        index: 2,
                        relevance_score: 0.95,
                    },
                    RankingItem {
                        index: 0,
                        relevance_score: 0.75,
                    },
                    RankingItem {
                        index: 1,
                        relevance_score: 0.50,
                    },
                ],
                provider_metadata: None,
                warnings: None,
                response: None,
            })
        }
    }

    #[tokio::test]
    async fn test_reranking_model_trait() {
        let model = TestRerankingModel;

        assert_eq!(model.provider(), "test");
        assert_eq!(model.model_id(), "test-model");
        assert_eq!(model.specification_version(), "v3");

        let options = RerankOptions {
            documents: Documents::Text {
                values: vec!["doc1".into(), "doc2".into(), "doc3".into()],
            },
            query: "test query".into(),
            top_n: None,
            abort_signal: None,
            provider_options: None,
            headers: None,
        };

        let response = model.do_rerank(options).await.unwrap();

        assert_eq!(response.ranking.len(), 3);
        assert_eq!(response.ranking[0].index, 2);
        assert_eq!(response.ranking[0].relevance_score, 0.95);
    }
}
