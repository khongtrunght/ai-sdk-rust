use ai_sdk_provider::{Documents, RankingItem, RerankOptions, RerankResponse, RerankingModel};
use async_trait::async_trait;

// Example dummy implementation for testing
struct DummyRerankingModel;

#[async_trait]
impl RerankingModel for DummyRerankingModel {
    fn provider(&self) -> &str {
        "dummy"
    }

    fn model_id(&self) -> &str {
        "dummy-rerank-1"
    }

    async fn do_rerank(
        &self,
        options: RerankOptions,
    ) -> Result<RerankResponse, Box<dyn std::error::Error + Send + Sync>> {
        let doc_count = match &options.documents {
            Documents::Text { values } => values.len(),
            Documents::Object { values } => values.len(),
        };

        // Simple dummy ranking: reverse order with scores 1.0 to 0.0
        let ranking: Vec<RankingItem> = (0..doc_count)
            .rev()
            .enumerate()
            .map(|(i, idx)| RankingItem {
                index: idx,
                relevance_score: 1.0 - (i as f64 / doc_count as f64),
            })
            .collect();

        Ok(RerankResponse {
            ranking,
            provider_metadata: None,
            warnings: None,
            response: None,
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model = DummyRerankingModel;

    let docs = vec![
        "The quick brown fox".to_string(),
        "Jumps over the lazy dog".to_string(),
        "Machine learning is fascinating".to_string(),
    ];

    let options = RerankOptions {
        documents: Documents::Text { values: docs },
        query: "animals".to_string(),
        top_n: Some(2),
        abort_signal: None,
        provider_options: None,
        headers: None,
    };

    let response = model.do_rerank(options).await?;

    println!("Reranked results:");
    for (rank, item) in response.ranking.iter().enumerate() {
        println!(
            "  {}. Document #{} (score: {:.3})",
            rank + 1,
            item.index,
            item.relevance_score
        );
    }

    Ok(())
}
