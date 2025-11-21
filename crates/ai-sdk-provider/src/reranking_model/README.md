# Reranking Model Interface

This module defines the trait interface for reranking models. Reranking models
take a query and a list of documents and reorder them by relevance.

## Example Usage

```rust
use ai_sdk_provider::{RerankingModel, RerankOptions, Documents};

async fn rerank_documents<M: RerankingModel>(
    model: &M,
    query: &str,
    docs: Vec<String>
) -> Result<Vec<(usize, f64)>, Box<dyn std::error::Error>> {
    let options = RerankOptions {
        documents: Documents::Text { values: docs },
        query: query.to_string(),
        top_n: Some(5),
        abort_signal: None,
        provider_options: None,
        headers: None,
    };

    let response = model.do_rerank(options).await?;

    Ok(response.ranking.into_iter()
        .map(|r| (r.index, r.relevance_score))
        .collect())
}
```

## Provider Implementations

Currently, no providers are implemented in this crate. Provider implementations
should be added in separate crates (e.g., `ai-sdk-cohere` for Cohere reranking).

## Future Work

- Implement Cohere reranking provider
- Add support for additional reranking models
- Add streaming reranking for large document sets
