//! Integration tests for embedding APIs
//!
//! These tests require OPENAI_API_KEY environment variable to be set.
//! Run with: cargo test --package ai-sdk-core --test embedding_test -- --ignored

use ai_sdk_core::{embed, embed_many};
use ai_sdk_openai::openai_embedding;

#[tokio::test]
#[ignore] // Requires OPENAI_API_KEY
async fn test_embed_single() {
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set");

    let result = embed()
        .model(openai_embedding("text-embedding-3-small", api_key))
        .value("Test text for embedding".to_string())
        .execute()
        .await
        .expect("Embedding should succeed");

    // text-embedding-3-small has 1536 dimensions
    assert_eq!(result.embedding().len(), 1536);
    assert!(result.usage().is_some());
    assert!(result.usage().unwrap().tokens > 0);
    assert_eq!(result.value(), "Test text for embedding");
}

#[tokio::test]
#[ignore] // Requires OPENAI_API_KEY
async fn test_embed_many_single_call() {
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set");

    let texts = vec![
        "First text".to_string(),
        "Second text".to_string(),
        "Third text".to_string(),
    ];

    let result = embed_many()
        .model(openai_embedding("text-embedding-3-small", api_key))
        .values(texts.clone())
        .execute()
        .await
        .expect("Embedding should succeed");

    assert_eq!(result.embeddings().len(), 3);
    assert_eq!(result.values().len(), 3);

    // Verify all embeddings have correct dimensions
    for embedding in result.embeddings() {
        assert_eq!(embedding.len(), 1536);
    }

    assert!(result.usage().tokens > 0);
}

#[tokio::test]
#[ignore] // Requires OPENAI_API_KEY
async fn test_embed_many_with_batching() {
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set");

    // Create more texts to trigger batching
    let texts: Vec<String> = (0..15).map(|i| format!("Test text number {}", i)).collect();

    let result = embed_many()
        .model(openai_embedding("text-embedding-3-small", api_key))
        .values(texts.clone())
        .max_parallel_calls(3)
        .execute()
        .await
        .expect("Embedding should succeed");

    assert_eq!(result.embeddings().len(), 15);
    assert_eq!(result.values().len(), 15);

    // Verify all embeddings have correct dimensions
    for embedding in result.embeddings() {
        assert_eq!(embedding.len(), 1536);
    }

    assert!(result.usage().tokens > 0);
}

#[tokio::test]
#[ignore] // Requires OPENAI_API_KEY
async fn test_embed_many_preserves_order() {
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set");

    let texts = vec![
        "First".to_string(),
        "Second".to_string(),
        "Third".to_string(),
        "Fourth".to_string(),
    ];

    let result = embed_many()
        .model(openai_embedding("text-embedding-3-small", api_key))
        .values(texts.clone())
        .max_parallel_calls(2) // Use parallel execution
        .execute()
        .await
        .expect("Embedding should succeed");

    // Verify order is preserved
    for (i, (value, _embedding)) in result.iter().enumerate() {
        assert_eq!(value, &texts[i], "Order should be preserved at index {}", i);
    }
}

#[tokio::test]
#[ignore] // Requires OPENAI_API_KEY
async fn test_embed_many_empty() {
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set");

    let result = embed_many()
        .model(openai_embedding("text-embedding-3-small", api_key))
        .values(Vec::<String>::new())
        .execute()
        .await
        .expect("Embedding should succeed");

    assert_eq!(result.embeddings().len(), 0);
    assert_eq!(result.values().len(), 0);
    assert_eq!(result.usage().tokens, 0);
}

#[tokio::test]
async fn test_embed_missing_model() {
    let result = embed::<String>().value("test".to_string()).execute().await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, ai_sdk_core::EmbedError::MissingModel));
}

#[tokio::test]
async fn test_embed_missing_value() {
    let api_key = "test-key";

    let result = embed()
        .model(openai_embedding("text-embedding-3-small", api_key))
        .execute()
        .await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, ai_sdk_core::EmbedError::MissingValue));
}

#[tokio::test]
async fn test_embed_many_missing_model() {
    let result = embed_many::<String>()
        .values(vec!["test".to_string()])
        .execute()
        .await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, ai_sdk_core::EmbedError::MissingModel));
}
