mod common;

use ai_sdk_openai::OpenAIEmbeddingModel;
use ai_sdk_provider::{EmbedOptions, EmbeddingModel};
use common::{load_json_fixture, TestServer};

#[tokio::test]
async fn test_embedding_model_trait() {
    let model = OpenAIEmbeddingModel::new("text-embedding-3-small", "test-key");

    assert_eq!(model.provider(), "openai");
    assert_eq!(model.model_id(), "text-embedding-3-small");
    assert_eq!(model.max_embeddings_per_call().await, Some(2048));
    assert!(model.supports_parallel_calls().await);
}

#[tokio::test]
async fn test_too_many_embeddings() {
    let model = OpenAIEmbeddingModel::new("text-embedding-3-small", "test-key");

    // Create more than 2048 values
    let values: Vec<String> = (0..2049).map(|i| format!("text {}", i)).collect();

    let options = EmbedOptions {
        values,
        provider_options: None,
        headers: None,
    };

    let result = model.do_embed(options).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Too many"));
}

// Fixture-based tests (run without API key)

#[tokio::test]
async fn test_openai_embedding_with_fixture() {
    // Setup mock server
    let test_server = TestServer::new().await;

    // Load fixture
    let fixture = load_json_fixture("embedding-basic-1");

    // Configure mock to return fixture
    test_server
        .mock_json_response("/v1/embeddings", fixture)
        .await;

    // Create model pointing to mock server
    let model = OpenAIEmbeddingModel::new("text-embedding-3-small", "test-key")
        .with_base_url(format!("{}/v1", test_server.base_url));

    let options = EmbedOptions {
        values: vec!["Hello world".into(), "Test embedding".into()],
        provider_options: None,
        headers: None,
    };

    let response = model.do_embed(options).await.unwrap();

    // Verify response structure
    assert_eq!(response.embeddings.len(), 2);
    assert!(!response.embeddings[0].is_empty());
    assert!(!response.embeddings[1].is_empty());
    assert!(response.usage.is_some());

    // Snapshot the response structure (but not the actual embeddings as they're large)
    if let Some(usage) = &response.usage {
        insta::assert_json_snapshot!(usage, @r###"
        {
          "tokens": 12
        }
        "###);
    }

    // Verify embedding dimensions (8 dimensions in our fixture)
    assert_eq!(response.embeddings[0].len(), 8);
    assert_eq!(response.embeddings[1].len(), 8);
}

#[tokio::test]
async fn test_openai_embedding_with_dimensions_fixture() {
    use ai_sdk_provider::JsonValue;
    use std::collections::HashMap;

    // Setup mock server
    let test_server = TestServer::new().await;

    // Load fixture with 512 dimensions
    let fixture = load_json_fixture("embedding-512-dims-1");

    // Configure mock to return fixture
    test_server
        .mock_json_response("/v1/embeddings", fixture)
        .await;

    // Create model pointing to mock server
    let model = OpenAIEmbeddingModel::new("text-embedding-3-small", "test-key")
        .with_base_url(format!("{}/v1", test_server.base_url));

    // Create provider options with dimensions
    let mut openai_opts = HashMap::new();
    openai_opts.insert(
        "dimensions".to_string(),
        JsonValue::Number(serde_json::Number::from(512)),
    );

    let mut provider_options = HashMap::new();
    provider_options.insert("openai".to_string(), openai_opts);

    let options = EmbedOptions {
        values: vec!["Hello world".into()],
        provider_options: Some(provider_options),
        headers: None,
    };

    let response = model.do_embed(options).await.unwrap();

    // Verify response structure
    assert_eq!(response.embeddings.len(), 1);

    // With dimensions=512, the embedding should have 512 dimensions
    assert_eq!(response.embeddings[0].len(), 512);

    // Verify usage
    assert!(response.usage.is_some());
    if let Some(usage) = &response.usage {
        assert_eq!(usage.tokens, 5);
    }
}
