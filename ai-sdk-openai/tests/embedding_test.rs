mod common;

use ai_sdk_openai::OpenAIEmbeddingModel;
use ai_sdk_provider::{EmbedOptions, EmbeddingModel};
use common::TestServer;
use serde_json::json;

// Dummy embeddings for testing (like TypeScript version)
const DUMMY_EMBEDDINGS: [[f32; 5]; 2] = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]];

/// Helper to create embedding response JSON (like TypeScript's prepareJsonResponse)
fn create_embedding_response(embeddings: &[&[f32]], usage: (u32, u32)) -> serde_json::Value {
    json!({
        "object": "list",
        "data": embeddings.iter().enumerate().map(|(i, embedding)| {
            json!({
                "object": "embedding",
                "index": i,
                "embedding": embedding
            })
        }).collect::<Vec<_>>(),
        "model": "text-embedding-3-small",
        "usage": {
            "prompt_tokens": usage.0,
            "total_tokens": usage.1
        }
    })
}

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

#[tokio::test]
async fn test_openai_embedding_extract_embedding() {
    // Setup mock server
    let test_server = TestServer::new().await;

    // Create response with dummy embeddings
    let response = create_embedding_response(&[&DUMMY_EMBEDDINGS[0], &DUMMY_EMBEDDINGS[1]], (8, 8));

    // Configure mock to return response
    test_server
        .mock_json_response("/v1/embeddings", response)
        .await;

    // Create model pointing to mock server
    let model = OpenAIEmbeddingModel::new("text-embedding-3-small", "test-key")
        .with_base_url(format!("{}/v1", test_server.base_url));

    let options = EmbedOptions {
        values: vec![
            "sunny day at the beach".into(),
            "rainy day in the city".into(),
        ],
        provider_options: None,
        headers: None,
    };

    let result = model.do_embed(options).await.unwrap();

    // Verify embeddings match dummy data
    assert_eq!(result.embeddings.len(), 2);
    assert_eq!(result.embeddings[0], DUMMY_EMBEDDINGS[0].to_vec());
    assert_eq!(result.embeddings[1], DUMMY_EMBEDDINGS[1].to_vec());
}

#[tokio::test]
async fn test_openai_embedding_extract_usage() {
    // Setup mock server
    let test_server = TestServer::new().await;

    // Create response with specific usage
    let response =
        create_embedding_response(&[&DUMMY_EMBEDDINGS[0], &DUMMY_EMBEDDINGS[1]], (20, 20));

    test_server
        .mock_json_response("/v1/embeddings", response)
        .await;

    let model = OpenAIEmbeddingModel::new("text-embedding-3-small", "test-key")
        .with_base_url(format!("{}/v1", test_server.base_url));

    let options = EmbedOptions {
        values: vec!["Hello world".into(), "Test embedding".into()],
        provider_options: None,
        headers: None,
    };

    let result = model.do_embed(options).await.unwrap();

    // Verify usage
    assert!(result.usage.is_some());
    if let Some(usage) = &result.usage {
        assert_eq!(usage.tokens, 20);
    }
}

#[tokio::test]
async fn test_openai_embedding_pass_dimensions() {
    use ai_sdk_provider::JsonValue;
    use std::collections::HashMap;

    // Setup mock server
    let test_server = TestServer::new().await;

    // Create response with 512-dimensional embedding
    let embedding_512: Vec<f32> = (0..512).map(|i| i as f32 / 512.0).collect();
    let response = create_embedding_response(&[&embedding_512], (5, 5));

    test_server
        .mock_json_response("/v1/embeddings", response)
        .await;

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

    let result = model.do_embed(options).await.unwrap();

    // Verify response structure
    assert_eq!(result.embeddings.len(), 1);
    assert_eq!(result.embeddings[0].len(), 512);

    // Verify usage
    assert!(result.usage.is_some());
    if let Some(usage) = &result.usage {
        assert_eq!(usage.tokens, 5);
    }
}

#[tokio::test]
async fn test_openai_embedding_pass_model_and_values() {
    // Setup mock server
    let test_server = TestServer::new().await;

    let response = create_embedding_response(&[&DUMMY_EMBEDDINGS[0], &DUMMY_EMBEDDINGS[1]], (8, 8));

    test_server
        .mock_json_response("/v1/embeddings", response)
        .await;

    let model = OpenAIEmbeddingModel::new("text-embedding-3-large", "test-key")
        .with_base_url(format!("{}/v1", test_server.base_url));

    let test_values = vec![
        "sunny day at the beach".into(),
        "rainy day in the city".into(),
    ];

    let options = EmbedOptions {
        values: test_values.clone(),
        provider_options: None,
        headers: None,
    };

    model.do_embed(options).await.unwrap();

    // Verify request body
    let request_body = test_server
        .last_request_body()
        .await
        .expect("Request body should exist");
    assert_eq!(request_body["model"], "text-embedding-3-large");
    assert_eq!(request_body["input"], json!(test_values));
    assert_eq!(request_body["encoding_format"], "float");
}
