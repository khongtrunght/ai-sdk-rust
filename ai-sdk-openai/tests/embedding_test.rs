use ai_sdk_openai::OpenAIEmbeddingModel;
use ai_sdk_provider::{EmbedOptions, EmbeddingModel};

#[tokio::test]
#[ignore] // Requires API key
async fn test_openai_embedding() {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");

    let model = OpenAIEmbeddingModel::new("text-embedding-3-small", api_key);

    let options = EmbedOptions {
        values: vec!["Hello world".into(), "Test embedding".into()],
        provider_options: None,
        headers: None,
    };

    let response = model.do_embed(options).await.unwrap();

    assert_eq!(response.embeddings.len(), 2);
    assert!(response.embeddings[0].len() > 0);
    assert!(response.usage.is_some());
}

#[tokio::test]
#[ignore] // Requires API key
async fn test_openai_embedding_with_dimensions() {
    use std::collections::HashMap;
    use ai_sdk_provider::JsonValue;

    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");

    let model = OpenAIEmbeddingModel::new("text-embedding-3-small", api_key);

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

    assert_eq!(response.embeddings.len(), 1);
    // With dimensions=512, the embedding should have 512 dimensions
    assert_eq!(response.embeddings[0].len(), 512);
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
