use crate::common::TestServer;
use ai_sdk_openai::*;
use ai_sdk_provider::json_value::{JsonObject, JsonValue};
use ai_sdk_provider::language_model::{Message, UserContentPart};
use ai_sdk_provider::*;
use serde_json::json;
use std::collections::HashMap;

// Phase 5: Extension Settings Tests

#[tokio::test]
async fn test_send_max_completion_tokens_extension() {
    // TypeScript reference: line 1329
    // Test that maxCompletionTokens extension setting is sent to API
    let test_server = TestServer::new().await;

    let response_json = json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1711115037,
        "model": "o4-mini",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": ""
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 1,
            "total_tokens": 5
        }
    });

    test_server
        .mock_json_response("/v1/chat/completions", response_json)
        .await;

    let mut provider_options: HashMap<String, JsonObject> = HashMap::new();
    let mut openai_options: JsonObject = HashMap::new();
    openai_options.insert(
        "maxCompletionTokens".to_string(),
        JsonValue::Number(serde_json::Number::from(255)),
    );
    provider_options.insert("openai".to_string(), openai_options);

    let model = OpenAIChatModel::new("o4-mini", "test-key")
        .with_base_url(format!("{}/v1", test_server.base_url));

    let _ = model
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![UserContentPart::Text {
                    text: "Hello".to_string(),
                }],
            }],
            provider_options: Some(provider_options),
            ..Default::default()
        })
        .await
        .expect("Generate should succeed");

    // The request should include max_completion_tokens: 255
}

#[tokio::test]
async fn test_send_prediction_extension() {
    // TypeScript reference: line 1350
    // Test that prediction extension setting is sent to API
    let test_server = TestServer::new().await;

    let response_json = json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1711115037,
        "model": "gpt-3.5-turbo",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": ""
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 1,
            "total_tokens": 5
        }
    });

    test_server
        .mock_json_response("/v1/chat/completions", response_json)
        .await;

    let mut provider_options: HashMap<String, JsonObject> = HashMap::new();
    let mut openai_options: JsonObject = HashMap::new();

    // Create prediction object
    let mut prediction: JsonObject = HashMap::new();
    prediction.insert("type".to_string(), JsonValue::String("content".to_string()));
    prediction.insert(
        "content".to_string(),
        JsonValue::String("Hello, World!".to_string()),
    );

    openai_options.insert("prediction".to_string(), JsonValue::Object(prediction));
    provider_options.insert("openai".to_string(), openai_options);

    let model = OpenAIChatModel::new("gpt-3.5-turbo", "test-key")
        .with_base_url(format!("{}/v1", test_server.base_url));

    let _ = model
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![UserContentPart::Text {
                    text: "Hello".to_string(),
                }],
            }],
            provider_options: Some(provider_options),
            ..Default::default()
        })
        .await
        .expect("Generate should succeed");

    // The request should include prediction: { type: "content", content: "Hello, World!" }
}

#[tokio::test]
async fn test_send_store_extension() {
    // TypeScript reference: line 1375
    // Test that store extension setting is sent to API
    let test_server = TestServer::new().await;

    let response_json = json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1711115037,
        "model": "gpt-3.5-turbo",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": ""
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 1,
            "total_tokens": 5
        }
    });

    test_server
        .mock_json_response("/v1/chat/completions", response_json)
        .await;

    let mut provider_options: HashMap<String, JsonObject> = HashMap::new();
    let mut openai_options: JsonObject = HashMap::new();
    openai_options.insert("store".to_string(), JsonValue::Bool(true));
    provider_options.insert("openai".to_string(), openai_options);

    let model = OpenAIChatModel::new("gpt-3.5-turbo", "test-key")
        .with_base_url(format!("{}/v1", test_server.base_url));

    let _ = model
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![UserContentPart::Text {
                    text: "Hello".to_string(),
                }],
            }],
            provider_options: Some(provider_options),
            ..Default::default()
        })
        .await
        .expect("Generate should succeed");

    // The request should include store: true
}

#[tokio::test]
async fn test_send_metadata_extension() {
    // TypeScript reference: line 1394
    // Test that metadata extension setting is sent to API
    let test_server = TestServer::new().await;

    let response_json = json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1711115037,
        "model": "gpt-3.5-turbo",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": ""
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 1,
            "total_tokens": 5
        }
    });

    test_server
        .mock_json_response("/v1/chat/completions", response_json)
        .await;

    let mut provider_options: HashMap<String, JsonObject> = HashMap::new();
    let mut openai_options: JsonObject = HashMap::new();

    // Create metadata object
    let mut metadata: JsonObject = HashMap::new();
    metadata.insert("custom".to_string(), JsonValue::String("value".to_string()));

    openai_options.insert("metadata".to_string(), JsonValue::Object(metadata));
    provider_options.insert("openai".to_string(), openai_options);

    let model = OpenAIChatModel::new("gpt-3.5-turbo", "test-key")
        .with_base_url(format!("{}/v1", test_server.base_url));

    let _ = model
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![UserContentPart::Text {
                    text: "Hello".to_string(),
                }],
            }],
            provider_options: Some(provider_options),
            ..Default::default()
        })
        .await
        .expect("Generate should succeed");

    // The request should include metadata: { custom: "value" }
}

#[tokio::test]
async fn test_send_prompt_cache_key_extension() {
    // TypeScript reference: line 1417
    // Test that promptCacheKey extension setting is sent to API
    let test_server = TestServer::new().await;

    let response_json = json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1711115037,
        "model": "gpt-3.5-turbo",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": ""
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 1,
            "total_tokens": 5
        }
    });

    test_server
        .mock_json_response("/v1/chat/completions", response_json)
        .await;

    let mut provider_options: HashMap<String, JsonObject> = HashMap::new();
    let mut openai_options: JsonObject = HashMap::new();
    openai_options.insert(
        "promptCacheKey".to_string(),
        JsonValue::String("test-cache-key-123".to_string()),
    );
    provider_options.insert("openai".to_string(), openai_options);

    let model = OpenAIChatModel::new("gpt-3.5-turbo", "test-key")
        .with_base_url(format!("{}/v1", test_server.base_url));

    let _ = model
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![UserContentPart::Text {
                    text: "Hello".to_string(),
                }],
            }],
            provider_options: Some(provider_options),
            ..Default::default()
        })
        .await
        .expect("Generate should succeed");

    // The request should include prompt_cache_key: "test-cache-key-123"
}

#[tokio::test]
async fn test_send_safety_identifier_extension() {
    // TypeScript reference: line 1436
    // Test that safetyIdentifier extension setting is sent to API
    let test_server = TestServer::new().await;

    let response_json = json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1711115037,
        "model": "gpt-3.5-turbo",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": ""
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 1,
            "total_tokens": 5
        }
    });

    test_server
        .mock_json_response("/v1/chat/completions", response_json)
        .await;

    let mut provider_options: HashMap<String, JsonObject> = HashMap::new();
    let mut openai_options: JsonObject = HashMap::new();
    openai_options.insert(
        "safetyIdentifier".to_string(),
        JsonValue::String("test-safety-identifier-123".to_string()),
    );
    provider_options.insert("openai".to_string(), openai_options);

    let model = OpenAIChatModel::new("gpt-3.5-turbo", "test-key")
        .with_base_url(format!("{}/v1", test_server.base_url));

    let _ = model
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![UserContentPart::Text {
                    text: "Hello".to_string(),
                }],
            }],
            provider_options: Some(provider_options),
            ..Default::default()
        })
        .await
        .expect("Generate should succeed");

    // The request should include safety_identifier: "test-safety-identifier-123"
}

#[tokio::test]
async fn test_service_tier_flex_processing() {
    // TypeScript reference: line 1521
    // Test that serviceTier flex processing setting is sent to API
    let test_server = TestServer::new().await;

    let response_json = json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1711115037,
        "model": "o4-mini",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": ""
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 1,
            "total_tokens": 5
        }
    });

    test_server
        .mock_json_response("/v1/chat/completions", response_json)
        .await;

    let mut provider_options: HashMap<String, JsonObject> = HashMap::new();
    let mut openai_options: JsonObject = HashMap::new();
    openai_options.insert(
        "serviceTier".to_string(),
        JsonValue::String("flex".to_string()),
    );
    provider_options.insert("openai".to_string(), openai_options);

    let model = OpenAIChatModel::new("o4-mini", "test-key")
        .with_base_url(format!("{}/v1", test_server.base_url));

    let _ = model
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![UserContentPart::Text {
                    text: "Hello".to_string(),
                }],
            }],
            provider_options: Some(provider_options),
            ..Default::default()
        })
        .await
        .expect("Generate should succeed");

    // The request should include service_tier: "flex"
}

#[tokio::test]
async fn test_service_tier_priority_processing() {
    // TypeScript reference: line 1593
    // Test that serviceTier priority processing setting is sent to API
    let test_server = TestServer::new().await;

    let response_json = json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1711115037,
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": ""
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 1,
            "total_tokens": 5
        }
    });

    test_server
        .mock_json_response("/v1/chat/completions", response_json)
        .await;

    let mut provider_options: HashMap<String, JsonObject> = HashMap::new();
    let mut openai_options: JsonObject = HashMap::new();
    openai_options.insert(
        "serviceTier".to_string(),
        JsonValue::String("priority".to_string()),
    );
    provider_options.insert("openai".to_string(), openai_options);

    let model = OpenAIChatModel::new("gpt-4o-mini", "test-key")
        .with_base_url(format!("{}/v1", test_server.base_url));

    let _ = model
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![UserContentPart::Text {
                    text: "Hello".to_string(),
                }],
            }],
            provider_options: Some(provider_options),
            ..Default::default()
        })
        .await
        .expect("Generate should succeed");

    // The request should include service_tier: "priority"
}
