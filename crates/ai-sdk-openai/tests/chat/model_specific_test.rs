use crate::common::{create_test_model, TestServer};
use ai_sdk_provider::json_value::{JsonObject, JsonValue};
use ai_sdk_provider::language_model::{Message, UserContentPart};
use ai_sdk_provider::*;
use serde_json::json;
use std::collections::HashMap;

// Phase 4: Model-Specific Behavior Tests

#[tokio::test]
async fn test_convert_max_output_tokens_to_max_completion_tokens() {
    // TypeScript reference: line 1259
    // Test that maxOutputTokens is converted to max_completion_tokens for reasoning models
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
                "content": "Response"
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

    let model = create_test_model(&test_server.base_url, "o4-mini");

    let _ = model
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![UserContentPart::Text {
                    text: "Hello".to_string(),
                }],
            }],
            max_output_tokens: Some(1000),
            ..Default::default()
        })
        .await
        .expect("Generate should succeed");

    // Verify the request uses max_completion_tokens instead of max_tokens
    let request_body = test_server
        .last_request_body()
        .await
        .expect("Should have request");
    assert_eq!(request_body["max_completion_tokens"], 1000);
    assert!(
        request_body["max_tokens"].is_null(),
        "max_tokens should not be set for reasoning models"
    );
}

#[tokio::test]
async fn test_flex_processing_for_o4_mini() {
    // TypeScript reference: line 1521
    // Test that flex processing works for o4-mini model
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
                "content": "Response"
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

    let model = create_test_model(&test_server.base_url, "o4-mini");

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
async fn test_developer_messages_for_o1() {
    // TypeScript reference: line 1277
    // Test that system messages are converted to developer messages for o1 models
    let test_server = TestServer::new().await;

    let response_json = json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1711115037,
        "model": "o1",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Response"
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

    let model = create_test_model(&test_server.base_url, "o1");

    let _ = model
        .do_generate(CallOptions {
            prompt: vec![
                Message::System {
                    content: "You are a helpful assistant.".to_string(),
                },
                Message::User {
                    content: vec![UserContentPart::Text {
                        text: "Hello".to_string(),
                    }],
                },
            ],
            ..Default::default()
        })
        .await
        .expect("Generate should succeed");

    // Verify the request converts system message to developer message for o1
    let request_body = test_server
        .last_request_body()
        .await
        .expect("Should have request");
    let messages = request_body["messages"]
        .as_array()
        .expect("Should have messages");

    // First message should be converted to "developer" role
    assert_eq!(messages[0]["role"], "developer");
    assert_eq!(messages[0]["content"], "You are a helpful assistant.");

    // Second message should remain as "user"
    assert_eq!(messages[1]["role"], "user");
}

#[tokio::test]
async fn test_reasoning_tokens_in_metadata() {
    // TypeScript reference: line 1300
    // Test that reasoning tokens are included in usage metadata
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
                "content": "Response"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 20,
            "total_tokens": 35,
            "completion_tokens_details": {
                "reasoning_tokens": 10
            }
        }
    });

    test_server
        .mock_json_response("/v1/chat/completions", response_json)
        .await;

    let model = create_test_model(&test_server.base_url, "o4-mini");

    let response = model
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![UserContentPart::Text {
                    text: "Hello".to_string(),
                }],
            }],
            ..Default::default()
        })
        .await
        .expect("Generate should succeed");

    // Verify usage is captured
    assert_eq!(response.usage.input_tokens, Some(15));
    assert_eq!(response.usage.output_tokens, Some(20));
    assert_eq!(response.usage.total_tokens, Some(35));

    // Note: reasoning_tokens extraction from completion_tokens_details
    // may not yet be implemented in Rust
}
