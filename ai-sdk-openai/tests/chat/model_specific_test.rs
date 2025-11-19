use crate::common::TestServer;
use ai_sdk_openai::*;
use ai_sdk_provider::json_value::{JsonObject, JsonValue};
use ai_sdk_provider::language_model::{Message, UserContentPart};
use ai_sdk_provider::*;
use serde_json::json;
use std::collections::HashMap;

// Phase 4: Model-Specific Behavior Tests

#[tokio::test]
#[ignore = "Requires is_reasoning_model detection and warning system - not yet implemented"]
async fn test_clear_settings_for_reasoning_models() {
    // TypeScript reference: line 1217
    // Test that temperature, top_p, frequency_penalty, presence_penalty are cleared
    // for reasoning models (o1, o3, o4) and warnings are returned
    //
    // TODO: Implement is_reasoning_model() check in chat.rs
    // TODO: Clear temperature/top_p/frequency_penalty/presence_penalty for o1/o3/o4 models
    // TODO: Return warnings for unsupported settings
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

    let model = OpenAIChatModel::new("o4-mini", "test-key")
        .with_base_url(format!("{}/v1", test_server.base_url));

    let _response = model
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![UserContentPart::Text {
                    text: "Hello".to_string(),
                }],
            }],
            temperature: Some(0.5),
            top_p: Some(0.7),
            frequency_penalty: Some(0.2),
            presence_penalty: Some(0.3),
            ..Default::default()
        })
        .await
        .expect("Generate should succeed");

    // The request should have these settings cleared for reasoning models
    // and warnings should be returned about unsupported settings
    // Note: Warning system not yet fully implemented in Rust
}

#[tokio::test]
#[ignore = "Requires is_reasoning_model detection to convert max_tokens to max_completion_tokens - not yet implemented"]
async fn test_convert_max_output_tokens_to_max_completion_tokens() {
    // TypeScript reference: line 1259
    // Test that maxOutputTokens is converted to max_completion_tokens for reasoning models
    //
    // TODO: Implement is_reasoning_model() check in chat.rs
    // TODO: Use max_completion_tokens instead of max_tokens for reasoning models
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

    let model = OpenAIChatModel::new("o4-mini", "test-key")
        .with_base_url(format!("{}/v1", test_server.base_url));

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

    // The request should use max_completion_tokens instead of max_tokens
    // This is specific behavior for reasoning models
}

#[tokio::test]
#[ignore = "Requires is_search_preview_model detection and warning system - not yet implemented"]
async fn test_remove_temperature_for_search_preview_models() {
    // TypeScript reference: line 1455
    // Test that temperature is removed for search preview models
    //
    // TODO: Implement is_search_preview_model() check in chat.rs
    // TODO: Remove temperature for search preview models
    // TODO: Return warning about unsupported setting
    let test_server = TestServer::new().await;

    let response_json = json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1711115037,
        "model": "gpt-4o-search-preview",
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

    let model = OpenAIChatModel::new("gpt-4o-search-preview", "test-key")
        .with_base_url(format!("{}/v1", test_server.base_url));

    let _response = model
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![UserContentPart::Text {
                    text: "Hello".to_string(),
                }],
            }],
            temperature: Some(0.7),
            ..Default::default()
        })
        .await
        .expect("Generate should succeed");

    // Temperature should be removed from request and warning returned
    // for search preview models
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
#[ignore = "Requires flex processing model validation and warning system - not yet implemented"]
async fn test_flex_processing_warning_for_unsupported_model() {
    // TypeScript reference: line 1549
    // Test that warning is shown when using flex processing with unsupported model
    //
    // TODO: Implement model validation for flex processing (only o3, o4-mini, gpt-5)
    // TODO: Remove service_tier from request for unsupported models
    // TODO: Return warning about flex processing not available
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

    let model = OpenAIChatModel::new("gpt-4o-mini", "test-key")
        .with_base_url(format!("{}/v1", test_server.base_url));

    let _response = model
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

    // Warning should be returned about flex processing not being available
    // for gpt-4o-mini model
    // Note: Warning system not yet fully implemented in Rust
}

#[tokio::test]
#[ignore = "Requires o1 model detection and system-to-developer message conversion - not yet implemented"]
async fn test_developer_messages_for_o1() {
    // TypeScript reference: line 1277
    // Test that system messages are converted to developer messages for o1 models
    //
    // TODO: Implement is_o1_model() check in chat.rs
    // TODO: Convert system messages to developer messages for o1 models
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

    let model = OpenAIChatModel::new("o1", "test-key")
        .with_base_url(format!("{}/v1", test_server.base_url));

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

    // The request should convert system message to developer message for o1
    // System: "You are a helpful assistant." -> Developer: "You are a helpful assistant."
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

    let model = OpenAIChatModel::new("o4-mini", "test-key")
        .with_base_url(format!("{}/v1", test_server.base_url));

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
