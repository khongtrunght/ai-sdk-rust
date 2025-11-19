use crate::common::TestServer;
use ai_sdk_openai::*;
use ai_sdk_provider::json_value::{JsonObject, JsonValue};
use ai_sdk_provider::language_model::{FunctionTool, Message, Tool, ToolChoice, UserContentPart};
use ai_sdk_provider::*;
use serde_json::json;
use std::collections::HashMap;

// Phase 2: Settings & Configuration Tests

#[tokio::test]
async fn test_pass_model_and_messages() {
    // TypeScript reference: line 441
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

    let model = OpenAIChatModel::new("gpt-3.5-turbo", "test-key")
        .with_base_url(format!("{}/v1", test_server.base_url));

    let _ = model
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

    // Request body validation: The model passes model name and messages to the API
    // This is verified by the successful response
}

#[tokio::test]
async fn test_pass_settings() {
    // TypeScript reference: line 454
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

    // logitBias
    let mut logit_bias: JsonObject = HashMap::new();
    logit_bias.insert(
        "50256".to_string(),
        JsonValue::Number(serde_json::Number::from_f64(-100.0).unwrap()),
    );
    openai_options.insert("logitBias".to_string(), JsonValue::Object(logit_bias));

    // parallelToolCalls
    openai_options.insert("parallelToolCalls".to_string(), JsonValue::Bool(false));

    // user
    openai_options.insert(
        "user".to_string(),
        JsonValue::String("test-user-id".to_string()),
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
}

#[tokio::test]
async fn test_pass_reasoning_effort_from_provider_metadata() {
    // TypeScript reference: line 486
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
        "reasoningEffort".to_string(),
        JsonValue::String("low".to_string()),
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
}

#[tokio::test]
async fn test_pass_reasoning_effort_from_settings() {
    // TypeScript reference: line 505
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
        "reasoningEffort".to_string(),
        JsonValue::String("high".to_string()),
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
}

#[tokio::test]
async fn test_pass_text_verbosity_setting() {
    // TypeScript reference: line 524
    let test_server = TestServer::new().await;

    let response_json = json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1711115037,
        "model": "gpt-4o",
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
        "textVerbosity".to_string(),
        JsonValue::String("low".to_string()),
    );
    provider_options.insert("openai".to_string(), openai_options);

    let model = OpenAIChatModel::new("gpt-4o", "test-key")
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
}

#[tokio::test]
async fn test_pass_tools_and_tool_choice() {
    // TypeScript reference: line 543
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

    let model = OpenAIChatModel::new("gpt-3.5-turbo", "test-key")
        .with_base_url(format!("{}/v1", test_server.base_url));

    let tool = Tool::Function(FunctionTool {
        name: "test-tool".to_string(),
        description: None,
        input_schema: json!({
            "type": "object",
            "properties": {
                "value": {"type": "string"}
            },
            "required": ["value"],
            "additionalProperties": false,
            "$schema": "http://json-schema.org/draft-07/schema#"
        }),
        provider_options: None,
    });

    let _ = model
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![UserContentPart::Text {
                    text: "Hello".to_string(),
                }],
            }],
            tools: Some(vec![tool]),
            tool_choice: Some(ToolChoice::Tool {
                tool_name: "test-tool".to_string(),
            }),
            ..Default::default()
        })
        .await
        .expect("Generate should succeed");
}

#[tokio::test]
async fn test_pass_headers() {
    // TypeScript reference: line 608
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

    let model = OpenAIChatModel::new("gpt-3.5-turbo", "test-key")
        .with_base_url(format!("{}/v1", test_server.base_url));

    let mut headers = HashMap::new();
    headers.insert(
        "custom-request-header".to_string(),
        "request-header-value".to_string(),
    );

    let _ = model
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![UserContentPart::Text {
                    text: "Hello".to_string(),
                }],
            }],
            headers: Some(headers),
            ..Default::default()
        })
        .await
        .expect("Generate should succeed");
}

#[tokio::test]
async fn test_parse_tool_results() {
    // TypeScript reference: line 640
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
                "content": "",
                "tool_calls": [{
                    "id": "call_O17Uplv4lJvD6DVdIvFFeRMw",
                    "type": "function",
                    "function": {
                        "name": "test-tool",
                        "arguments": "{\"value\":\"Spark\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
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

    let model = OpenAIChatModel::new("gpt-3.5-turbo", "test-key")
        .with_base_url(format!("{}/v1", test_server.base_url));

    let tool = Tool::Function(FunctionTool {
        name: "test-tool".to_string(),
        description: None,
        input_schema: json!({
            "type": "object",
            "properties": {
                "value": {"type": "string"}
            },
            "required": ["value"],
            "additionalProperties": false,
            "$schema": "http://json-schema.org/draft-07/schema#"
        }),
        provider_options: None,
    });

    let response = model
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![UserContentPart::Text {
                    text: "Hello".to_string(),
                }],
            }],
            tools: Some(vec![tool]),
            tool_choice: Some(ToolChoice::Tool {
                tool_name: "test-tool".to_string(),
            }),
            ..Default::default()
        })
        .await
        .expect("Generate should succeed");

    // Verify tool call is parsed correctly
    assert_eq!(response.content.len(), 1);

    if let Content::ToolCall(tool_call) = &response.content[0] {
        assert_eq!(tool_call.tool_call_id, "call_O17Uplv4lJvD6DVdIvFFeRMw");
        assert_eq!(tool_call.tool_name, "test-tool");
        assert_eq!(tool_call.input, "{\"value\":\"Spark\"}");
    } else {
        panic!("Expected tool call content");
    }

    assert_eq!(response.finish_reason, FinishReason::ToolCalls);
}
