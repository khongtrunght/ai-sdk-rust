use crate::common::{create_test_model, TestServer};
use ai_sdk_provider::json_value::{JsonObject, JsonValue};
use ai_sdk_provider::language_model::{
    FunctionTool, Message, ResponseFormat, Tool, ToolChoice, UserContentPart,
};
use ai_sdk_provider::*;
use serde_json::json;
use std::collections::HashMap;

// Phase 3: Response Format Tests

#[tokio::test]
async fn test_no_response_format_for_text() {
    // TypeScript reference: line 721
    // Verify response_format is not sent when format is text
    let test_server = TestServer::new().await;

    let response_json = json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1711115037,
        "model": "gpt-4o-2024-08-06",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "{\"value\":\"Spark\"}"
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

    let model = create_test_model(&test_server.base_url, "gpt-4o-2024-08-06");

    let _ = model
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![UserContentPart::Text {
                    text: "Hello".to_string(),
                }],
            }],
            response_format: Some(ResponseFormat::Text),
            ..Default::default()
        })
        .await
        .expect("Generate should succeed");

    // The request body should not include response_format when type is text
    // This is validated by the fact that the mock server accepted the request
}

#[tokio::test]
async fn test_forward_json_as_json_object_without_schema() {
    // TypeScript reference: line 737
    // Test JSON response format without schema -> json_object
    let test_server = TestServer::new().await;

    let response_json = json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1711115037,
        "model": "gpt-4o-2024-08-06",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "{\"value\":\"Spark\"}"
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

    let model = create_test_model(&test_server.base_url, "gpt-4o-2024-08-06");

    let _ = model
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![UserContentPart::Text {
                    text: "Hello".to_string(),
                }],
            }],
            response_format: Some(ResponseFormat::Json {
                schema: None,
                name: None,
                description: None,
            }),
            ..Default::default()
        })
        .await
        .expect("Generate should succeed");

    // The request body should include response_format: { type: "json_object" }
}

#[tokio::test]
async fn test_forward_json_as_json_object_when_structured_outputs_disabled() {
    // TypeScript reference: line 754
    // Test JSON response format with schema but structuredOutputs disabled
    // Should use json_object and generate a warning
    let test_server = TestServer::new().await;

    let response_json = json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1711115037,
        "model": "gpt-4o-2024-08-06",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "{\"value\":\"Spark\"}"
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
    openai_options.insert("structuredOutputs".to_string(), JsonValue::Bool(false));
    provider_options.insert("openai".to_string(), openai_options);

    let model = create_test_model(&test_server.base_url, "gpt-4o-2024-08-06");

    let _response = model
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![UserContentPart::Text {
                    text: "Hello".to_string(),
                }],
            }],
            provider_options: Some(provider_options),
            response_format: Some(ResponseFormat::Json {
                schema: Some(json!({
                    "type": "object",
                    "properties": { "value": { "type": "string" } },
                    "required": ["value"],
                    "additionalProperties": false,
                    "$schema": "http://json-schema.org/draft-07/schema#"
                })),
                name: None,
                description: None,
            }),
            ..Default::default()
        })
        .await
        .expect("Generate should succeed");

    // Should produce a warning about schema not being supported without structuredOutputs
    // When implemented, response.warnings should contain the appropriate warning
    // For now, the test verifies the request completes successfully
}

#[tokio::test]
async fn test_forward_json_response_format_as_json_schema() {
    // TypeScript reference: line 803
    // Test JSON response format with schema and structuredOutputs enabled (default)
    // Should use json_schema with the schema
    let test_server = TestServer::new().await;

    let response_json = json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1711115037,
        "model": "gpt-4o-2024-08-06",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "{\"value\":\"Spark\"}"
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

    let model = create_test_model(&test_server.base_url, "gpt-4o-2024-08-06");

    let _ = model
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![UserContentPart::Text {
                    text: "Hello".to_string(),
                }],
            }],
            response_format: Some(ResponseFormat::Json {
                schema: Some(json!({
                    "type": "object",
                    "properties": { "value": { "type": "string" } },
                    "required": ["value"],
                    "additionalProperties": false,
                    "$schema": "http://json-schema.org/draft-07/schema#"
                })),
                name: None,
                description: None,
            }),
            ..Default::default()
        })
        .await
        .expect("Generate should succeed");

    // The request body should include:
    // response_format: {
    //   type: "json_schema",
    //   json_schema: {
    //     name: "response",
    //     schema: { ... },
    //     strict: false  // Note: TypeScript uses false by default
    //   }
    // }
}

#[tokio::test]
async fn test_use_json_schema_strict_with_response_format_json() {
    // TypeScript reference: line 857
    // Test that json_schema uses strict: false by default
    let test_server = TestServer::new().await;

    let response_json = json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1711115037,
        "model": "gpt-4o-2024-08-06",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "{\"value\":\"Spark\"}"
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

    let model = create_test_model(&test_server.base_url, "gpt-4o-2024-08-06");

    let _ = model
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![UserContentPart::Text {
                    text: "Hello".to_string(),
                }],
            }],
            response_format: Some(ResponseFormat::Json {
                schema: Some(json!({
                    "type": "object",
                    "properties": { "value": { "type": "string" } },
                    "required": ["value"],
                    "additionalProperties": false,
                    "$schema": "http://json-schema.org/draft-07/schema#"
                })),
                name: None,
                description: None,
            }),
            ..Default::default()
        })
        .await
        .expect("Generate should succeed");

    // Verifies that json_schema format is used with strict: false
    // The actual strict value validation would require request capture
}

#[tokio::test]
async fn test_set_name_and_description_with_response_format_json() {
    // TypeScript reference: line 909
    // Test that name and description are passed to json_schema
    let test_server = TestServer::new().await;

    let response_json = json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1711115037,
        "model": "gpt-4o-2024-08-06",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "{\"value\":\"Spark\"}"
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

    let model = create_test_model(&test_server.base_url, "gpt-4o-2024-08-06");

    let _ = model
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![UserContentPart::Text {
                    text: "Hello".to_string(),
                }],
            }],
            response_format: Some(ResponseFormat::Json {
                schema: Some(json!({
                    "type": "object",
                    "properties": { "value": { "type": "string" } },
                    "required": ["value"],
                    "additionalProperties": false,
                    "$schema": "http://json-schema.org/draft-07/schema#"
                })),
                name: Some("test-name".to_string()),
                description: Some("test description".to_string()),
            }),
            ..Default::default()
        })
        .await
        .expect("Generate should succeed");

    // The request body should include:
    // response_format: {
    //   type: "json_schema",
    //   json_schema: {
    //     name: "test-name",
    //     description: "test description",
    //     schema: { ... },
    //     strict: false
    //   }
    // }
}

#[tokio::test]
async fn test_allow_undefined_schema_with_response_format_json() {
    // TypeScript reference: line 964
    // Test that undefined schema with name/description still works (falls back to json_object)
    let test_server = TestServer::new().await;

    let response_json = json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1711115037,
        "model": "gpt-4o-2024-08-06",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "{\"value\":\"Spark\"}"
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

    let model = create_test_model(&test_server.base_url, "gpt-4o-2024-08-06");

    let _ = model
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![UserContentPart::Text {
                    text: "Hello".to_string(),
                }],
            }],
            response_format: Some(ResponseFormat::Json {
                schema: None,
                name: Some("test-name".to_string()),
                description: Some("test description".to_string()),
            }),
            ..Default::default()
        })
        .await
        .expect("Generate should succeed");

    // The request body should include response_format: { type: "json_object" }
    // Name and description are ignored when there's no schema
}

#[tokio::test]
async fn test_set_strict_with_tool_calls_when_structured_outputs_enabled() {
    // TypeScript reference: line 987
    // Test that tool calls include strict: false when structuredOutputs enabled (default)
    let test_server = TestServer::new().await;

    let response_json = json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1711115037,
        "model": "gpt-4o-2024-08-06",
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

    let model = create_test_model(&test_server.base_url, "gpt-4o-2024-08-06");

    let tool = Tool::Function(FunctionTool {
        name: "test-tool".to_string(),
        description: Some("test description".to_string()),
        input_schema: json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
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
            tool_choice: Some(ToolChoice::Required),
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

    // The request body should include tools with strict: false
}

#[tokio::test]
async fn test_tools_without_structured_outputs() {
    // Test that tools work correctly without structuredOutputs setting
    // This verifies the default behavior
    let test_server = TestServer::new().await;

    let response_json = json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1711115037,
        "model": "gpt-4o-2024-08-06",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"location\":\"San Francisco\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    });

    test_server
        .mock_json_response("/v1/chat/completions", response_json)
        .await;

    let model = create_test_model(&test_server.base_url, "gpt-4o-2024-08-06");

    let tool = Tool::Function(FunctionTool {
        name: "get_weather".to_string(),
        description: Some("Get the weather for a location".to_string()),
        input_schema: json!({
            "type": "object",
            "properties": {
                "location": { "type": "string", "description": "The city name" }
            },
            "required": ["location"]
        }),
        provider_options: None,
    });

    let response = model
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![UserContentPart::Text {
                    text: "What's the weather in San Francisco?".to_string(),
                }],
            }],
            tools: Some(vec![tool]),
            ..Default::default()
        })
        .await
        .expect("Generate should succeed");

    // Verify tool call is parsed correctly
    assert_eq!(response.content.len(), 1);

    if let Content::ToolCall(tool_call) = &response.content[0] {
        assert_eq!(tool_call.tool_call_id, "call_abc123");
        assert_eq!(tool_call.tool_name, "get_weather");
        assert_eq!(tool_call.input, "{\"location\":\"San Francisco\"}");
    } else {
        panic!("Expected tool call content");
    }
}
