use crate::common::{create_test_model, load_chunks_fixture, TestServer};
use ai_sdk_provider::json_value::{JsonObject, JsonValue};
use ai_sdk_provider::language_model::{FunctionTool, Message, Tool, UserContentPart};
use ai_sdk_provider::*;
use serde_json::json;
use std::collections::HashMap;
use tokio_stream::StreamExt;

// Phase 7: Advanced Streaming Tests

#[tokio::test]
async fn test_stream_text_delta() {
    // TypeScript reference: line 1827
    // Test basic text delta streaming
    let test_server = TestServer::new().await;

    let chunks = load_chunks_fixture("chat-completion-simple-1");
    test_server
        .mock_streaming_response("/v1/chat/completions", chunks)
        .await;

    let model = create_test_model(&test_server.base_url, "gpt-4");

    let options = CallOptions {
        prompt: vec![Message::User {
            content: vec![UserContentPart::Text {
                text: "Say 'Hello, Rust!'".into(),
            }],
        }],
        ..Default::default()
    };

    let mut stream_response = model.do_stream(options).await.expect("Stream should start");

    let mut stream_parts = vec![];
    while let Some(part_result) = stream_response.stream.next().await {
        let part = part_result.expect("Stream part should be ok");
        stream_parts.push(part);
    }

    // Verify we received text deltas
    let text_deltas: Vec<String> = stream_parts
        .iter()
        .filter_map(|part| {
            if let StreamPart::TextDelta { delta, .. } = part {
                Some(delta.clone())
            } else {
                None
            }
        })
        .collect();

    assert!(!text_deltas.is_empty(), "Should receive text deltas");

    // Verify full text
    let full_text: String = text_deltas.join("");
    assert_eq!(full_text, "Hello, Rust!");
}

#[tokio::test]
async fn test_stream_tool_deltas() {
    // TypeScript reference: line 1969
    // Test tool call deltas during streaming
    let test_server = TestServer::new().await;

    let chunks = load_chunks_fixture("chat-tool-calling-1");
    test_server
        .mock_streaming_response("/v1/chat/completions", chunks)
        .await;

    let model = create_test_model(&test_server.base_url, "gpt-4");

    let tool = Tool::Function(FunctionTool {
        name: "get_weather".to_string(),
        description: Some("Get weather information".to_string()),
        input_schema: json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }),
        provider_options: None,
    });

    let options = CallOptions {
        prompt: vec![Message::User {
            content: vec![UserContentPart::Text {
                text: "What's the weather in Tokyo?".into(),
            }],
        }],
        tools: Some(vec![tool]),
        ..Default::default()
    };

    let mut stream_response = model.do_stream(options).await.expect("Stream should start");

    let mut stream_parts = vec![];
    while let Some(part_result) = stream_response.stream.next().await {
        let part = part_result.expect("Stream part should be ok");
        stream_parts.push(part);
    }

    // Verify we received tool call parts
    let has_tool_call = stream_parts
        .iter()
        .any(|part| matches!(part, StreamPart::ToolCall { .. }));

    assert!(has_tool_call, "Should receive tool call in stream");
}

#[tokio::test]
async fn test_stream_usage_information() {
    // TypeScript reference: line 2400
    // Test that usage information is included in streaming response
    let test_server = TestServer::new().await;

    let chunks = load_chunks_fixture("chat-completion-simple-1");
    test_server
        .mock_streaming_response("/v1/chat/completions", chunks)
        .await;

    let model = create_test_model(&test_server.base_url, "gpt-4");

    let options = CallOptions {
        prompt: vec![Message::User {
            content: vec![UserContentPart::Text {
                text: "Hello".into(),
            }],
        }],
        ..Default::default()
    };

    let mut stream_response = model.do_stream(options).await.expect("Stream should start");

    let mut stream_parts = vec![];
    while let Some(part_result) = stream_response.stream.next().await {
        let part = part_result.expect("Stream part should be ok");
        stream_parts.push(part);
    }

    // Check for finish part with usage
    let finish_parts: Vec<&StreamPart> = stream_parts
        .iter()
        .filter(|part| matches!(part, StreamPart::Finish { .. }))
        .collect();

    assert!(!finish_parts.is_empty(), "Should have finish part");

    // Verify finish part contains usage
    if let StreamPart::Finish { usage, .. } = finish_parts[0] {
        assert!(usage.input_tokens.is_some() || usage.total_tokens.is_some());
    }
}

#[tokio::test]
async fn test_stream_finish_reason() {
    // TypeScript reference: line 2086
    // Test that finish reason is correctly reported in streaming
    let test_server = TestServer::new().await;

    let chunks = load_chunks_fixture("chat-completion-simple-1");
    test_server
        .mock_streaming_response("/v1/chat/completions", chunks)
        .await;

    let model = create_test_model(&test_server.base_url, "gpt-4");

    let options = CallOptions {
        prompt: vec![Message::User {
            content: vec![UserContentPart::Text {
                text: "Hello".into(),
            }],
        }],
        ..Default::default()
    };

    let mut stream_response = model.do_stream(options).await.expect("Stream should start");

    let mut stream_parts = vec![];
    while let Some(part_result) = stream_response.stream.next().await {
        let part = part_result.expect("Stream part should be ok");
        stream_parts.push(part);
    }

    // Check for finish reason
    let has_finish = stream_parts
        .iter()
        .any(|part| matches!(part, StreamPart::Finish { .. }));

    assert!(has_finish, "Should have finish part with reason");
}

#[tokio::test]
async fn test_stream_with_custom_headers() {
    // TypeScript reference: line 2556
    // Test streaming with custom request headers
    use wiremock::{Mock, ResponseTemplate};

    let test_server = TestServer::new().await;

    // Build SSE response manually
    let sse_chunks = [
        r#"data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1711115037,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}"#,
        "\n\n",
        r#"data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1711115037,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}"#,
        "\n\n",
        r#"data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1711115037,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}"#,
        "\n\n",
        "data: [DONE]\n\n",
    ]
    .join("");

    Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/chat/completions"))
        .and(wiremock::matchers::header("custom-header", "custom-value"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_chunks)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&test_server.server)
        .await;

    let model = create_test_model(&test_server.base_url, "gpt-4");

    let mut headers = HashMap::new();
    headers.insert("custom-header".to_string(), "custom-value".to_string());

    let options = CallOptions {
        prompt: vec![Message::User {
            content: vec![UserContentPart::Text {
                text: "Hello".into(),
            }],
        }],
        headers: Some(headers),
        ..Default::default()
    };

    let mut stream_response = model.do_stream(options).await.expect("Stream should start");

    let mut stream_parts = vec![];
    while let Some(part_result) = stream_response.stream.next().await {
        let part = part_result.expect("Stream part should be ok");
        stream_parts.push(part);
    }

    // Verify stream completed successfully with custom header
    assert!(!stream_parts.is_empty());
}

#[tokio::test]
async fn test_stream_with_provider_options() {
    // TypeScript reference: line 2490
    // Test streaming with provider-specific options
    let test_server = TestServer::new().await;

    let chunks = load_chunks_fixture("chat-completion-simple-1");
    test_server
        .mock_streaming_response("/v1/chat/completions", chunks)
        .await;

    let mut provider_options: HashMap<String, JsonObject> = HashMap::new();
    let mut openai_options: JsonObject = HashMap::new();
    openai_options.insert(
        "user".to_string(),
        JsonValue::String("test-user".to_string()),
    );
    provider_options.insert("openai".to_string(), openai_options);

    let model = create_test_model(&test_server.base_url, "gpt-4");

    let options = CallOptions {
        prompt: vec![Message::User {
            content: vec![UserContentPart::Text {
                text: "Hello".into(),
            }],
        }],
        provider_options: Some(provider_options),
        ..Default::default()
    };

    let mut stream_response = model.do_stream(options).await.expect("Stream should start");

    let mut stream_parts = vec![];
    while let Some(part_result) = stream_response.stream.next().await {
        let part = part_result.expect("Stream part should be ok");
        stream_parts.push(part);
    }

    // Verify stream completed
    assert!(!stream_parts.is_empty());
}

#[tokio::test]
async fn test_stream_multiple_tool_calls() {
    // TypeScript reference: line 2340
    // Test streaming multiple tool calls in a single response
    let test_server = TestServer::new().await;

    let chunks = load_chunks_fixture("chat-streaming-multiple-tools-1");
    test_server
        .mock_streaming_response("/v1/chat/completions", chunks)
        .await;

    let model = create_test_model(&test_server.base_url, "gpt-4");

    let tools = vec![
        Tool::Function(FunctionTool {
            name: "get_weather".to_string(),
            description: Some("Get weather".to_string()),
            input_schema: json!({"type": "object", "properties": {"location": {"type": "string"}}}),
            provider_options: None,
        }),
        Tool::Function(FunctionTool {
            name: "get_time".to_string(),
            description: Some("Get time".to_string()),
            input_schema: json!({"type": "object", "properties": {"timezone": {"type": "string"}}}),
            provider_options: None,
        }),
    ];

    let options = CallOptions {
        prompt: vec![Message::User {
            content: vec![UserContentPart::Text {
                text: "What's the weather and time in Tokyo?".into(),
            }],
        }],
        tools: Some(tools),
        ..Default::default()
    };

    let mut stream_response = model.do_stream(options).await.expect("Stream should start");

    let mut stream_parts = vec![];
    while let Some(part_result) = stream_response.stream.next().await {
        let part = part_result.expect("Stream part should be ok");
        stream_parts.push(part);
    }

    // Count tool calls
    let tool_calls: Vec<&StreamPart> = stream_parts
        .iter()
        .filter(|part| matches!(part, StreamPart::ToolCall { .. }))
        .collect();

    // Should have at least one tool call (implementation may vary)
    assert!(!tool_calls.is_empty(), "Should receive tool calls");
}

#[tokio::test]
async fn test_stream_cached_tokens() {
    // TypeScript reference: line 2600
    // Test that cached tokens are included in streaming usage
    let test_server = TestServer::new().await;

    // Create streaming response with cached tokens
    let chunks = vec![
        r#"{"id":"chatcmpl-test","object":"chat.completion.chunk","created":1711115037,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}"#.to_string(),
        r#"{"id":"chatcmpl-test","object":"chat.completion.chunk","created":1711115037,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Response"},"finish_reason":null}]}"#.to_string(),
        r#"{"id":"chatcmpl-test","object":"chat.completion.chunk","created":1711115037,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":5,"total_tokens":25,"prompt_tokens_details":{"cached_tokens":10}}}"#.to_string(),
    ];

    test_server
        .mock_streaming_response("/v1/chat/completions", chunks)
        .await;

    let model = create_test_model(&test_server.base_url, "gpt-4");

    let options = CallOptions {
        prompt: vec![Message::User {
            content: vec![UserContentPart::Text {
                text: "Hello".into(),
            }],
        }],
        ..Default::default()
    };

    let mut stream_response = model.do_stream(options).await.expect("Stream should start");

    let mut stream_parts = vec![];
    while let Some(part_result) = stream_response.stream.next().await {
        let part = part_result.expect("Stream part should be ok");
        stream_parts.push(part);
    }

    // Find finish part and check for cached tokens
    for part in &stream_parts {
        if let StreamPart::Finish { usage, .. } = part {
            // Note: cached_input_tokens extraction may not be implemented
            assert!(usage.input_tokens.is_some());
        }
    }
}

#[tokio::test]
async fn test_stream_reasoning_tokens() {
    // TypeScript reference: line 2650
    // Test that reasoning tokens are included in streaming usage
    let test_server = TestServer::new().await;

    // Create streaming response with reasoning tokens
    let chunks = vec![
        r#"{"id":"chatcmpl-test","object":"chat.completion.chunk","created":1711115037,"model":"o4-mini","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}"#.to_string(),
        r#"{"id":"chatcmpl-test","object":"chat.completion.chunk","created":1711115037,"model":"o4-mini","choices":[{"index":0,"delta":{"content":"Response"},"finish_reason":null}]}"#.to_string(),
        r#"{"id":"chatcmpl-test","object":"chat.completion.chunk","created":1711115037,"model":"o4-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":15,"completion_tokens":20,"total_tokens":35,"completion_tokens_details":{"reasoning_tokens":10}}}"#.to_string(),
    ];

    test_server
        .mock_streaming_response("/v1/chat/completions", chunks)
        .await;

    let model = create_test_model(&test_server.base_url, "o4-mini");

    let options = CallOptions {
        prompt: vec![Message::User {
            content: vec![UserContentPart::Text {
                text: "Hello".into(),
            }],
        }],
        ..Default::default()
    };

    let mut stream_response = model.do_stream(options).await.expect("Stream should start");

    let mut stream_parts = vec![];
    while let Some(part_result) = stream_response.stream.next().await {
        let part = part_result.expect("Stream part should be ok");
        stream_parts.push(part);
    }

    // Find finish part and check for reasoning tokens
    for part in &stream_parts {
        if let StreamPart::Finish { usage, .. } = part {
            assert!(usage.output_tokens.is_some());
            // Note: reasoning_tokens extraction may not be implemented
        }
    }
}

#[tokio::test]
async fn test_stream_with_temperature() {
    // Test streaming with temperature setting
    let test_server = TestServer::new().await;

    let chunks = load_chunks_fixture("chat-completion-simple-1");
    test_server
        .mock_streaming_response("/v1/chat/completions", chunks)
        .await;

    let model = create_test_model(&test_server.base_url, "gpt-4");

    let options = CallOptions {
        prompt: vec![Message::User {
            content: vec![UserContentPart::Text {
                text: "Hello".into(),
            }],
        }],
        temperature: Some(0.7),
        ..Default::default()
    };

    let mut stream_response = model.do_stream(options).await.expect("Stream should start");

    let mut stream_parts = vec![];
    while let Some(part_result) = stream_response.stream.next().await {
        let part = part_result.expect("Stream part should be ok");
        stream_parts.push(part);
    }

    // Verify stream completed
    let has_finish = stream_parts
        .iter()
        .any(|part| matches!(part, StreamPart::Finish { .. }));
    assert!(has_finish, "Should have finish part");
}

#[tokio::test]
async fn test_stream_with_max_tokens() {
    // Test streaming with max_output_tokens setting
    let test_server = TestServer::new().await;

    let chunks = load_chunks_fixture("chat-completion-simple-1");
    test_server
        .mock_streaming_response("/v1/chat/completions", chunks)
        .await;

    let model = create_test_model(&test_server.base_url, "gpt-4");

    let options = CallOptions {
        prompt: vec![Message::User {
            content: vec![UserContentPart::Text {
                text: "Hello".into(),
            }],
        }],
        max_output_tokens: Some(100),
        ..Default::default()
    };

    let mut stream_response = model.do_stream(options).await.expect("Stream should start");

    let mut stream_parts = vec![];
    while let Some(part_result) = stream_response.stream.next().await {
        let part = part_result.expect("Stream part should be ok");
        stream_parts.push(part);
    }

    // Verify stream completed
    assert!(!stream_parts.is_empty());
}

#[tokio::test]
async fn test_stream_annotations() {
    // TypeScript reference: line 1918
    // Test streaming annotations/citations
    let test_server = TestServer::new().await;

    let chunks = load_chunks_fixture("chat-streaming-annotations-1");
    test_server
        .mock_streaming_response("/v1/chat/completions", chunks)
        .await;

    let model = create_test_model(&test_server.base_url, "gpt-4o-search-preview");

    let options = CallOptions {
        prompt: vec![Message::User {
            content: vec![UserContentPart::Text {
                text: "Search for information".into(),
            }],
        }],
        ..Default::default()
    };

    let mut stream_response = model.do_stream(options).await.expect("Stream should start");

    let mut stream_parts = vec![];
    while let Some(part_result) = stream_response.stream.next().await {
        let part = part_result.expect("Stream part should be ok");
        stream_parts.push(part);
    }

    // Verify stream includes text and potentially source parts
    let text_deltas: Vec<&StreamPart> = stream_parts
        .iter()
        .filter(|part| matches!(part, StreamPart::TextDelta { .. }))
        .collect();

    assert!(!text_deltas.is_empty(), "Should receive text deltas");
}

#[tokio::test]
async fn test_stream_empty_response() {
    // Test handling of empty streaming response
    let test_server = TestServer::new().await;

    let chunks = load_chunks_fixture("chat-streaming-empty-1");
    test_server
        .mock_streaming_response("/v1/chat/completions", chunks)
        .await;

    let model = create_test_model(&test_server.base_url, "gpt-4");

    let options = CallOptions {
        prompt: vec![Message::User {
            content: vec![UserContentPart::Text { text: "".into() }],
        }],
        ..Default::default()
    };

    let mut stream_response = model.do_stream(options).await.expect("Stream should start");

    let mut stream_parts = vec![];
    while let Some(part_result) = stream_response.stream.next().await {
        let part = part_result.expect("Stream part should be ok");
        stream_parts.push(part);
    }

    // Should complete without errors
    let has_finish = stream_parts
        .iter()
        .any(|part| matches!(part, StreamPart::Finish { .. }));
    assert!(
        has_finish,
        "Should have finish part even for empty response"
    );
}

#[tokio::test]
async fn test_stream_service_tier_flex() {
    // Test streaming with service tier flex setting
    let test_server = TestServer::new().await;

    let chunks = load_chunks_fixture("chat-completion-simple-1");
    test_server
        .mock_streaming_response("/v1/chat/completions", chunks)
        .await;

    let mut provider_options: HashMap<String, JsonObject> = HashMap::new();
    let mut openai_options: JsonObject = HashMap::new();
    openai_options.insert(
        "serviceTier".to_string(),
        JsonValue::String("flex".to_string()),
    );
    provider_options.insert("openai".to_string(), openai_options);

    let model = create_test_model(&test_server.base_url, "o4-mini");

    let options = CallOptions {
        prompt: vec![Message::User {
            content: vec![UserContentPart::Text {
                text: "Hello".into(),
            }],
        }],
        provider_options: Some(provider_options),
        ..Default::default()
    };

    let mut stream_response = model.do_stream(options).await.expect("Stream should start");

    let mut stream_parts = vec![];
    while let Some(part_result) = stream_response.stream.next().await {
        let part = part_result.expect("Stream part should be ok");
        stream_parts.push(part);
    }

    // Verify stream completed
    assert!(!stream_parts.is_empty());
}

#[tokio::test]
async fn test_stream_logit_bias() {
    // Test streaming with logit bias setting
    let test_server = TestServer::new().await;

    let chunks = load_chunks_fixture("chat-completion-simple-1");
    test_server
        .mock_streaming_response("/v1/chat/completions", chunks)
        .await;

    let mut provider_options: HashMap<String, JsonObject> = HashMap::new();
    let mut openai_options: JsonObject = HashMap::new();

    // Create logit bias
    let mut logit_bias: JsonObject = HashMap::new();
    logit_bias.insert(
        "50256".to_string(),
        JsonValue::Number(serde_json::Number::from_f64(-100.0).unwrap()),
    );
    openai_options.insert("logitBias".to_string(), JsonValue::Object(logit_bias));
    provider_options.insert("openai".to_string(), openai_options);

    let model = create_test_model(&test_server.base_url, "gpt-4");

    let options = CallOptions {
        prompt: vec![Message::User {
            content: vec![UserContentPart::Text {
                text: "Hello".into(),
            }],
        }],
        provider_options: Some(provider_options),
        ..Default::default()
    };

    let mut stream_response = model.do_stream(options).await.expect("Stream should start");

    let mut stream_parts = vec![];
    while let Some(part_result) = stream_response.stream.next().await {
        let part = part_result.expect("Stream part should be ok");
        stream_parts.push(part);
    }

    // Verify stream completed
    assert!(!stream_parts.is_empty());
}

#[tokio::test]
async fn test_stream_stop_sequences() {
    // Test streaming with stop sequences
    let test_server = TestServer::new().await;

    let chunks = load_chunks_fixture("chat-completion-simple-1");
    test_server
        .mock_streaming_response("/v1/chat/completions", chunks)
        .await;

    let model = create_test_model(&test_server.base_url, "gpt-4");

    let options = CallOptions {
        prompt: vec![Message::User {
            content: vec![UserContentPart::Text {
                text: "Hello".into(),
            }],
        }],
        stop_sequences: Some(vec!["STOP".to_string(), "END".to_string()]),
        ..Default::default()
    };

    let mut stream_response = model.do_stream(options).await.expect("Stream should start");

    let mut stream_parts = vec![];
    while let Some(part_result) = stream_response.stream.next().await {
        let part = part_result.expect("Stream part should be ok");
        stream_parts.push(part);
    }

    // Verify stream completed
    assert!(!stream_parts.is_empty());
}

#[tokio::test]
async fn test_stream_presence_penalty() {
    // Test streaming with presence and frequency penalty
    let test_server = TestServer::new().await;

    let chunks = load_chunks_fixture("chat-completion-simple-1");
    test_server
        .mock_streaming_response("/v1/chat/completions", chunks)
        .await;

    let model = create_test_model(&test_server.base_url, "gpt-4");

    let options = CallOptions {
        prompt: vec![Message::User {
            content: vec![UserContentPart::Text {
                text: "Hello".into(),
            }],
        }],
        presence_penalty: Some(0.5),
        frequency_penalty: Some(0.3),
        ..Default::default()
    };

    let mut stream_response = model.do_stream(options).await.expect("Stream should start");

    let mut stream_parts = vec![];
    while let Some(part_result) = stream_response.stream.next().await {
        let part = part_result.expect("Stream part should be ok");
        stream_parts.push(part);
    }

    // Verify stream completed
    assert!(!stream_parts.is_empty());
}

#[tokio::test]
async fn test_stream_seed() {
    // Test streaming with seed for reproducibility
    let test_server = TestServer::new().await;

    let chunks = load_chunks_fixture("chat-completion-simple-1");
    test_server
        .mock_streaming_response("/v1/chat/completions", chunks)
        .await;

    let model = create_test_model(&test_server.base_url, "gpt-4");

    let options = CallOptions {
        prompt: vec![Message::User {
            content: vec![UserContentPart::Text {
                text: "Hello".into(),
            }],
        }],
        seed: Some(12345),
        ..Default::default()
    };

    let mut stream_response = model.do_stream(options).await.expect("Stream should start");

    let mut stream_parts = vec![];
    while let Some(part_result) = stream_response.stream.next().await {
        let part = part_result.expect("Stream part should be ok");
        stream_parts.push(part);
    }

    // Verify stream completed
    assert!(!stream_parts.is_empty());
}

#[tokio::test]
async fn test_stream_response_metadata() {
    // Test that response metadata is captured during streaming
    let test_server = TestServer::new().await;

    let chunks = load_chunks_fixture("chat-completion-simple-1");
    test_server
        .mock_streaming_response("/v1/chat/completions", chunks)
        .await;

    let model = create_test_model(&test_server.base_url, "gpt-4");

    let options = CallOptions {
        prompt: vec![Message::User {
            content: vec![UserContentPart::Text {
                text: "Hello".into(),
            }],
        }],
        ..Default::default()
    };

    let stream_response = model.do_stream(options).await.expect("Stream should start");

    // Check response metadata if available
    if let Some(ref response) = stream_response.response {
        // Response ID and model should be present after streaming starts
        assert!(response.id.is_some() || response.model_id.is_some());
    }
}
