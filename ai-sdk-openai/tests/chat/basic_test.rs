use crate::common::{create_test_model, load_chunks_fixture, load_json_fixture, TestServer};
use ai_sdk_provider::language_model::{Message, UserContentPart};
use ai_sdk_provider::*;

#[tokio::test]
async fn test_openai_generate_with_fixture() {
    // Setup mock server
    let test_server = TestServer::new().await;

    // Load fixture
    let fixture = load_json_fixture("chat-completion-simple-1");

    // Configure mock to return fixture
    test_server
        .mock_json_response("/v1/chat/completions", fixture)
        .await;

    // Create model pointing to mock server (using builder pattern)
    // Note: base_url needs to include /v1 since the model appends /chat/completions
    let model = create_test_model(&test_server.base_url, "gpt-4");

    let options = CallOptions {
        prompt: vec![Message::User {
            content: vec![UserContentPart::Text {
                text: "Say 'Hello, Rust!'".into(),
            }],
        }],
        temperature: Some(0.0),
        max_output_tokens: Some(10),
        ..Default::default()
    };

    // Execute (hits mock server, not real API)
    let response = model
        .do_generate(options)
        .await
        .expect("Generate should succeed");

    // Snapshot assertion
    insta::assert_json_snapshot!(response.content, @r###"
    [
      {
        "type": "text",
        "text": "Hello, Rust!"
      }
    ]
    "###);

    // Traditional assertions still work
    assert_eq!(response.finish_reason, FinishReason::Stop);
    assert!(!response.content.is_empty());

    if let Content::Text(text) = &response.content[0] {
        assert_eq!(text.text, "Hello, Rust!");
    } else {
        panic!("Expected text content");
    }
}

#[tokio::test]
async fn test_openai_stream_with_fixture() {
    use tokio_stream::StreamExt;

    // Setup mock server
    let test_server = TestServer::new().await;

    // Load streaming chunks
    let chunks = load_chunks_fixture("chat-completion-simple-1");

    // Configure mock to return streaming response
    test_server
        .mock_streaming_response("/v1/chat/completions", chunks)
        .await;

    // Create model pointing to mock server
    // Note: base_url needs to include /v1 since the model appends /chat/completions
    let model = create_test_model(&test_server.base_url, "gpt-4");

    let options = CallOptions {
        prompt: vec![Message::User {
            content: vec![UserContentPart::Text {
                text: "Say 'Hello, Rust!'".into(),
            }],
        }],
        temperature: Some(0.0),
        max_output_tokens: Some(50),
        ..Default::default()
    };

    let mut stream_response = model.do_stream(options).await.expect("Stream should start");

    let mut stream_parts = vec![];
    while let Some(part_result) = stream_response.stream.next().await {
        let part = part_result.expect("Stream part should be ok");
        stream_parts.push(part);
    }

    assert!(!stream_parts.is_empty(), "Should receive stream chunks");

    // Verify we got the expected chunks
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

    // Combine all text deltas
    let full_text: String = text_deltas.join("");
    assert_eq!(full_text, "Hello, Rust!");
}

// Phase 1: Basic Functionality Tests

#[tokio::test]
async fn test_extract_usage() {
    // TypeScript reference: line 249
    let test_server = TestServer::new().await;

    // Create response inline (mimics prepareJsonResponse)
    let response_json = serde_json::json!({
        "id": "chatcmpl-95ZTZkhr0mHNKqerQfiwkuox3PHAd",
        "object": "chat.completion",
        "created": 1711115037,
        "model": "gpt-3.5-turbo-0125",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": ""
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 5,
            "total_tokens": 25
        },
        "system_fingerprint": "fp_3bc1b5746c"
    });

    test_server
        .mock_json_response("/v1/chat/completions", response_json)
        .await;

    let model = create_test_model(&test_server.base_url, "gpt-4");

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

    // Verify usage extraction
    assert_eq!(response.usage.input_tokens, Some(20));
    assert_eq!(response.usage.output_tokens, Some(5));
    assert_eq!(response.usage.total_tokens, Some(25));
    assert_eq!(response.usage.cached_input_tokens, None);
    assert_eq!(response.usage.reasoning_tokens, None);
}

#[tokio::test]
async fn test_send_request_body() {
    // TypeScript reference: line 269
    use wiremock::{Mock, ResponseTemplate};

    let test_server = TestServer::new().await;

    let response_json = serde_json::json!({
        "id": "chatcmpl-95ZTZkhr0mHNKqerQfiwkuox3PHAd",
        "object": "chat.completion",
        "created": 1711115037,
        "model": "gpt-3.5-turbo-0125",
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
            "completion_tokens": 30,
            "total_tokens": 34
        },
        "system_fingerprint": "fp_3bc1b5746c"
    });

    // Verify request is sent correctly
    Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(response_json)
                .insert_header("content-type", "application/json"),
        )
        .expect(1)
        .mount(&test_server.server)
        .await;

    let model = create_test_model(&test_server.base_url, "gpt-3.5-turbo");

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

    // Mock expectations verified automatically (expect(1))
}

#[tokio::test]
async fn test_additional_response_information() {
    // TypeScript reference: line 315
    let test_server = TestServer::new().await;

    let response_json = serde_json::json!({
        "id": "test-id",
        "object": "chat.completion",
        "created": 123,
        "model": "test-model",
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
            "completion_tokens": 30,
            "total_tokens": 34
        },
        "system_fingerprint": "fp_3bc1b5746c"
    });

    test_server
        .mock_json_response("/v1/chat/completions", response_json)
        .await;

    let model = create_test_model(&test_server.base_url, "gpt-4");

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

    // Validate response metadata
    let response_info = response.response.expect("Response info should be present");
    assert_eq!(response_info.id, Some("test-id".to_string()));
    assert_eq!(response_info.model_id, Some("test-model".to_string()));
    assert_eq!(
        response_info.timestamp,
        Some("1970-01-01T00:02:03Z".to_string())
    );
    assert!(response_info.headers.is_some());
}

#[tokio::test]
async fn test_support_partial_usage() {
    // TypeScript reference: line 361
    let test_server = TestServer::new().await;

    // Response with partial usage (no completion_tokens)
    let response_json = serde_json::json!({
        "id": "chatcmpl-95ZTZkhr0mHNKqerQfiwkuox3PHAd",
        "object": "chat.completion",
        "created": 1711115037,
        "model": "gpt-3.5-turbo-0125",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": ""
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 20,
            "total_tokens": 20
        },
        "system_fingerprint": "fp_3bc1b5746c"
    });

    test_server
        .mock_json_response("/v1/chat/completions", response_json)
        .await;

    let model = create_test_model(&test_server.base_url, "gpt-4");

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

    // Verify partial usage (no completion_tokens)
    assert_eq!(response.usage.input_tokens, Some(20));
    assert_eq!(response.usage.output_tokens, None);
    assert_eq!(response.usage.total_tokens, Some(20));
}

#[tokio::test]
async fn test_extract_logprobs() {
    // TypeScript reference: line 381
    use ai_sdk_provider::json_value::JsonValue;

    let test_server = TestServer::new().await;

    // Response with logprobs
    let response_json = serde_json::json!({
        "id": "chatcmpl-95ZTZkhr0mHNKqerQfiwkuox3PHAd",
        "object": "chat.completion",
        "created": 1711115037,
        "model": "gpt-3.5-turbo",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I assist you today?"
            },
            "logprobs": {
                "content": [
                    {
                        "token": "Hello",
                        "logprob": -0.0009994634,
                        "top_logprobs": [{"token": "Hello", "logprob": -0.0009994634}]
                    },
                    {
                        "token": "!",
                        "logprob": -0.13410144,
                        "top_logprobs": [{"token": "!", "logprob": -0.13410144}]
                    }
                ]
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 8,
            "total_tokens": 12
        },
        "system_fingerprint": "fp_3bc1b5746c"
    });

    test_server
        .mock_json_response("/v1/chat/completions", response_json)
        .await;

    let model = create_test_model(&test_server.base_url, "gpt-3.5-turbo");

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

    // Verify logprobs are present in provider metadata
    let provider_metadata = response
        .provider_metadata
        .expect("Provider metadata should be present");

    let openai_metadata = provider_metadata
        .get("openai")
        .expect("OpenAI metadata should be present");

    let logprobs = openai_metadata
        .get("logprobs")
        .expect("Logprobs should be present");

    // Verify it's an array with expected content
    if let JsonValue::Array(ref logprobs_array) = logprobs {
        assert_eq!(logprobs_array.len(), 2);
    } else {
        panic!("Expected logprobs to be an array");
    }
}

#[tokio::test]
async fn test_extract_finish_reason() {
    // TypeScript reference: line 399
    let test_server = TestServer::new().await;

    let response_json = serde_json::json!({
        "id": "chatcmpl-95ZTZkhr0mHNKqerQfiwkuox3PHAd",
        "object": "chat.completion",
        "created": 1711115037,
        "model": "gpt-3.5-turbo",
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
            "completion_tokens": 30,
            "total_tokens": 34
        },
        "system_fingerprint": "fp_3bc1b5746c"
    });

    test_server
        .mock_json_response("/v1/chat/completions", response_json)
        .await;

    let model = create_test_model(&test_server.base_url, "gpt-3.5-turbo");

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

    assert_eq!(response.finish_reason, FinishReason::Stop);
}

#[tokio::test]
async fn test_unknown_finish_reason() {
    // TypeScript reference: line 411
    let test_server = TestServer::new().await;

    // Response with unknown finish_reason "eos"
    let response_json = serde_json::json!({
        "id": "chatcmpl-95ZTZkhr0mHNKqerQfiwkuox3PHAd",
        "object": "chat.completion",
        "created": 1711115037,
        "model": "gpt-3.5-turbo",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Response"
            },
            "finish_reason": "eos"
        }],
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 30,
            "total_tokens": 34
        },
        "system_fingerprint": "fp_3bc1b5746c"
    });

    test_server
        .mock_json_response("/v1/chat/completions", response_json)
        .await;

    let model = create_test_model(&test_server.base_url, "gpt-3.5-turbo");

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

    // Unknown finish_reason "eos" should map to FinishReason::Unknown
    assert_eq!(response.finish_reason, FinishReason::Unknown);
}

#[tokio::test]
async fn test_expose_raw_response_headers() {
    // TypeScript reference: line 423
    use wiremock::{Mock, ResponseTemplate};

    let test_server = TestServer::new().await;

    let response_json = serde_json::json!({
        "id": "chatcmpl-95ZTZkhr0mHNKqerQfiwkuox3PHAd",
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
            "completion_tokens": 30,
            "total_tokens": 34
        },
        "system_fingerprint": "fp_3bc1b5746c"
    });

    // Mock with custom header
    Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(response_json)
                .insert_header("content-type", "application/json")
                .insert_header("test-header", "test-value"),
        )
        .mount(&test_server.server)
        .await;

    let model = create_test_model(&test_server.base_url, "gpt-3.5-turbo");

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

    // Verify custom header is exposed
    let response_info = response.response.expect("Response info should be present");
    let headers = response_info.headers.expect("Headers should be present");

    assert!(headers.contains_key("test-header"));
    assert_eq!(headers.get("test-header").unwrap(), "test-value");
    assert!(headers.contains_key("content-type"));
}
