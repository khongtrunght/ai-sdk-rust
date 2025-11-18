mod common;

use ai_sdk_openai::*;
use ai_sdk_provider::language_model::{Message, UserContentPart};
use ai_sdk_provider::*;
use common::{load_chunks_fixture, load_json_fixture, TestServer};

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
    let model = OpenAIChatModel::new("gpt-4", "test-key")
        .with_base_url(format!("{}/v1", test_server.base_url));

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
    let model = OpenAIChatModel::new("gpt-4", "test-key")
        .with_base_url(format!("{}/v1", test_server.base_url));

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
