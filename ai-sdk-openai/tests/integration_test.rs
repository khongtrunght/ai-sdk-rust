// NOTE: These tests require OPENAI_API_KEY environment variable
// Run with: OPENAI_API_KEY=sk-... cargo test --package ai-sdk-openai -- --ignored

use ai_sdk_openai::*;
use ai_sdk_provider::language_model::{Message, UserContentPart};
use ai_sdk_provider::*;

#[tokio::test]
#[ignore] // Ignore by default to avoid API costs
async fn test_openai_generate() {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    let model = openai("gpt-4", api_key);

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

    let response = model
        .do_generate(options)
        .await
        .expect("Generate should succeed");

    assert_eq!(response.finish_reason, FinishReason::Stop);
    assert!(!response.content.is_empty());

    if let Content::Text(text) = &response.content[0] {
        println!("Response: {}", text.text);
        assert!(text.text.contains("Hello"));
    } else {
        panic!("Expected text content");
    }
}

#[tokio::test]
#[ignore]
async fn test_openai_stream() {
    use tokio_stream::StreamExt;

    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    let model = openai("gpt-4", api_key);

    let options = CallOptions {
        prompt: vec![Message::User {
            content: vec![UserContentPart::Text {
                text: "Count to 3".into(),
            }],
        }],
        temperature: Some(0.0),
        max_output_tokens: Some(50),
        ..Default::default()
    };

    let mut stream_response = model.do_stream(options).await.expect("Stream should start");

    let mut chunks = vec![];
    while let Some(part_result) = stream_response.stream.next().await {
        let part = part_result.expect("Stream part should be ok");
        chunks.push(part);
    }

    assert!(!chunks.is_empty(), "Should receive stream chunks");
    println!("Received {} chunks", chunks.len());
}
