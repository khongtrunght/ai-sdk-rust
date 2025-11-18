mod common;

use ai_sdk_openai::OpenAIChatModel;
use ai_sdk_provider::language_model::{FunctionTool, Message, Tool, ToolChoice, UserContentPart};
use ai_sdk_provider::{CallOptions, Content, FinishReason, LanguageModel, StreamPart};
use common::{load_chunks_fixture, load_json_fixture, TestServer};
use futures::stream::StreamExt;

#[tokio::test]
async fn test_tool_calling_non_streaming_with_fixture() {
    // Setup mock server
    let test_server = TestServer::new().await;

    // Load fixture
    let fixture = load_json_fixture("chat-tool-calling-1");

    // Configure mock to return fixture
    test_server
        .mock_json_response("/v1/chat/completions", fixture)
        .await;

    // Create model pointing to mock server
    let model = OpenAIChatModel::new("gpt-4", "test-key")
        .with_base_url(format!("{}/v1", test_server.base_url));

    let tool = Tool::Function(FunctionTool {
        name: "get_weather".to_string(),
        description: Some("Get the weather for a location".to_string()),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name"
                }
            },
            "required": ["location"]
        }),
        provider_options: None,
    });

    let options = CallOptions {
        prompt: vec![Message::User {
            content: vec![UserContentPart::Text {
                text: "What's the weather in Tokyo?".to_string(),
            }],
        }],
        tools: Some(vec![tool]),
        tool_choice: Some(ToolChoice::Auto),
        ..Default::default()
    };

    let response = model
        .do_generate(options)
        .await
        .expect("Failed to generate response");

    // Should contain a tool call
    let has_tool_call = response
        .content
        .iter()
        .any(|c| matches!(c, Content::ToolCall(_)));
    assert!(
        has_tool_call,
        "Response should contain a tool call, got: {:?}",
        response.content
    );

    // Should have ToolCalls finish reason
    assert_eq!(
        response.finish_reason,
        FinishReason::ToolCalls,
        "Finish reason should be ToolCalls"
    );

    // Verify tool call structure
    if let Some(Content::ToolCall(tool_call)) = response
        .content
        .iter()
        .find(|c| matches!(c, Content::ToolCall(_)))
    {
        assert_eq!(tool_call.tool_name, "get_weather");
        assert_eq!(tool_call.tool_call_id, "call_abc123");
        assert!(!tool_call.input.is_empty());

        // Snapshot the tool call
        insta::assert_json_snapshot!(tool_call, @r#"
        {
          "toolCallId": "call_abc123",
          "toolName": "get_weather",
          "input": "{\"location\":\"Tokyo\"}"
        }
        "#);
    }
}

#[tokio::test]
async fn test_tool_calling_streaming_with_fixture() {
    // Setup mock server
    let test_server = TestServer::new().await;

    // Load streaming chunks
    let chunks = load_chunks_fixture("chat-tool-calling-calculate-1");

    // Configure mock to return streaming response
    test_server
        .mock_streaming_response("/v1/chat/completions", chunks)
        .await;

    // Create model pointing to mock server
    let model = OpenAIChatModel::new("gpt-4", "test-key")
        .with_base_url(format!("{}/v1", test_server.base_url));

    let tool = Tool::Function(FunctionTool {
        name: "calculate".to_string(),
        description: Some("Perform a calculation".to_string()),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }),
        provider_options: None,
    });

    let options = CallOptions {
        prompt: vec![Message::User {
            content: vec![UserContentPart::Text {
                text: "Calculate 15 * 7".to_string(),
            }],
        }],
        tools: Some(vec![tool]),
        tool_choice: Some(ToolChoice::Auto),
        ..Default::default()
    };

    let response = model
        .do_stream(options)
        .await
        .expect("Failed to start stream");

    let mut stream = response.stream;

    let mut has_tool_input_start = false;
    let mut has_tool_input_delta = false;
    let mut has_tool_input_end = false;
    let mut has_tool_call = false;
    let mut finish_reason = None;

    while let Some(part) = stream.next().await {
        let part = part.expect("Stream part should be Ok");
        match part {
            StreamPart::ToolInputStart { .. } => {
                has_tool_input_start = true;
            }
            StreamPart::ToolInputDelta { .. } => {
                has_tool_input_delta = true;
            }
            StreamPart::ToolInputEnd { .. } => {
                has_tool_input_end = true;
            }
            StreamPart::ToolCall(_) => {
                has_tool_call = true;
            }
            StreamPart::Finish {
                finish_reason: fr, ..
            } => {
                finish_reason = Some(fr);
            }
            _ => {}
        }
    }

    assert!(has_tool_input_start, "Stream should contain ToolInputStart");
    assert!(has_tool_input_delta, "Stream should contain ToolInputDelta");
    assert!(has_tool_input_end, "Stream should contain ToolInputEnd");
    assert!(has_tool_call, "Stream should contain ToolCall");
    assert_eq!(
        finish_reason,
        Some(FinishReason::ToolCalls),
        "Finish reason should be ToolCalls"
    );
}

#[tokio::test]
async fn test_multiple_tools_with_fixture() {
    // Setup mock server
    let test_server = TestServer::new().await;

    // Load fixture with multiple tool calls
    let fixture = load_json_fixture("chat-multiple-tools-1");

    // Configure mock to return fixture
    test_server
        .mock_json_response("/v1/chat/completions", fixture)
        .await;

    // Create model pointing to mock server
    let model = OpenAIChatModel::new("gpt-4", "test-key")
        .with_base_url(format!("{}/v1", test_server.base_url));

    let weather_tool = Tool::Function(FunctionTool {
        name: "get_weather".to_string(),
        description: Some("Get the weather for a location".to_string()),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }),
        provider_options: None,
    });

    let time_tool = Tool::Function(FunctionTool {
        name: "get_time".to_string(),
        description: Some("Get the current time for a location".to_string()),
        input_schema: serde_json::json!({
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
                text: "What's the weather and time in Paris?".to_string(),
            }],
        }],
        tools: Some(vec![weather_tool, time_tool]),
        tool_choice: Some(ToolChoice::Auto),
        ..Default::default()
    };

    let response = model
        .do_generate(options)
        .await
        .expect("Failed to generate response");

    // Should contain multiple tool calls
    let tool_calls: Vec<_> = response
        .content
        .iter()
        .filter_map(|c| match c {
            Content::ToolCall(tc) => Some(tc),
            _ => None,
        })
        .collect();

    assert_eq!(
        tool_calls.len(),
        2,
        "Response should contain exactly 2 tool calls"
    );

    // Verify tool names
    let tool_names: Vec<&str> = tool_calls.iter().map(|tc| tc.tool_name.as_str()).collect();
    assert!(tool_names.contains(&"get_weather"));
    assert!(tool_names.contains(&"get_time"));

    // Snapshot the tool calls
    insta::assert_json_snapshot!(tool_calls, @r#"
    [
      {
        "toolCallId": "call_weather123",
        "toolName": "get_weather",
        "input": "{\"location\":\"Paris\"}"
      },
      {
        "toolCallId": "call_time123",
        "toolName": "get_time",
        "input": "{\"location\":\"Paris\"}"
      }
    ]
    "#);
}

#[tokio::test]
async fn test_no_tool_calls_when_not_needed_with_fixture() {
    // Setup mock server
    let test_server = TestServer::new().await;

    // Load fixture with no tool calls (regular text response)
    let fixture = load_json_fixture("chat-no-tool-call-1");

    // Configure mock to return fixture
    test_server
        .mock_json_response("/v1/chat/completions", fixture)
        .await;

    // Create model pointing to mock server
    let model = OpenAIChatModel::new("gpt-4", "test-key")
        .with_base_url(format!("{}/v1", test_server.base_url));

    let tool = Tool::Function(FunctionTool {
        name: "get_weather".to_string(),
        description: Some("Get the weather for a location".to_string()),
        input_schema: serde_json::json!({
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
                text: "Hello, how are you?".to_string(),
            }],
        }],
        tools: Some(vec![tool]),
        tool_choice: Some(ToolChoice::Auto),
        ..Default::default()
    };

    let response = model
        .do_generate(options)
        .await
        .expect("Failed to generate response");

    // Should NOT contain a tool call (just a regular text response)
    let has_tool_call = response
        .content
        .iter()
        .any(|c| matches!(c, Content::ToolCall(_)));
    assert!(
        !has_tool_call,
        "Response should not contain a tool call for a greeting"
    );

    // Should have Stop finish reason, not ToolCalls
    assert_eq!(
        response.finish_reason,
        FinishReason::Stop,
        "Finish reason should be Stop"
    );

    // Verify we got a text response
    if let Some(Content::Text(text)) = response.content.first() {
        assert!(!text.text.is_empty());
        assert!(text.text.contains("Hello") || text.text.contains("well"));
    } else {
        panic!("Expected text content in response");
    }
}
