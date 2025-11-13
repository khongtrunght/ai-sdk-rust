use ai_sdk_openai::OpenAIChatModel;
use ai_sdk_provider::language_model::{
    FunctionTool, Tool, ToolChoice, UserContentPart,
};
use ai_sdk_provider::{CallOptions, Content, FinishReason, LanguageModel, Message, StreamPart};
use futures::stream::StreamExt;

#[tokio::test]
#[ignore] // Requires OPENAI_API_KEY
async fn test_tool_calling_non_streaming() {
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY environment variable not set");

    let model = OpenAIChatModel::new("gpt-4", api_key);

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
        assert!(!tool_call.tool_call_id.is_empty());
        assert!(!tool_call.input.is_empty());

        // Parse the input to verify it contains location
        let input: serde_json::Value = serde_json::from_str(&tool_call.input)
            .expect("Tool input should be valid JSON");
        assert!(
            input.get("location").is_some(),
            "Tool input should contain location"
        );
    }
}

#[tokio::test]
#[ignore] // Requires OPENAI_API_KEY
async fn test_tool_calling_streaming() {
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY environment variable not set");

    let model = OpenAIChatModel::new("gpt-4", api_key);

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

    assert!(
        has_tool_input_start,
        "Stream should contain ToolInputStart"
    );
    assert!(
        has_tool_input_delta,
        "Stream should contain ToolInputDelta"
    );
    assert!(has_tool_input_end, "Stream should contain ToolInputEnd");
    assert!(has_tool_call, "Stream should contain ToolCall");
    assert_eq!(
        finish_reason,
        Some(FinishReason::ToolCalls),
        "Finish reason should be ToolCalls"
    );
}

#[tokio::test]
#[ignore] // Requires OPENAI_API_KEY
async fn test_multiple_tools() {
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY environment variable not set");

    let model = OpenAIChatModel::new("gpt-4", api_key);

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

    // Should contain at least one tool call
    let tool_calls: Vec<_> = response
        .content
        .iter()
        .filter_map(|c| match c {
            Content::ToolCall(tc) => Some(tc),
            _ => None,
        })
        .collect();

    assert!(
        !tool_calls.is_empty(),
        "Response should contain at least one tool call"
    );
}

#[tokio::test]
#[ignore] // Requires OPENAI_API_KEY
async fn test_no_tool_calls_when_not_needed() {
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY environment variable not set");

    let model = OpenAIChatModel::new("gpt-4", api_key);

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
}
