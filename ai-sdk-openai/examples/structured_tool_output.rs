use ai_sdk_core::{Tool, ToolContext, ToolError, ToolExecutor};
use ai_sdk_provider::language_model::{ContentPart, ToolCallPart, ToolResultOutput};
use ai_sdk_provider::JsonValue;
use async_trait::async_trait;
use std::sync::Arc;

// Example 1: Default conversion (string → text, object → json)
struct SimpleTextTool;

#[async_trait]
impl Tool for SimpleTextTool {
    fn name(&self) -> &str {
        "simple_text"
    }

    fn description(&self) -> &str {
        "Returns simple text"
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn execute(
        &self,
        _input: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Result<JsonValue, ToolError> {
        // Return a simple string - will be converted to ToolResultOutput::Text
        Ok(JsonValue::String("Hello, world!".to_string()))
    }
}

// Example 2: Custom conversion - multi-part content
struct ImageGenerationTool;

#[async_trait]
impl Tool for ImageGenerationTool {
    fn name(&self) -> &str {
        "generate_image"
    }

    fn description(&self) -> &str {
        "Generates an image"
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "prompt": { "type": "string" }
            }
        })
    }

    async fn execute(
        &self,
        _input: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Result<JsonValue, ToolError> {
        // Return a marker value - will be converted by custom to_model_output
        Ok(JsonValue::String("image_generated".to_string()))
    }

    // Custom conversion: return text + image
    fn to_model_output(&self, _output: JsonValue) -> ToolResultOutput {
        // Ignore the output and return custom multi-part content
        ToolResultOutput::Content {
            value: vec![
                ContentPart::Text {
                    text: "A beautiful sunset over mountains".to_string(),
                    provider_metadata: None,
                },
                ContentPart::ImageUrl {
                    url: "https://example.com/sunset.png".to_string(),
                    provider_metadata: None,
                },
            ],
            provider_metadata: None,
        }
    }
}

// Example 3: Tool that returns JSON data
struct DataFetchTool;

#[async_trait]
impl Tool for DataFetchTool {
    fn name(&self) -> &str {
        "fetch_data"
    }

    fn description(&self) -> &str {
        "Fetches structured data"
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "id": { "type": "string" }
            }
        })
    }

    async fn execute(
        &self,
        _input: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Result<JsonValue, ToolError> {
        // Return structured data as JsonValue::Object
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert("id".to_string(), JsonValue::String("item-123".to_string()));
        map.insert(
            "name".to_string(),
            JsonValue::String("Test Item".to_string()),
        );
        map.insert(
            "count".to_string(),
            JsonValue::Number(serde_json::Number::from(42)),
        );

        Ok(JsonValue::Object(map))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Structured Tool Output Examples ===\n");

    // Create tool executor with all tools
    let tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(SimpleTextTool),
        Arc::new(ImageGenerationTool),
        Arc::new(DataFetchTool),
    ];
    let executor = ToolExecutor::new(tools);

    // Example 1: Simple text tool
    println!("1. Simple Text Tool:");
    let tool_call = ToolCallPart {
        tool_call_id: "call_1".to_string(),
        tool_name: "simple_text".to_string(),
        input: "{}".to_string(),
        provider_executed: None,
        dynamic: None,
        provider_metadata: None,
    };

    let results = executor.execute_tools(vec![tool_call]).await;
    for result in &results {
        println!("   Tool: {}", result.tool_name);
        match &result.output {
            ToolResultOutput::Text { value, .. } => {
                println!("   Output (Text): {}", value);
            }
            _ => println!("   Output: {:?}", result.output),
        }
    }
    println!();

    // Example 2: Image generation tool with multi-part content
    println!("2. Image Generation Tool (Multi-Part Content):");
    let tool_call = ToolCallPart {
        tool_call_id: "call_2".to_string(),
        tool_name: "generate_image".to_string(),
        input: "{\"prompt\": \"sunset\"}".to_string(),
        provider_executed: None,
        dynamic: None,
        provider_metadata: None,
    };

    let results = executor.execute_tools(vec![tool_call]).await;
    for result in &results {
        println!("   Tool: {}", result.tool_name);
        match &result.output {
            ToolResultOutput::Content { value, .. } => {
                println!("   Output (Multi-Part Content):");
                for (i, part) in value.iter().enumerate() {
                    match part {
                        ContentPart::Text { text, .. } => {
                            println!("     Part {}: Text = {}", i, text);
                        }
                        ContentPart::ImageUrl { url, .. } => {
                            println!("     Part {}: Image URL = {}", i, url);
                        }
                        _ => println!("     Part {}: {:?}", i, part),
                    }
                }
            }
            _ => println!("   Output: {:?}", result.output),
        }
    }
    println!();

    // Example 3: Data fetch tool with JSON output
    println!("3. Data Fetch Tool (JSON Output):");
    let tool_call = ToolCallPart {
        tool_call_id: "call_3".to_string(),
        tool_name: "fetch_data".to_string(),
        input: "{\"id\": \"item-123\"}".to_string(),
        provider_executed: None,
        dynamic: None,
        provider_metadata: None,
    };

    let results = executor.execute_tools(vec![tool_call]).await;
    for result in &results {
        println!("   Tool: {}", result.tool_name);
        match &result.output {
            ToolResultOutput::Json { value, .. } => {
                println!("   Output (JSON): {}", serde_json::to_string_pretty(value)?);
            }
            _ => println!("   Output: {:?}", result.output),
        }
    }
    println!();

    // Example 4: Tool not found (error output)
    println!("4. Tool Not Found (Error Output):");
    let tool_call = ToolCallPart {
        tool_call_id: "call_4".to_string(),
        tool_name: "nonexistent_tool".to_string(),
        input: "{}".to_string(),
        provider_executed: None,
        dynamic: None,
        provider_metadata: None,
    };

    let results = executor.execute_tools(vec![tool_call]).await;
    for result in &results {
        println!("   Tool: {}", result.tool_name);
        match &result.output {
            ToolResultOutput::ErrorText { value, .. } => {
                println!("   Error: {}", value);
            }
            _ => println!("   Output: {:?}", result.output),
        }
    }
    println!();

    println!("=== Examples Complete ===");
    Ok(())
}
