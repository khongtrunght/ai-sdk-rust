//! Example of a streaming tool that emits preliminary results
//!
//! This demonstrates how to create tools that provide progressive updates,
//! such as an image generation tool that reports progress.

use ai_sdk_core::{generate_text, Tool, ToolContext, ToolError, ToolOutput};
use ai_sdk_openai::OpenAIChatModel;
use ai_sdk_provider::JsonValue;
use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;

/// Image generation tool that reports progress
struct ImageGenerationTool;

#[async_trait]
impl Tool for ImageGenerationTool {
    fn name(&self) -> &str {
        "generate_image"
    }

    fn description(&self) -> &str {
        "Generates an image with progress updates"
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Description of the image to generate"
                }
            },
            "required": ["prompt"]
        })
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        use async_stream::stream;

        let prompt = input["prompt"]
            .as_str()
            .unwrap_or("default prompt")
            .to_string();

        let stream = stream! {
            // Preliminary result 1: Starting
            let value: JsonValue = serde_json::from_value(json!({
                "status": "loading",
                "progress": 0,
                "message": format!("Starting image generation for: {}", prompt)
            })).unwrap();
            yield Ok(value);

            tokio::time::sleep(Duration::from_secs(1)).await;

            // Preliminary result 2: Progress 25%
            let value: JsonValue = serde_json::from_value(json!({
                "status": "loading",
                "progress": 25,
                "message": "Processing prompt..."
            })).unwrap();
            yield Ok(value);

            tokio::time::sleep(Duration::from_secs(1)).await;

            // Preliminary result 3: Progress 50%
            let value: JsonValue = serde_json::from_value(json!({
                "status": "loading",
                "progress": 50,
                "message": "Generating image..."
            })).unwrap();
            yield Ok(value);

            tokio::time::sleep(Duration::from_secs(1)).await;

            // Preliminary result 4: Progress 75%
            let value: JsonValue = serde_json::from_value(json!({
                "status": "loading",
                "progress": 75,
                "message": "Finalizing..."
            })).unwrap();
            yield Ok(value);

            tokio::time::sleep(Duration::from_secs(1)).await;

            // Final result
            let value: JsonValue = serde_json::from_value(json!({
                "status": "success",
                "progress": 100,
                "image_url": format!("https://example.com/images/{}.png", prompt.replace(' ', "_")),
                "description": format!("Generated image for: {}", prompt)
            })).unwrap();
            yield Ok(value);
        };

        Ok(ToolOutput::Stream(Box::pin(stream)))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get API key from environment
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| "OPENAI_API_KEY environment variable not set")?;

    // Create model
    let model = OpenAIChatModel::new("gpt-4", api_key);

    // Create tools
    let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(ImageGenerationTool)];

    println!("Starting text generation with streaming tool...\n");

    // Generate text with preliminary tool result callback
    let result = generate_text()
        .model(model)
        .prompt("Please generate an image of a sunset over mountains")
        .tools(tools)
        .max_steps(5)
        .on_preliminary_tool_result(Arc::new(|preliminary| {
            Box::pin(async move {
                println!("📊 Preliminary result:");
                println!("   Tool: {}", preliminary.tool_name);
                println!("   Output: {:?}", preliminary.output);
                println!();
            })
        }))
        .execute()
        .await?;

    println!("\n✅ Final result:");
    println!("{}", result.text());
    println!("\nToken usage: {:?}", result.usage());

    Ok(())
}
