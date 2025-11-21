//! Example of using a ToolLoopAgent with autonomous tool execution
//!
//! To run this example:
//! ```bash
//! export OPENAI_API_KEY=your_key
//! cargo run --example tool_loop_agent
//! ```

use ai_sdk_core::agent::{
    step_count_is, Agent, AgentCallParameters, ToolLoopAgent, ToolLoopAgentSettings,
};
use ai_sdk_core::{Tool, ToolContext, ToolError, ToolOutput};
use ai_sdk_openai::OpenAIProvider;
use ai_sdk_provider::{JsonValue, ProviderV3};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;

/// Example weather tool that simulates weather data retrieval
struct WeatherTool;

#[async_trait]
impl Tool for WeatherTool {
    fn name(&self) -> &str {
        "get_weather"
    }

    fn description(&self) -> &str {
        "Get the current weather for a location"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                }
            },
            "required": ["location"]
        })
    }

    async fn execute(&self, input: Value, _context: &ToolContext) -> Result<ToolOutput, ToolError> {
        // Extract location from input
        let location = input
            .get("location")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_input("Missing location parameter"))?;

        println!("🌤️  Fetching weather for: {}", location);

        // Simulate weather API call with mock data
        let weather_data: JsonValue = serde_json::from_value(json!({
            "location": location,
            "temperature": 72,
            "condition": "sunny",
            "humidity": 65,
            "wind_speed": 10,
            "forecast": "Clear skies expected throughout the day"
        }))
        .map_err(|e| ToolError::execution(format!("Failed to create result: {}", e)))?;

        Ok(ToolOutput::Value(weather_data))
    }
}

/// Example calculation tool
struct CalculatorTool;

#[async_trait]
impl Tool for CalculatorTool {
    fn name(&self) -> &str {
        "calculate"
    }

    fn description(&self) -> &str {
        "Perform basic arithmetic calculations"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform"
                },
                "a": {
                    "type": "number",
                    "description": "First operand"
                },
                "b": {
                    "type": "number",
                    "description": "Second operand"
                }
            },
            "required": ["operation", "a", "b"]
        })
    }

    async fn execute(&self, input: Value, _context: &ToolContext) -> Result<ToolOutput, ToolError> {
        let operation = input
            .get("operation")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_input("Missing operation"))?;

        let a = input
            .get("a")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| ToolError::invalid_input("Missing or invalid parameter 'a'"))?;

        let b = input
            .get("b")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| ToolError::invalid_input("Missing or invalid parameter 'b'"))?;

        println!("🔢 Calculating: {} {} {}", a, operation, b);

        let result = match operation {
            "add" => a + b,
            "subtract" => a - b,
            "multiply" => a * b,
            "divide" => {
                if b == 0.0 {
                    return Err(ToolError::execution("Division by zero"));
                }
                a / b
            }
            _ => {
                return Err(ToolError::invalid_input(format!(
                    "Unknown operation: {}",
                    operation
                )))
            }
        };

        let result_json: JsonValue = serde_json::from_value(json!({
            "operation": operation,
            "a": a,
            "b": b,
            "result": result
        }))
        .map_err(|e| ToolError::execution(format!("Failed to create result: {}", e)))?;

        Ok(ToolOutput::Value(result_json))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get API key from environment
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set");

    println!("🤖 Tool Loop Agent Example");
    println!("═══════════════════════════\n");

    // Create the language model
    let provider = OpenAIProvider::new(api_key);
    let model = provider.language_model("gpt-4o-mini").unwrap();

    // Create executable tools
    let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(WeatherTool), Arc::new(CalculatorTool)];

    println!("📋 Available tools:");
    for tool in &tools {
        println!("   • {} - {}", tool.name(), tool.description());
    }
    println!();

    // Create the agent settings
    let settings = ToolLoopAgentSettings::builder(model.clone())
        .id("weather-agent")
        .instructions("You are a helpful assistant with access to weather and calculation tools. Use the tools when needed to answer user questions accurately.")
        .tools(tools)
        .tool_choice(ai_sdk_provider::language_model::ToolChoice::Auto)
        .stop_conditions(vec![step_count_is(10)])
        .on_step_finish(Arc::new(|step| {
            Box::pin(async move {
                println!(
                    "\n📊 Step finished - Tokens used: {}",
                    step.usage.total_tokens.unwrap_or(0)
                );
                if let Some(tool_calls) = &step.tool_calls {
                    println!("   Tool calls: {}", tool_calls.len());
                }
            })
        }))
        .build();

    // Create the agent
    let agent = ToolLoopAgent::new(settings);

    // Test with a weather query
    println!("💬 User: What's the weather like in San Francisco?\n");

    let result = agent
        .generate(AgentCallParameters::from_prompt(
            "What's the weather like in San Francisco?",
        ))
        .await?;

    println!("\n✅ Agent Response:");
    println!("   {}", result.text());
    println!("\n📈 Total Statistics:");
    println!("   • Total steps: {}", result.steps().len());
    println!(
        "   • Total tokens: {}",
        result.usage().total_tokens.unwrap_or(0)
    );

    // Test with a calculation query
    println!("\n{}\n", "─".repeat(60));
    println!("💬 User: What is 15 multiplied by 23?\n");

    let result2 = agent
        .generate(AgentCallParameters::from_prompt(
            "What is 15 multiplied by 23?",
        ))
        .await?;

    println!("\n✅ Agent Response:");
    println!("   {}", result2.text());
    println!("\n📈 Total Statistics:");
    println!("   • Total steps: {}", result2.steps().len());
    println!(
        "   • Total tokens: {}",
        result2.usage().total_tokens.unwrap_or(0)
    );

    // Test with a complex multi-step query
    println!("\n{}\n", "─".repeat(60));
    println!("💬 User: What's the weather in New York and if the temperature is above 70, calculate 70 * 2\n");

    let result3 = agent
        .generate(AgentCallParameters::from_prompt(
            "What's the weather in New York and if the temperature is above 70, calculate 70 * 2",
        ))
        .await?;

    println!("\n✅ Agent Response:");
    println!("   {}", result3.text());
    println!("\n📈 Total Statistics:");
    println!("   • Total steps: {}", result3.steps().len());
    println!(
        "   • Total tokens: {}",
        result3.usage().total_tokens.unwrap_or(0)
    );

    println!("\n🎉 Example completed successfully!");

    Ok(())
}
