//! Example of using middleware to customize language model behavior
//!
//! This example demonstrates:
//! - DefaultSettingsMiddleware for applying default parameters
//! - SimulateStreamingMiddleware for converting generate to stream
//! - Middleware composition (multiple middlewares in a chain)
//!
//! To run this example:
//! ```bash
//! export OPENAI_API_KEY=your_key
//! cargo run --example middleware_demo
//! ```

use ai_sdk_core::middleware::{
    wrap_language_model, DefaultSettingsMiddleware, SimulateStreamingMiddleware,
};
use ai_sdk_openai::OpenAIChatModel;
use ai_sdk_provider::language_model::{CallOptions, Message, UserContentPart};
use futures::StreamExt;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get API key from environment
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    println!("=== Middleware Demo ===\n");

    // Example 1: DefaultSettingsMiddleware
    println!("1. Using DefaultSettingsMiddleware to set default temperature");
    {
        let base_model = Box::new(OpenAIChatModel::new("gpt-4o-mini", api_key.clone()));

        // Wrap with middleware that sets default temperature
        let wrapped_model = wrap_language_model(
            base_model,
            vec![Box::new(DefaultSettingsMiddleware::new(CallOptions {
                temperature: Some(0.9), // High temperature for creative responses
                max_output_tokens: Some(50),
                ..Default::default()
            }))],
        );

        let response = wrapped_model
            .do_generate(CallOptions {
                prompt: vec![Message::User {
                    content: vec![UserContentPart::Text {
                        text: "Tell me a short joke about programming.".to_string(),
                    }],
                }],
                // Not specifying temperature - will use default from middleware
                ..Default::default()
            })
            .await
            .map_err(|e| format!("{}", e))?;

        println!("Response: {:?}\n", response.content);
    }

    // Example 2: SimulateStreamingMiddleware
    println!("2. Using SimulateStreamingMiddleware to convert generate to stream");
    {
        let base_model = Box::new(OpenAIChatModel::new("gpt-4o-mini", api_key.clone()));

        // Wrap with simulate streaming middleware
        let wrapped_model =
            wrap_language_model(base_model, vec![Box::new(SimulateStreamingMiddleware)]);

        let stream_response = wrapped_model
            .do_stream(CallOptions {
                prompt: vec![Message::User {
                    content: vec![UserContentPart::Text {
                        text: "Count from 1 to 5.".to_string(),
                    }],
                }],
                max_output_tokens: Some(50),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("{}", e))?;

        println!("Streaming response:");
        let mut stream = stream_response.stream;
        while let Some(part) = stream.next().await {
            match part? {
                ai_sdk_provider::language_model::StreamPart::TextDelta { delta, .. } => {
                    print!("{}", delta);
                }
                ai_sdk_provider::language_model::StreamPart::Finish { .. } => {
                    println!("\n[Stream finished]");
                }
                _ => {}
            }
        }
        println!();
    }

    // Example 3: Composing multiple middlewares
    println!("3. Composing DefaultSettings + SimulateStreaming");
    {
        let base_model = Box::new(OpenAIChatModel::new("gpt-4o-mini", api_key.clone()));

        // Compose multiple middlewares
        // First middleware (DefaultSettings) transforms params first
        // Second middleware (SimulateStreaming) wraps the call
        let wrapped_model = wrap_language_model(
            base_model,
            vec![
                Box::new(DefaultSettingsMiddleware::new(CallOptions {
                    temperature: Some(0.7),
                    max_output_tokens: Some(100),
                    ..Default::default()
                })),
                Box::new(SimulateStreamingMiddleware),
            ],
        );

        println!(
            "Using composed middlewares (default temp=0.7, max_tokens=100, simulated streaming):"
        );
        let stream_response = wrapped_model
            .do_stream(CallOptions {
                prompt: vec![Message::User {
                    content: vec![UserContentPart::Text {
                        text: "What is 2+2?".to_string(),
                    }],
                }],
                // Temperature and max_output_tokens will come from DefaultSettingsMiddleware
                ..Default::default()
            })
            .await
            .map_err(|e| format!("{}", e))?;

        let mut stream = stream_response.stream;
        while let Some(part) = stream.next().await {
            match part? {
                ai_sdk_provider::language_model::StreamPart::TextDelta { delta, .. } => {
                    print!("{}", delta);
                }
                ai_sdk_provider::language_model::StreamPart::Finish { .. } => {
                    println!("\n[Stream finished]");
                }
                _ => {}
            }
        }
        println!();
    }

    // Example 4: Override defaults
    println!("4. Overriding default settings");
    {
        let base_model = Box::new(OpenAIChatModel::new("gpt-4o-mini", api_key));

        let wrapped_model = wrap_language_model(
            base_model,
            vec![Box::new(DefaultSettingsMiddleware::new(CallOptions {
                temperature: Some(0.3), // Default: low temperature
                max_output_tokens: Some(50),
                ..Default::default()
            }))],
        );

        let response = wrapped_model
            .do_generate(CallOptions {
                prompt: vec![Message::User {
                    content: vec![UserContentPart::Text {
                        text: "Write one sentence about clouds.".to_string(),
                    }],
                }],
                temperature: Some(1.0), // Override: high temperature for creativity
                ..Default::default()
            })
            .await
            .map_err(|e| format!("{}", e))?;

        println!(
            "Response with overridden temperature: {:?}\n",
            response.content
        );
    }

    println!("=== Demo Complete ===");
    Ok(())
}
