//! Example of streaming structured objects using the OpenAI API.
//!
//! This example demonstrates how to use stream_object to receive partial
//! updates as the model generates a structured object.
//!
//! Set the OPENAI_API_KEY environment variable before running:
//! ```bash
//! export OPENAI_API_KEY="your-api-key-here"
//! cargo run --example stream_object
//! ```

use ai_sdk_core::generate_object::{stream_object, ObjectOutputStrategy, ObjectStreamPart};
use ai_sdk_openai::{OpenAIChatModel, OpenAIConfig};
use serde::{Deserialize, Serialize};
use tokio_stream::StreamExt;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Story {
    title: String,
    characters: Vec<String>,
    plot: String,
    setting: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get API key from environment
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable must be set");

    // Create the model
    let model = OpenAIChatModel::new("gpt-4o-mini", OpenAIConfig::from_api_key(api_key));

    // Define the JSON schema for the output
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "The title of the story"
            },
            "characters": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Main characters in the story"
            },
            "plot": {
                "type": "string",
                "description": "Brief plot summary"
            },
            "setting": {
                "type": "string",
                "description": "Where and when the story takes place"
            }
        },
        "required": ["title", "characters", "plot", "setting"],
        "additionalProperties": false
    });

    // Create the output strategy
    let strategy = ObjectOutputStrategy::<Story>::new(schema);

    println!("Streaming story generation...\n");

    // Stream the object
    let mut result = stream_object::<ObjectOutputStrategy<Story>>()
        .model(model)
        .prompt(
            "Write a short science fiction story outline about a robot learning to feel emotions.",
        )
        .output_strategy(strategy)
        .schema_name("Story")
        .schema_description("A story outline with title, characters, plot, and setting")
        .temperature(0.8)
        .execute()
        .await?;

    // Process streaming updates
    println!("Receiving partial updates:");
    println!("==========================\n");

    let mut update_count = 0;
    while let Some(part) = result.partial_object_stream.next().await {
        match part {
            ObjectStreamPart::Object { object } => {
                update_count += 1;
                println!("Update #{}:", update_count);
                println!("  Title: {}", object.title);
                println!("  Characters: {:?}", object.characters);
                if !object.plot.is_empty() {
                    println!(
                        "  Plot: {}...",
                        object.plot.chars().take(50).collect::<String>()
                    );
                }
                if !object.setting.is_empty() {
                    println!("  Setting: {}", object.setting);
                }
                println!();
            }
            ObjectStreamPart::TextDelta { text_delta: _ } => {
                // Could display the text delta if desired
            }
            ObjectStreamPart::Finish { finish_reason } => {
                println!("Stream finished: {:?}\n", finish_reason);
            }
            ObjectStreamPart::Error { error } => {
                eprintln!("Error: {}", error);
            }
        }
    }

    // Get the final object
    let final_story = result.object.await?;
    let final_usage = result.usage.await?;

    println!("Final Story:");
    println!("============");
    println!("Title: {}", final_story.title);
    println!("Setting: {}", final_story.setting);
    println!();
    println!("Characters:");
    for character in &final_story.characters {
        println!("  - {}", character);
    }
    println!();
    println!("Plot:");
    println!("{}", final_story.plot);
    println!();

    println!("Token Usage:");
    println!("  Input: {:?}", final_usage.input_tokens);
    println!("  Output: {:?}", final_usage.output_tokens);
    println!("  Total: {:?}", final_usage.total_tokens);

    Ok(())
}
