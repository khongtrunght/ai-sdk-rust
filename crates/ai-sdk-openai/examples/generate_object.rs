//! Example of generating structured objects using the OpenAI API.
//!
//! This example demonstrates how to use generate_object to create validated,
//! schema-based outputs from a language model.
//!
//! Set the OPENAI_API_KEY environment variable before running:
//! ```bash
//! export OPENAI_API_KEY="your-api-key-here"
//! cargo run --example generate_object
//! ```

use ai_sdk_core::generate_object::{generate_object, ObjectOutputStrategy};
use ai_sdk_openai::{OpenAIChatModel, OpenAIConfig};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Recipe {
    name: String,
    ingredients: Vec<String>,
    steps: Vec<String>,
    prep_time_minutes: u32,
    servings: u32,
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
            "name": {
                "type": "string",
                "description": "The name of the recipe"
            },
            "ingredients": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "List of ingredients with quantities"
            },
            "steps": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Step-by-step cooking instructions"
            },
            "prep_time_minutes": {
                "type": "integer",
                "description": "Total preparation time in minutes"
            },
            "servings": {
                "type": "integer",
                "description": "Number of servings this recipe makes"
            }
        },
        "required": ["name", "ingredients", "steps", "prep_time_minutes", "servings"],
        "additionalProperties": false
    });

    // Create the output strategy
    let strategy = ObjectOutputStrategy::<Recipe>::new(schema);

    println!("Generating recipe for chocolate chip cookies...\n");

    // Generate the object
    let result = generate_object::<ObjectOutputStrategy<Recipe>>()
        .model(model)
        .prompt("Generate a detailed recipe for chocolate chip cookies. Include exact measurements and clear instructions.")
        .output_strategy(strategy)
        .schema_name("Recipe")
        .schema_description("A cooking recipe with ingredients and instructions")
        .temperature(0.7)
        .execute()
        .await?;

    // Display the result
    println!("Generated Recipe:");
    println!("================");
    println!("Name: {}", result.object.name);
    println!("Servings: {}", result.object.servings);
    println!("Prep Time: {} minutes", result.object.prep_time_minutes);
    println!();

    println!("Ingredients:");
    for (i, ingredient) in result.object.ingredients.iter().enumerate() {
        println!("  {}. {}", i + 1, ingredient);
    }
    println!();

    println!("Instructions:");
    for (i, step) in result.object.steps.iter().enumerate() {
        println!("  {}. {}", i + 1, step);
    }
    println!();

    println!("Token Usage:");
    println!("  Input: {:?}", result.usage.input_tokens);
    println!("  Output: {:?}", result.usage.output_tokens);
    println!("  Total: {:?}", result.usage.total_tokens);

    Ok(())
}
