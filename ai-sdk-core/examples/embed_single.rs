//! Example of embedding a single value using the embed() API
//!
//! To run this example:
//! ```bash
//! export OPENAI_API_KEY=your_key
//! cargo run --example embed_single
//! ```

use ai_sdk_core::embed;
use ai_sdk_openai::openai_embedding;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get API key from environment
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set");

    println!("Embedding a single text value...\n");

    // Embed a single value
    let result = embed()
        .model(openai_embedding("text-embedding-3-small", api_key))
        .value("Hello, world! This is a test of the embedding API.".to_string())
        .execute()
        .await?;

    println!("✓ Embedding generated successfully");
    println!("  Dimensions: {}", result.embedding().len());
    println!(
        "  Token usage: {}",
        result.usage().map(|u| u.tokens).unwrap_or(0)
    );
    println!(
        "  First 5 values: {:?}",
        &result.embedding()[..5.min(result.embedding().len())]
    );
    println!("\nOriginal value: {}", result.value());

    Ok(())
}
