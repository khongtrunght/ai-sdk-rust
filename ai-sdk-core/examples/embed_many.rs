//! Example of embedding multiple values with automatic batching
//!
//! To run this example:
//! ```bash
//! export OPENAI_API_KEY=your_key
//! cargo run --example embed_many
//! ```

use ai_sdk_core::embed_many;
use ai_sdk_openai::openai_embedding;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get API key from environment
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set");

    println!("Embedding multiple text values with automatic batching...\n");

    // Create a list of texts to embed
    let texts = vec![
        "Machine learning is a subset of artificial intelligence".to_string(),
        "Rust is a systems programming language".to_string(),
        "Neural networks are inspired by the human brain".to_string(),
        "Deep learning uses multiple layers of neural networks".to_string(),
        "Natural language processing helps computers understand text".to_string(),
        "Computer vision enables machines to interpret images".to_string(),
        "Reinforcement learning trains agents through rewards".to_string(),
        "Transfer learning reuses knowledge from related tasks".to_string(),
    ];

    println!("Embedding {} texts...", texts.len());

    // Embed all values with automatic batching and parallel execution
    let result = embed_many()
        .model(openai_embedding("text-embedding-3-small", api_key))
        .values(texts)
        .max_parallel_calls(3) // Process up to 3 batches in parallel
        .execute()
        .await?;

    println!("\n✓ All embeddings generated successfully");
    println!("  Total embeddings: {}", result.embeddings().len());
    println!(
        "  Embedding dimensions: {}",
        result.embedding(0).map(|e| e.len()).unwrap_or(0)
    );
    println!("  Total tokens used: {}", result.usage().tokens);

    // Calculate and display similarity between first two texts
    if result.embeddings().len() >= 2 {
        let similarity =
            cosine_similarity(result.embedding(0).unwrap(), result.embedding(1).unwrap());
        println!("\nSimilarity between first two texts: {:.4}", similarity);
    }

    // Display sample of embeddings
    println!("\nSample embeddings:");
    for (i, (value, embedding)) in result.iter().take(3).enumerate() {
        let truncated_value = if value.len() > 50 {
            format!("{}...", &value[..50])
        } else {
            value.clone()
        };
        println!("  {}. \"{}\"", i + 1, truncated_value);
        println!(
            "     First 5 values: {:?}",
            &embedding[..5.min(embedding.len())]
        );
    }

    Ok(())
}

/// Calculate cosine similarity between two embeddings
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot_product / (magnitude_a * magnitude_b)
}
