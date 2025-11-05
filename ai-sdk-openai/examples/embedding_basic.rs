use ai_sdk_openai::OpenAIEmbeddingModel;
use ai_sdk_provider::{EmbedOptions, EmbeddingModel};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?;
    let model = OpenAIEmbeddingModel::new("text-embedding-3-small", api_key);

    let options = EmbedOptions {
        values: vec![
            "The cat sat on the mat".into(),
            "A feline rested on the rug".into(),
        ],
        provider_options: None,
        headers: None,
    };

    let response = model.do_embed(options).await?;

    println!("Generated {} embeddings", response.embeddings.len());
    println!(
        "Embedding dimension: {}",
        response.embeddings[0].len()
    );

    if let Some(usage) = &response.usage {
        println!("Tokens used: {}", usage.tokens);
    }

    // Calculate similarity between the two embeddings
    let similarity = cosine_similarity(&response.embeddings[0], &response.embeddings[1]);
    println!("\nCosine similarity between the two texts: {:.4}", similarity);
    println!("(Values closer to 1.0 indicate more similar texts)");

    Ok(())
}

// Helper function to calculate cosine similarity
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot_product / (magnitude_a * magnitude_b)
}
