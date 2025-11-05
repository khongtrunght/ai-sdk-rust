use ai_sdk_openai::OpenAIImageModel;
use ai_sdk_provider::{ImageData, ImageGenerateOptions, ImageModel};
use base64::Engine;
use std::fs::File;
use std::io::Write;

#[tokio::main]
async fn main() {
    match run().await {
        Ok(_) => println!("Success!"),
        Err(e) => eprintln!("Error: {}", e),
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let api_key = std::env::var("OPENAI_API_KEY")?;
    let model = OpenAIImageModel::new("dall-e-3", api_key);

    let options = ImageGenerateOptions {
        prompt: "A serene landscape with mountains and a lake at sunset".into(),
        n: 1,
        size: Some("1024x1024".into()),
        aspect_ratio: None,
        seed: None,
        provider_options: None,
        headers: None,
    };

    println!("Generating image...");
    let response = model.do_generate(options).await?;

    println!("Generated {} images", response.images.len());
    println!("Warnings: {:?}", response.warnings);

    // Save first image
    if let ImageData::Base64(data) = &response.images[0] {
        let decoded = base64::engine::general_purpose::STANDARD.decode(data)?;
        let mut file = File::create("generated_image.png")?;
        file.write_all(&decoded)?;
        println!("Saved to generated_image.png");
        println!("Image size: {} bytes", decoded.len());
    }

    // Print provider metadata if available
    if let Some(metadata) = response.provider_metadata {
        println!("Provider metadata: {:?}", metadata);
    }

    Ok(())
}
