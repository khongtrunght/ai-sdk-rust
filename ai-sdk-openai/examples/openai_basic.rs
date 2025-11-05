use ai_sdk_openai::*;
use ai_sdk_provider::language_model::{Message, UserContentPart};
use ai_sdk_provider::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;
    let model = openai("gpt-4", api_key);

    let options = CallOptions {
        prompt: vec![
            Message::System {
                content: "You are a helpful assistant.".into(),
            },
            Message::User {
                content: vec![UserContentPart::Text {
                    text: "What is Rust?".into(),
                }],
            },
        ],
        temperature: Some(0.7),
        max_output_tokens: Some(100),
        ..Default::default()
    };

    println!("Generating response...");
    let response = model.do_generate(options).await?;

    println!("Finish reason: {:?}", response.finish_reason);
    println!("Usage: {:?}", response.usage);

    for content in &response.content {
        if let Content::Text(text) = content {
            println!("\nResponse:\n{}", text.text);
        }
    }

    Ok(())
}
