use ai_sdk_openai::responses::OpenAIResponsesLanguageModel;
use ai_sdk_openai::{OpenAIConfig, OpenAIUrlOptions};
use ai_sdk_provider::language_model::{CallOptions, LanguageModel, Message, UserContentPart};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Get API key from environment
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set");

    // Create OpenAI configuration
    let config = OpenAIConfig {
        provider: "openai".to_string(),
        url: Arc::new(|opts: OpenAIUrlOptions| format!("https://api.openai.com/v1{}", opts.path)),
        headers: Arc::new(move || {
            let mut headers = std::collections::HashMap::new();
            headers.insert("Authorization".to_string(), format!("Bearer {}", api_key));
            headers
        }),
        generate_id: None,
        file_id_prefixes: Some(vec!["file-".to_string()]),
    };

    // Create model using the Responses API
    let model = OpenAIResponsesLanguageModel::new("gpt-4o", config);

    // Create a simple prompt
    let options = CallOptions {
        prompt: vec![Message::User {
            content: vec![UserContentPart::Text {
                text: "What is the capital of France?".to_string(),
            }],
        }],
        max_output_tokens: Some(50),
        ..Default::default()
    };

    // Generate response
    println!("Sending request to OpenAI Responses API...");
    let response = model.do_generate(options).await?;

    // Print response
    println!("\nResponse:");
    for content in &response.content {
        if let ai_sdk_provider::language_model::Content::Text(text) = content {
            println!("{}", text.text);
        }
    }

    // Print usage
    println!("\nUsage:");
    println!(
        "  Input tokens: {}",
        response.usage.input_tokens.unwrap_or(0)
    );
    println!(
        "  Output tokens: {}",
        response.usage.output_tokens.unwrap_or(0)
    );
    println!(
        "  Total tokens: {}",
        response.usage.total_tokens.unwrap_or(0)
    );

    // Print warnings if any
    if !response.warnings.is_empty() {
        println!("\nWarnings:");
        for warning in &response.warnings {
            println!("  - {}", warning.message);
        }
    }

    println!("\nFinish reason: {:?}", response.finish_reason);

    Ok(())
}
