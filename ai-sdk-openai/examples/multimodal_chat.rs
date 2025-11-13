/// Example demonstrating multi-modal chat with images and audio
///
/// This example shows how to send images (both URLs and binary data) to OpenAI's
/// vision models like GPT-4o.
///
/// Usage:
/// ```
/// OPENAI_API_KEY=your-key-here cargo run --example multimodal_chat
/// ```
use ai_sdk_openai::OpenAIChatModel;
use ai_sdk_provider::language_model::{
    CallOptions, FileData, LanguageModel, Message, UserContentPart,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Get API key from environment
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable must be set");

    // Create a vision-capable model
    let model = OpenAIChatModel::new("gpt-4o", api_key);

    println!("=== Multi-Modal Chat Example ===\n");

    // Example 1: Image from URL
    println!("Example 1: Analyzing an image from URL");
    let messages_url = vec![Message::User {
        content: vec![
            UserContentPart::Text {
                text: "What's in this image?".to_string(),
            },
            UserContentPart::File {
                data: FileData::Url(
                    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg".to_string()
                ),
                media_type: "image/jpeg".to_string(),
            },
        ],
    }];

    let response = model
        .do_generate(CallOptions {
            prompt: messages_url,
            max_output_tokens: Some(300),
            ..Default::default()
        })
        .await?;

    println!("Response: {:?}\n", response.content);

    // Example 2: Image from binary data (create a simple 1x1 PNG)
    println!("Example 2: Analyzing binary image data");

    // This is a minimal valid 1x1 red pixel PNG
    let png_data = vec![
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR chunk
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, // 1x1 dimensions
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, // bit depth, color type
        0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, // IDAT chunk
        0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00, // pixel data (red)
        0x00, 0x03, 0x01, 0x01, 0x00, 0x18, 0xDD, 0x8D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,
        0x4E, // IEND chunk
        0x44, 0xAE, 0x42, 0x60, 0x82,
    ];

    let messages_binary = vec![Message::User {
        content: vec![
            UserContentPart::Text {
                text: "What color is this pixel?".to_string(),
            },
            UserContentPart::File {
                data: FileData::Binary(png_data),
                media_type: "image/png".to_string(),
            },
        ],
    }];

    let response = model
        .do_generate(CallOptions {
            prompt: messages_binary,
            max_output_tokens: Some(100),
            ..Default::default()
        })
        .await?;

    println!("Response: {:?}\n", response.content);

    // Example 3: Multiple images in one message
    println!("Example 3: Multiple images in one message");
    let messages_multiple = vec![Message::User {
        content: vec![
            UserContentPart::Text {
                text: "Compare these two images:".to_string(),
            },
            UserContentPart::File {
                data: FileData::Url(
                    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg".to_string()
                ),
                media_type: "image/jpeg".to_string(),
            },
            UserContentPart::File {
                data: FileData::Url(
                    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/1920px-Camponotus_flavomarginatus_ant.jpg".to_string()
                ),
                media_type: "image/jpeg".to_string(),
            },
            UserContentPart::Text {
                text: "What's the main difference?".to_string(),
            },
        ],
    }];

    let response = model
        .do_generate(CallOptions {
            prompt: messages_multiple,
            max_output_tokens: Some(300),
            ..Default::default()
        })
        .await?;

    println!("Response: {:?}\n", response.content);

    println!("=== Example Complete ===");
    Ok(())
}
