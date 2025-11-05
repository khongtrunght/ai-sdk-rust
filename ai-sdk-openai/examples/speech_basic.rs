use ai_sdk_openai::OpenAISpeechModel;
use ai_sdk_provider::{AudioData, SpeechGenerateOptions, SpeechModel};
use std::fs::File;
use std::io::Write;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Get API key from environment
    let api_key = std::env::var("OPENAI_API_KEY")?;

    // Create the speech model
    let model = OpenAISpeechModel::new("tts-1", api_key);

    println!("Generating speech from text...");

    // Configure the speech generation options
    let options = SpeechGenerateOptions {
        text: "Welcome to the AI SDK in Rust! This is a demonstration of text to speech synthesis using OpenAI's API.".into(),
        voice: Some("nova".into()),
        output_format: Some("mp3".into()),
        instructions: None,
        speed: Some(1.0),
        language: None,
        provider_options: None,
        abort_signal: None,
        headers: None,
    };

    // Generate the speech
    let response = model.do_generate(options).await?;

    // Print response info
    println!("Model ID: {}", response.response.model_id);
    println!("Timestamp: {:?}", response.response.timestamp);

    // Print warnings if any
    if !response.warnings.is_empty() {
        println!("\nWarnings:");
        for warning in &response.warnings {
            println!("  {:?}", warning);
        }
    }

    // Save the audio to a file
    if let AudioData::Binary(data) = response.audio {
        let mut file = File::create("speech_output.mp3")?;
        file.write_all(&data)?;
        println!("\n✓ Saved audio to speech_output.mp3");
        println!("  Audio size: {} bytes", data.len());
        println!("\nYou can play the file with: mpv speech_output.mp3");
    }

    Ok(())
}
