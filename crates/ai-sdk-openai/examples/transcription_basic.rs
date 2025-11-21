use ai_sdk_openai::OpenAIProvider;
use ai_sdk_provider::{AudioInput, ProviderV3, TranscriptionOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get API key from environment
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable must be set");

    // Create the model
    let provider = OpenAIProvider::builder().with_api_key(api_key).build();
    let model = provider.transcription_model("whisper-1").unwrap();

    println!("OpenAI Transcription Example");
    println!("============================\n");

    // Check if a sample audio file exists
    let audio_file_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "sample.mp3".to_string());

    println!("Loading audio file: {}", audio_file_path);

    // Load audio file
    let audio_data = std::fs::read(&audio_file_path).map_err(|e| {
        format!(
            "Failed to read audio file '{}': {}\n\nUsage: cargo run --example transcription_basic <path-to-audio-file>",
            audio_file_path, e
        )
    })?;

    println!("Audio file size: {} bytes\n", audio_data.len());

    // Create transcription options
    let options = TranscriptionOptions {
        audio: AudioInput::Binary(audio_data),
        media_type: "audio/mpeg".into(),
        provider_options: None,
        abort_signal: None,
        headers: None,
    };

    // Generate transcription
    println!("Transcribing audio...\n");
    let response = model
        .do_generate(options)
        .await
        .map_err(|e| format!("Transcription failed: {}", e))?;

    // Print results
    println!("Transcription Results");
    println!("====================\n");

    println!("Text: {}\n", response.text);

    if let Some(lang) = &response.language {
        println!("Detected Language: {} (ISO-639-1)", lang);
    }

    if let Some(duration) = response.duration_in_seconds {
        println!("Duration: {:.2} seconds", duration);
    }

    println!("\nSegments: {}", response.segments.len());
    if !response.segments.is_empty() {
        println!("\nDetailed Segments:");
        println!("-----------------");
        for (i, segment) in response.segments.iter().enumerate() {
            println!(
                "{}. [{:.2}s - {:.2}s] {}",
                i + 1,
                segment.start_second,
                segment.end_second,
                segment.text
            );
        }
    }

    if !response.warnings.is_empty() {
        println!("\nWarnings: {:?}", response.warnings);
    }

    println!("\nModel: {}", response.response.model_id);
    println!("Timestamp: {:?}", response.response.timestamp);

    Ok(())
}
