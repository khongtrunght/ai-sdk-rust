use ai_sdk_openai::{OpenAIConfig, OpenAIProvider, OpenAISpeechModel};
use ai_sdk_provider::{AudioData, ProviderV3, SpeechGenerateOptions, SpeechModel};

#[tokio::test]
#[ignore] // Requires OPENAI_API_KEY
async fn test_openai_speech_generation() {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");

    let provider = OpenAIProvider::builder().with_api_key(api_key).build();
    let model = provider.speech_model("tts-1").unwrap();

    let options = SpeechGenerateOptions {
        text: "Hello, this is a test of text to speech.".into(),
        voice: Some("alloy".into()),
        output_format: Some("mp3".into()),
        instructions: None,
        speed: None,
        language: None,
        provider_options: None,
        abort_signal: None,
        headers: None,
    };

    let response = model.do_generate(options).await.unwrap();

    match response.audio {
        AudioData::Binary(data) => {
            assert!(!data.is_empty(), "Audio data should not be empty");
            println!("Generated audio size: {} bytes", data.len());
        }
        AudioData::Base64(_) => panic!("Expected binary data, got base64"),
    }

    assert_eq!(response.response.model_id, "tts-1");
}

#[tokio::test]
#[ignore] // Requires OPENAI_API_KEY
async fn test_openai_speech_with_speed() {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");

    let provider = OpenAIProvider::builder().with_api_key(api_key).build();
    let model = provider.speech_model("tts-1-hd").unwrap();

    let options = SpeechGenerateOptions {
        text: "Testing speed control.".into(),
        voice: Some("nova".into()),
        output_format: Some("mp3".into()),
        instructions: None,
        speed: Some(1.5),
        language: None,
        provider_options: None,
        abort_signal: None,
        headers: None,
    };

    let response = model.do_generate(options).await.unwrap();

    match response.audio {
        AudioData::Binary(data) => {
            assert!(!data.is_empty());
        }
        AudioData::Base64(_) => panic!("Expected binary data"),
    }
}

#[tokio::test]
#[ignore] // Requires OPENAI_API_KEY
async fn test_openai_speech_unsupported_format_warning() {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");

    let provider = OpenAIProvider::builder().with_api_key(api_key).build();
    let model = provider.speech_model("tts-1").unwrap();

    let options = SpeechGenerateOptions {
        text: "Testing unsupported format.".into(),
        voice: Some("alloy".into()),
        output_format: Some("ogg".into()), // Unsupported format
        instructions: None,
        speed: None,
        language: None,
        provider_options: None,
        abort_signal: None,
        headers: None,
    };

    let response = model.do_generate(options).await.unwrap();

    // Should have a warning about unsupported format
    assert!(!response.warnings.is_empty());
    assert!(response
        .warnings
        .iter()
        .any(|w| matches!(w, ai_sdk_provider::SpeechCallWarning::UnsupportedSetting { setting, .. } if setting == "outputFormat")));
}

#[tokio::test]
#[ignore] // Requires OPENAI_API_KEY
async fn test_openai_speech_language_warning() {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");

    let model = OpenAISpeechModel::new("tts-1", OpenAIConfig::from_api_key(api_key));

    let options = SpeechGenerateOptions {
        text: "Testing language parameter.".into(),
        voice: Some("alloy".into()),
        output_format: Some("mp3".into()),
        instructions: None,
        speed: None,
        language: Some("en".into()), // Language is not supported
        provider_options: None,
        abort_signal: None,
        headers: None,
    };

    let response = model.do_generate(options).await.unwrap();

    // Should have a warning about language not being supported
    assert!(!response.warnings.is_empty());
    assert!(response
        .warnings
        .iter()
        .any(|w| matches!(w, ai_sdk_provider::SpeechCallWarning::UnsupportedSetting { setting, .. } if setting == "language")));
}
