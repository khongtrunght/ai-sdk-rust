use ai_sdk_openai::OpenAITranscriptionModel;
use ai_sdk_provider::{AudioInput, TranscriptionModel, TranscriptionOptions};

#[tokio::test]
#[ignore] // Requires OPENAI_API_KEY and test audio file
async fn test_openai_transcription_with_binary_audio() {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");

    let model = OpenAITranscriptionModel::new("whisper-1", api_key);

    // Load a test audio file (you need to provide a sample audio file)
    // For actual testing, place a test audio file in the tests directory
    let audio_data = std::fs::read("tests/test_audio.mp3")
        .expect("test_audio.mp3 not found - please provide a test audio file");

    let options = TranscriptionOptions {
        audio: AudioInput::Binary(audio_data),
        media_type: "audio/mpeg".into(),
        provider_options: None,
        abort_signal: None,
        headers: None,
    };

    let response = model.do_generate(options).await.unwrap();

    assert!(
        !response.text.is_empty(),
        "Transcription text should not be empty"
    );
    println!("Transcribed text: {}", response.text);

    if !response.segments.is_empty() {
        println!("Number of segments: {}", response.segments.len());
        assert!(response.segments[0].start_second >= 0.0);
        assert!(response.segments[0].end_second > response.segments[0].start_second);
    }

    if let Some(lang) = &response.language {
        println!("Detected language: {}", lang);
        assert_eq!(
            lang.len(),
            2,
            "Language code should be ISO-639-1 (2 characters)"
        );
    }

    if let Some(duration) = response.duration_in_seconds {
        println!("Audio duration: {} seconds", duration);
        assert!(duration > 0.0);
    }

    assert_eq!(response.response.model_id, "whisper-1");
    assert!(response.warnings.is_empty());
}

#[tokio::test]
#[ignore] // Requires OPENAI_API_KEY and test audio file
async fn test_openai_transcription_with_base64_audio() {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");

    let model = OpenAITranscriptionModel::new("whisper-1", api_key);

    // Load and encode audio to base64
    let audio_bytes = std::fs::read("tests/test_audio.mp3")
        .expect("test_audio.mp3 not found - please provide a test audio file");

    use base64::Engine;
    let base64_audio = base64::engine::general_purpose::STANDARD.encode(&audio_bytes);

    let options = TranscriptionOptions {
        audio: AudioInput::Base64(base64_audio),
        media_type: "audio/mpeg".into(),
        provider_options: None,
        abort_signal: None,
        headers: None,
    };

    let response = model.do_generate(options).await.unwrap();

    assert!(!response.text.is_empty());
    assert_eq!(response.response.model_id, "whisper-1");
}

#[tokio::test]
#[ignore] // Requires OPENAI_API_KEY and test audio file
async fn test_openai_transcription_gpt4o_model() {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");

    // Test with gpt-4o-mini-transcribe which uses "json" format instead of "verbose_json"
    let model = OpenAITranscriptionModel::new("gpt-4o-mini-transcribe", api_key);

    let audio_data = std::fs::read("tests/test_audio.mp3")
        .expect("test_audio.mp3 not found - please provide a test audio file");

    let options = TranscriptionOptions {
        audio: AudioInput::Binary(audio_data),
        media_type: "audio/mpeg".into(),
        provider_options: None,
        abort_signal: None,
        headers: None,
    };

    let response = model.do_generate(options).await.unwrap();

    assert!(!response.text.is_empty());
    assert_eq!(response.response.model_id, "gpt-4o-mini-transcribe");
}

#[tokio::test]
#[ignore] // Requires OPENAI_API_KEY and test audio file
async fn test_transcription_with_different_media_types() {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");

    let _model = OpenAITranscriptionModel::new("whisper-1", api_key);

    // Test with different media types - ensure file extension is correctly determined
    let test_cases = vec![
        ("audio/mpeg", "mp3"),
        ("audio/wav", "wav"),
        ("audio/mp4", "m4a"),
        ("audio/webm", "webm"),
    ];

    for (media_type, _expected_ext) in test_cases {
        // In a real test, you'd have different audio files
        // For now, we just verify the model accepts different media types
        println!("Testing media type: {}", media_type);

        // This would fail without the actual file, but the test structure is here
        // You can uncomment when you have test files:
        // let audio_data = std::fs::read(format!("tests/test_audio.{}", expected_ext)).unwrap();
        // let options = TranscriptionOptions {
        //     audio: AudioInput::Binary(audio_data),
        //     media_type: media_type.into(),
        //     provider_options: None,
        //     abort_signal: None,
        //     headers: None,
        // };
        // let response = model.do_generate(options).await.unwrap();
        // assert!(!response.text.is_empty());
    }
}

#[test]
fn test_model_metadata() {
    let model = OpenAITranscriptionModel::new("whisper-1", "test-key");

    assert_eq!(model.provider(), "openai");
    assert_eq!(model.model_id(), "whisper-1");
    assert_eq!(model.specification_version(), "v3");
}
