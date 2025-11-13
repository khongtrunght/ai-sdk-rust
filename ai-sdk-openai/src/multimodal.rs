/// OpenAI multi-modal content conversion
///
/// This module handles the conversion of file content (images, audio) to OpenAI's API format.
use ai_sdk_provider::language_model::FileData;
use serde::{Deserialize, Serialize};

/// OpenAI content part (can be text, image, or audio)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAIContentPart {
    /// Text content
    Text {
        /// The text content
        text: String,
    },
    /// Image URL (can be actual URL or data URL)
    ImageUrl {
        /// Image URL structure
        image_url: ImageUrl,
    },
    /// Input audio (base64 encoded)
    InputAudio {
        /// Audio input structure
        input_audio: InputAudio,
    },
}

/// Image URL structure for OpenAI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    /// URL (can be https:// or data:)
    pub url: String,
    /// Detail level: "low", "high", or "auto"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Input audio structure for OpenAI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputAudio {
    /// Base64-encoded audio data
    pub data: String,
    /// Audio format: "wav" or "mp3"
    pub format: String,
}

/// Error types for multimodal conversion
#[derive(Debug, thiserror::Error)]
pub enum MultimodalError {
    /// Audio URLs are not supported by OpenAI
    #[error("Audio URLs are not supported, only binary audio data")]
    AudioUrlNotSupported,

    /// Unsupported audio format
    #[error("Unsupported audio format: {0}")]
    UnsupportedAudioFormat(String),

    /// Unsupported media type
    #[error("Unsupported media type: {0}")]
    UnsupportedMediaType(String),
}

/// Converts an image file to OpenAI format
///
/// Supports both URLs and binary data. Binary data is converted to data URLs.
pub fn convert_image_part(
    data: &FileData,
    media_type: &str,
) -> Result<OpenAIContentPart, MultimodalError> {
    match data {
        FileData::Url(url) => {
            // Direct URL passthrough
            Ok(OpenAIContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: url.clone(),
                    detail: None,
                },
            })
        }
        FileData::Binary(bytes) => {
            // Create data URL from binary data
            let base64_data = ai_sdk_core::util::encode_base64(bytes);
            let data_url = format!("data:{};base64,{}", media_type, base64_data);
            Ok(OpenAIContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: data_url,
                    detail: None,
                },
            })
        }
    }
}

/// Converts an audio file to OpenAI format
///
/// Only supports binary audio data, not URLs.
/// Supported formats: WAV, MP3
pub fn convert_audio_part(
    data: &FileData,
    media_type: &str,
) -> Result<OpenAIContentPart, MultimodalError> {
    let FileData::Binary(bytes) = data else {
        return Err(MultimodalError::AudioUrlNotSupported);
    };

    // Determine audio format from media type
    let format = match media_type {
        "audio/wav" | "audio/wave" => "wav",
        "audio/mp3" | "audio/mpeg" => "mp3",
        _ => {
            return Err(MultimodalError::UnsupportedAudioFormat(
                media_type.to_string(),
            ))
        }
    };

    let base64_data = ai_sdk_core::util::encode_base64(bytes);

    Ok(OpenAIContentPart::InputAudio {
        input_audio: InputAudio {
            data: base64_data,
            format: format.to_string(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialization_text() {
        let part = OpenAIContentPart::Text {
            text: "Hello".to_string(),
        };
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(json["type"], "text");
        assert_eq!(json["text"], "Hello");
    }

    #[test]
    fn test_serialization_image_url() {
        let part = OpenAIContentPart::ImageUrl {
            image_url: ImageUrl {
                url: "https://example.com/image.jpg".to_string(),
                detail: None,
            },
        };
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(json["type"], "image_url");
        assert_eq!(json["image_url"]["url"], "https://example.com/image.jpg");
    }

    #[test]
    fn test_serialization_input_audio() {
        let part = OpenAIContentPart::InputAudio {
            input_audio: InputAudio {
                data: "base64data".to_string(),
                format: "wav".to_string(),
            },
        };
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(json["type"], "input_audio");
        assert_eq!(json["input_audio"]["format"], "wav");
    }

    #[test]
    fn test_convert_image_url() {
        let data = FileData::Url("https://example.com/image.jpg".to_string());
        let result = convert_image_part(&data, "image/jpeg").unwrap();

        match result {
            OpenAIContentPart::ImageUrl { image_url } => {
                assert_eq!(image_url.url, "https://example.com/image.jpg");
            }
            _ => panic!("Expected ImageUrl"),
        }
    }

    #[test]
    fn test_convert_image_binary() {
        let data = FileData::Binary(vec![1, 2, 3, 4]);
        let result = convert_image_part(&data, "image/png").unwrap();

        match result {
            OpenAIContentPart::ImageUrl { image_url } => {
                assert!(image_url.url.starts_with("data:image/png;base64,"));
            }
            _ => panic!("Expected ImageUrl"),
        }
    }

    #[test]
    fn test_convert_audio_wav() {
        let data = FileData::Binary(vec![1, 2, 3, 4]);
        let result = convert_audio_part(&data, "audio/wav").unwrap();

        match result {
            OpenAIContentPart::InputAudio { input_audio } => {
                assert_eq!(input_audio.format, "wav");
                assert!(!input_audio.data.is_empty());
            }
            _ => panic!("Expected InputAudio"),
        }
    }

    #[test]
    fn test_convert_audio_mp3() {
        let data = FileData::Binary(vec![1, 2, 3, 4]);
        let result = convert_audio_part(&data, "audio/mpeg").unwrap();

        match result {
            OpenAIContentPart::InputAudio { input_audio } => {
                assert_eq!(input_audio.format, "mp3");
            }
            _ => panic!("Expected InputAudio"),
        }
    }

    #[test]
    fn test_convert_audio_url_not_supported() {
        let data = FileData::Url("https://example.com/audio.mp3".to_string());
        let result = convert_audio_part(&data, "audio/mp3");
        assert!(matches!(result, Err(MultimodalError::AudioUrlNotSupported)));
    }

    #[test]
    fn test_convert_audio_unsupported_format() {
        let data = FileData::Binary(vec![1, 2, 3, 4]);
        let result = convert_audio_part(&data, "audio/ogg");
        assert!(matches!(
            result,
            Err(MultimodalError::UnsupportedAudioFormat(_))
        ));
    }
}
