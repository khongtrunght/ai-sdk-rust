use ai_sdk_core::JsonValue;
use ai_sdk_provider::SharedProviderOptions;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenAITranscriptionProviderOptions {
    /// Additional information to include in the transcription response.
    pub include: Option<Vec<String>>,

    /// The language of the input audio in ISO-639-1 format.
    pub language: Option<String>,

    /// An optional text to guide the model's style or continue a previous audio segment.
    pub prompt: Option<String>,

    /// The sampling temperature, between 0 and 1.
    pub temperature: Option<f64>,

    /// The timestamp granularities to populate for this transcription.
    pub timestamp_granularities: Option<Vec<String>>,
}

impl From<Option<SharedProviderOptions>> for OpenAITranscriptionProviderOptions {
    fn from(opts: Option<SharedProviderOptions>) -> Self {
        match opts {
            None => Self {
                include: None,
                language: None,
                prompt: None,
                temperature: None,
                timestamp_granularities: None,
            },
            Some(opts) => {
                let openai_opts = opts.get("openai");
                Self {
                    include: openai_opts
                        .and_then(|o| o.get("include"))
                        .and_then(|i| match i {
                            JsonValue::Array(arr) => Some(
                                arr.iter()
                                    .filter_map(|v| {
                                        if let JsonValue::String(s) = v {
                                            Some(s.clone())
                                        } else {
                                            None
                                        }
                                    })
                                    .collect(),
                            ),
                            _ => None,
                        }),
                    language: openai_opts
                        .and_then(|o| o.get("language"))
                        .and_then(|l| match l {
                            JsonValue::String(s) => Some(s.clone()),
                            _ => None,
                        }),
                    prompt: openai_opts
                        .and_then(|o| o.get("prompt"))
                        .and_then(|p| match p {
                            JsonValue::String(s) => Some(s.clone()),
                            _ => None,
                        }),
                    temperature: openai_opts.and_then(|o| o.get("temperature")).and_then(
                        |t| match t {
                            JsonValue::Number(n) => Some(n.as_f64().unwrap_or(0.0)),
                            _ => None,
                        },
                    ),
                    timestamp_granularities: openai_opts
                        .and_then(|o| o.get("timestamp_granularities"))
                        .and_then(|t| match t {
                            JsonValue::Array(arr) => Some(
                                arr.iter()
                                    .filter_map(|v| {
                                        if let JsonValue::String(s) = v {
                                            Some(s.clone())
                                        } else {
                                            None
                                        }
                                    })
                                    .collect(),
                            ),
                            _ => None,
                        }),
                }
            }
        }
    }
}
