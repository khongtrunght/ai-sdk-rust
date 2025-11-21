use ai_sdk_core::JsonValue;
use ai_sdk_provider::SharedProviderOptions;

/// OpenAI-specific options for speech synthesis
#[derive(Debug, Clone, Default)]
pub struct OpenAISpeechProviderOptions {
    /// Additional instructions for the model on how to speak the text.
    ///
    /// This can include guidance on tone, pacing, or specific pronunciation.
    pub instructions: Option<String>,

    /// The speed of the generated audio.
    ///
    /// Valid range: 0.25 to 4.0
    /// Default: 1.0
    ///
    /// Values less than 1.0 will slow down the speech,
    /// values greater than 1.0 will speed it up.
    pub speed: Option<f32>,
}

impl From<Option<SharedProviderOptions>> for OpenAISpeechProviderOptions {
    fn from(opts: Option<SharedProviderOptions>) -> Self {
        match opts {
            None => Self::default(),
            Some(opts) => {
                let openai_opts = opts.get("openai");
                Self {
                    instructions: openai_opts
                        .and_then(|o| o.get("instructions"))
                        .and_then(|i| match i {
                            JsonValue::String(s) => Some(s.clone()),
                            _ => None,
                        }),
                    speed: openai_opts
                        .and_then(|o| o.get("speed"))
                        .and_then(|s| match s {
                            JsonValue::Number(n) => {
                                let speed = n.as_f64().unwrap_or(1.0) as f32;
                                // Validate range: 0.25 to 4.0
                                if (0.25..=4.0).contains(&speed) {
                                    Some(speed)
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        }),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_sdk_core::JsonValue;
    use std::collections::HashMap;

    #[test]
    fn test_default_options() {
        let opts = OpenAISpeechProviderOptions::from(None);
        assert!(opts.instructions.is_none());
        assert!(opts.speed.is_none());
    }

    #[test]
    fn test_parse_instructions() {
        let mut provider_opts = HashMap::new();
        let mut openai_opts = HashMap::new();
        openai_opts.insert(
            "instructions".to_string(),
            JsonValue::String("Speak slowly and clearly".to_string()),
        );
        provider_opts.insert("openai".to_string(), openai_opts);

        let opts = OpenAISpeechProviderOptions::from(Some(provider_opts));
        assert_eq!(
            opts.instructions,
            Some("Speak slowly and clearly".to_string())
        );
    }

    #[test]
    fn test_parse_speed() {
        let mut provider_opts = HashMap::new();
        let mut openai_opts = HashMap::new();
        openai_opts.insert(
            "speed".to_string(),
            JsonValue::Number(serde_json::Number::from_f64(1.5).unwrap()),
        );
        provider_opts.insert("openai".to_string(), openai_opts);

        let opts = OpenAISpeechProviderOptions::from(Some(provider_opts));
        assert_eq!(opts.speed, Some(1.5));
    }

    #[test]
    fn test_speed_validation() {
        // Test speed too low
        let mut provider_opts = HashMap::new();
        let mut openai_opts = HashMap::new();
        openai_opts.insert(
            "speed".to_string(),
            JsonValue::Number(serde_json::Number::from_f64(0.1).unwrap()),
        );
        provider_opts.insert("openai".to_string(), openai_opts.clone());

        let opts = OpenAISpeechProviderOptions::from(Some(provider_opts));
        assert!(opts.speed.is_none()); // Out of range, should be None

        // Test speed too high
        let mut provider_opts = HashMap::new();
        openai_opts.insert(
            "speed".to_string(),
            JsonValue::Number(serde_json::Number::from_f64(5.0).unwrap()),
        );
        provider_opts.insert("openai".to_string(), openai_opts);

        let opts = OpenAISpeechProviderOptions::from(Some(provider_opts));
        assert!(opts.speed.is_none()); // Out of range, should be None
    }

    #[test]
    fn test_speed_edge_cases() {
        // Test minimum valid speed
        let mut provider_opts = HashMap::new();
        let mut openai_opts = HashMap::new();
        openai_opts.insert(
            "speed".to_string(),
            JsonValue::Number(serde_json::Number::from_f64(0.25).unwrap()),
        );
        provider_opts.insert("openai".to_string(), openai_opts.clone());

        let opts = OpenAISpeechProviderOptions::from(Some(provider_opts));
        assert_eq!(opts.speed, Some(0.25));

        // Test maximum valid speed
        let mut provider_opts = HashMap::new();
        openai_opts.insert(
            "speed".to_string(),
            JsonValue::Number(serde_json::Number::from_f64(4.0).unwrap()),
        );
        provider_opts.insert("openai".to_string(), openai_opts);

        let opts = OpenAISpeechProviderOptions::from(Some(provider_opts));
        assert_eq!(opts.speed, Some(4.0));
    }
}
