use super::prompt::Prompt;
use super::tools::{Tool, ToolChoice};
use crate::SharedProviderOptions;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Options for language model generation
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CallOptions {
    /// The prompt to send to the model
    pub prompt: Prompt,

    /// Maximum number of tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,

    /// Sampling temperature (0.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Sequences that will stop generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,

    /// Nucleus sampling parameter (0.0 to 1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Top-k sampling parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    /// Presence penalty to reduce repetition (-2.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    /// Frequency penalty to reduce repetition (-2.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    /// Format for the model's response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,

    /// Random seed for deterministic generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,

    /// Tools available for the model to call
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,

    /// Strategy for tool selection
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Whether to include raw streaming chunks in response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_raw_chunks: Option<bool>,

    /// Additional HTTP headers for the request
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,

    /// Provider-specific options
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<SharedProviderOptions>,
}

/// Format for the model's response
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ResponseFormat {
    /// Plain text response format
    Text,
    /// Structured JSON response format
    Json {
        /// JSON schema for the response structure
        #[serde(skip_serializing_if = "Option::is_none")]
        schema: Option<serde_json::Value>,
        /// Name of the JSON schema
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// Description of the JSON schema
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language_model::prompt::Message;

    #[test]
    fn test_call_options() {
        let opts = CallOptions {
            prompt: vec![Message::System {
                content: "test".into(),
            }],
            temperature: Some(0.7),
            max_output_tokens: Some(1000),
            ..Default::default()
        };
        let json = serde_json::to_string(&opts).unwrap();
        assert!(json.contains("temperature"));
        assert!(json.contains("maxOutputTokens"));
    }

    #[test]
    fn test_response_format_text() {
        let format = ResponseFormat::Text;
        let json = serde_json::to_value(&format).unwrap();
        assert_eq!(json["type"], "text");
    }

    #[test]
    fn test_response_format_json() {
        let format = ResponseFormat::Json {
            schema: None,
            name: Some("MySchema".into()),
            description: None,
        };
        let json = serde_json::to_value(&format).unwrap();
        assert_eq!(json["type"], "json");
        assert_eq!(json["name"], "MySchema");
    }
}
