use super::prompt::Prompt;
use super::tools::{Tool, ToolChoice};
use crate::SharedProviderOptions;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Options for language model generation
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CallOptions {
    pub prompt: Prompt,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_raw_chunks: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<SharedProviderOptions>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ResponseFormat {
    Text,
    Json {
        #[serde(skip_serializing_if = "Option::is_none")]
        schema: Option<serde_json::Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
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
