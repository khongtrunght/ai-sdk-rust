use crate::{JsonValue, SharedProviderMetadata};
use serde::{Deserialize, Serialize};

/// Structured output from a tool execution
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ToolResultOutput {
    /// Plain text output
    Text {
        /// The text content
        value: String,
        /// Provider-specific metadata
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedProviderMetadata>,
    },

    /// Structured JSON output
    Json {
        /// The JSON value
        value: JsonValue,
        /// Provider-specific metadata
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedProviderMetadata>,
    },

    /// Execution denied by user
    ExecutionDenied {
        /// Optional reason for denial
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
        /// Provider-specific metadata
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedProviderMetadata>,
    },

    /// Error as text
    ErrorText {
        /// The error message
        value: String,
        /// Provider-specific metadata
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedProviderMetadata>,
    },

    /// Error as JSON
    ErrorJson {
        /// The error data as JSON
        value: JsonValue,
        /// Provider-specific metadata
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedProviderMetadata>,
    },

    /// Multi-part content (text, images, files)
    Content {
        /// The content parts
        value: Vec<ContentPart>,
        /// Provider-specific metadata
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedProviderMetadata>,
    },
}

/// Content part for multi-part tool results
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ContentPart {
    /// Plain text content
    Text {
        /// The text content
        text: String,
        /// Provider-specific metadata
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedProviderMetadata>,
    },
    /// Image from URL
    ImageUrl {
        /// The image URL
        url: String,
        /// Provider-specific metadata
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedProviderMetadata>,
    },
    /// Image from base64 data
    ImageData {
        /// Base64-encoded image data
        data: String,
        /// MIME type of the image (e.g., "image/png")
        #[serde(rename = "mediaType")]
        media_type: String,
        /// Provider-specific metadata
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedProviderMetadata>,
    },
    /// File from base64 data
    FileData {
        /// Base64-encoded file data
        data: String,
        /// MIME type of the file
        #[serde(rename = "mediaType")]
        media_type: String,
        /// Optional filename
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
        /// Provider-specific metadata
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedProviderMetadata>,
    },
    /// File from URL
    FileUrl {
        /// The file URL
        url: String,
        /// Provider-specific metadata
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedProviderMetadata>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_result_output_text_serialization() {
        let output = ToolResultOutput::Text {
            value: "Hello, world!".to_string(),
            provider_metadata: None,
        };
        let json = serde_json::to_value(&output).unwrap();
        assert_eq!(json["type"], "text");
        assert_eq!(json["value"], "Hello, world!");
    }

    #[test]
    fn test_tool_result_output_json_serialization() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(
            "result".to_string(),
            JsonValue::String("success".to_string()),
        );

        let output = ToolResultOutput::Json {
            value: JsonValue::Object(map),
            provider_metadata: None,
        };
        let json = serde_json::to_value(&output).unwrap();
        assert_eq!(json["type"], "json");
        assert_eq!(json["value"]["result"], "success");
    }

    #[test]
    fn test_tool_result_output_error_text() {
        let output = ToolResultOutput::ErrorText {
            value: "An error occurred".to_string(),
            provider_metadata: None,
        };
        let json = serde_json::to_value(&output).unwrap();
        assert_eq!(json["type"], "error-text");
        assert_eq!(json["value"], "An error occurred");
    }

    #[test]
    fn test_tool_result_output_error_json() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(
            "error".to_string(),
            JsonValue::String("Not found".to_string()),
        );
        map.insert(
            "code".to_string(),
            JsonValue::Number(serde_json::Number::from(404)),
        );

        let output = ToolResultOutput::ErrorJson {
            value: JsonValue::Object(map),
            provider_metadata: None,
        };
        let json = serde_json::to_value(&output).unwrap();
        assert_eq!(json["type"], "error-json");
        assert_eq!(json["value"]["error"], "Not found");
        assert_eq!(json["value"]["code"], 404);
    }

    #[test]
    fn test_tool_result_output_execution_denied() {
        let output = ToolResultOutput::ExecutionDenied {
            reason: Some("User denied execution".to_string()),
            provider_metadata: None,
        };
        let json = serde_json::to_value(&output).unwrap();
        assert_eq!(json["type"], "execution-denied");
        assert_eq!(json["reason"], "User denied execution");
    }

    #[test]
    fn test_content_part_text() {
        let part = ContentPart::Text {
            text: "Sample text".to_string(),
            provider_metadata: None,
        };
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(json["type"], "text");
        assert_eq!(json["text"], "Sample text");
    }

    #[test]
    fn test_content_part_image_url() {
        let part = ContentPart::ImageUrl {
            url: "https://example.com/image.png".to_string(),
            provider_metadata: None,
        };
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(json["type"], "image-url");
        assert_eq!(json["url"], "https://example.com/image.png");
    }

    #[test]
    fn test_content_part_image_data() {
        let part = ContentPart::ImageData {
            data: "base64data".to_string(),
            media_type: "image/png".to_string(),
            provider_metadata: None,
        };
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(json["type"], "image-data");
        assert_eq!(json["data"], "base64data");
        assert_eq!(json["mediaType"], "image/png");
    }

    #[test]
    fn test_tool_result_output_multi_part_content() {
        let output = ToolResultOutput::Content {
            value: vec![
                ContentPart::Text {
                    text: "Here is the image:".to_string(),
                    provider_metadata: None,
                },
                ContentPart::ImageUrl {
                    url: "https://example.com/image.png".to_string(),
                    provider_metadata: None,
                },
            ],
            provider_metadata: None,
        };
        let json = serde_json::to_value(&output).unwrap();
        assert_eq!(json["type"], "content");
        assert_eq!(json["value"].as_array().unwrap().len(), 2);
        assert_eq!(json["value"][0]["type"], "text");
        assert_eq!(json["value"][1]["type"], "image-url");
    }
}
