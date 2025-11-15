use ai_sdk_provider::language_model::ToolResultOutput;
use ai_sdk_provider::JsonValue;

/// Error mode for tool output conversion
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum ErrorMode {
    /// No error, normal output
    None,
    /// Convert output to error text
    Text,
    /// Convert output to error JSON
    Json,
}

/// Convert raw tool output to structured ToolResultOutput
///
/// # Arguments
/// * `output` - The raw JSON output from tool execution
/// * `error_mode` - How to handle the output (normal, error text, or error JSON)
/// * `custom_converter` - Optional custom conversion function
///
/// # Returns
/// A structured ToolResultOutput enum variant
#[allow(dead_code)]
pub fn create_tool_output(
    output: JsonValue,
    error_mode: ErrorMode,
    custom_converter: Option<&dyn Fn(JsonValue) -> ToolResultOutput>,
) -> ToolResultOutput {
    // Handle errors first
    match error_mode {
        ErrorMode::Text => {
            return ToolResultOutput::ErrorText {
                value: match output {
                    JsonValue::String(s) => s,
                    other => serde_json::to_string(&other)
                        .unwrap_or_else(|_| "Error serializing value".to_string()),
                },
                provider_metadata: None,
            };
        }
        ErrorMode::Json => {
            return ToolResultOutput::ErrorJson {
                value: output,
                provider_metadata: None,
            };
        }
        ErrorMode::None => {}
    }

    // Custom conversion via hook
    if let Some(converter) = custom_converter {
        return converter(output);
    }

    // Default conversion
    match output {
        JsonValue::String(s) => ToolResultOutput::Text {
            value: s,
            provider_metadata: None,
        },
        other => ToolResultOutput::Json {
            value: other,
            provider_metadata: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_tool_output_string_to_text() {
        let output = JsonValue::String("Hello, world!".to_string());
        let result = create_tool_output(output, ErrorMode::None, None);

        match result {
            ToolResultOutput::Text { value, .. } => {
                assert_eq!(value, "Hello, world!");
            }
            _ => panic!("Expected Text variant"),
        }
    }

    #[test]
    fn test_create_tool_output_object_to_json() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(
            "result".to_string(),
            JsonValue::String("success".to_string()),
        );
        map.insert(
            "count".to_string(),
            JsonValue::Number(serde_json::Number::from(42)),
        );
        let output = JsonValue::Object(map);

        let result = create_tool_output(output, ErrorMode::None, None);

        match result {
            ToolResultOutput::Json { value, .. } => {
                // Verify it's the same object
                if let JsonValue::Object(obj) = value {
                    assert!(obj.contains_key("result"));
                    assert!(obj.contains_key("count"));
                } else {
                    panic!("Expected Object");
                }
            }
            _ => panic!("Expected Json variant"),
        }
    }

    #[test]
    fn test_create_tool_output_error_text() {
        let output = JsonValue::String("An error occurred".to_string());
        let result = create_tool_output(output, ErrorMode::Text, None);

        match result {
            ToolResultOutput::ErrorText { value, .. } => {
                assert_eq!(value, "An error occurred");
            }
            _ => panic!("Expected ErrorText variant"),
        }
    }

    #[test]
    fn test_create_tool_output_error_json() {
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
        let output = JsonValue::Object(map);

        let result = create_tool_output(output, ErrorMode::Json, None);

        match result {
            ToolResultOutput::ErrorJson { value, .. } => {
                if let JsonValue::Object(obj) = value {
                    assert!(obj.contains_key("error"));
                    assert!(obj.contains_key("code"));
                } else {
                    panic!("Expected Object");
                }
            }
            _ => panic!("Expected ErrorJson variant"),
        }
    }

    #[test]
    fn test_create_tool_output_custom_converter() {
        let output = JsonValue::Null;

        let custom_converter = |_: JsonValue| ToolResultOutput::Text {
            value: "Custom conversion".to_string(),
            provider_metadata: None,
        };

        let result = create_tool_output(output, ErrorMode::None, Some(&custom_converter));

        match result {
            ToolResultOutput::Text { value, .. } => {
                assert_eq!(value, "Custom conversion");
            }
            _ => panic!("Expected Text variant from custom converter"),
        }
    }

    #[test]
    fn test_error_mode_takes_precedence_over_custom() {
        let output = JsonValue::String("test".to_string());

        let custom_converter = |_: JsonValue| ToolResultOutput::Text {
            value: "Should not be used".to_string(),
            provider_metadata: None,
        };

        let result = create_tool_output(output, ErrorMode::Text, Some(&custom_converter));

        match result {
            ToolResultOutput::ErrorText { value, .. } => {
                assert_eq!(value, "test");
            }
            _ => panic!("Expected ErrorText variant - error mode should take precedence"),
        }
    }
}
