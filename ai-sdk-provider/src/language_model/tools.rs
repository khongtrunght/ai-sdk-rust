use crate::SharedProviderOptions;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonSchema;

/// Tool choice strategy
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ToolChoice {
    /// Model chooses whether to call tools
    Auto,
    /// Model does not call tools
    None,
    /// Model must call at least one tool
    Required,
    /// Model must call a specific tool
    Tool {
        /// Name of the tool to call
        tool_name: String,
    },
}

/// Function tool definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionTool {
    /// Name of the tool
    pub name: String,

    /// Description of what the tool does
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// JSON Schema defining the tool's input parameters
    pub input_schema: JsonSchema,

    /// Provider-specific options for the tool
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<SharedProviderOptions>,
}

/// Provider-defined tool (e.g., openai.code-interpreter)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderDefinedTool {
    /// Must match pattern: `provider.tool-name`
    pub id: String,
    /// Human-readable name of the tool
    pub name: String,
    /// Arguments for the provider-defined tool
    pub args: serde_json::Map<String, serde_json::Value>,
}

/// Tool union type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Tool {
    /// A function tool defined by the application
    Function(FunctionTool),
    /// A tool defined by the provider
    ProviderDefined(ProviderDefinedTool),
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_tool_choice_auto() {
        let choice = ToolChoice::Auto;
        let json = serde_json::to_value(&choice).unwrap();
        assert_eq!(json["type"], "auto");
    }

    #[test]
    fn test_tool_choice_tool() {
        let choice = ToolChoice::Tool {
            tool_name: "search".into(),
        };
        let json = serde_json::to_value(&choice).unwrap();
        assert_eq!(json["type"], "tool");
        assert_eq!(json["tool_name"], "search");
    }

    #[test]
    fn test_function_tool() {
        let tool = FunctionTool {
            name: "get_weather".into(),
            description: Some("Get weather info".into()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }),
            provider_options: None,
        };
        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["name"], "get_weather");
        assert_eq!(json["inputSchema"]["type"], "object");
    }
}
