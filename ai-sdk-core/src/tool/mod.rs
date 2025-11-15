mod tool_output;

use crate::error::ToolError;
use ai_sdk_provider::language_model::{
    FunctionTool, Message, ToolCallPart, ToolResultOutput, ToolResultPart,
};
use ai_sdk_provider::JsonValue;
use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;

/// Context provided to tools during execution
pub struct ToolContext {
    /// ID of the tool call being executed
    pub tool_call_id: String,
    /// Conversation messages leading up to this tool call
    pub messages: Vec<Message>,
}

/// Trait that tools must implement
#[async_trait]
pub trait Tool: Send + Sync {
    /// Name of the tool (used by LLM)
    fn name(&self) -> &str;

    /// Description of what the tool does
    fn description(&self) -> &str;

    /// JSON Schema for the tool's input
    fn input_schema(&self) -> Value;

    /// Execute the tool with given input
    async fn execute(&self, input: Value, context: &ToolContext) -> Result<JsonValue, ToolError>;

    /// Check if this tool requires approval before execution
    fn needs_approval(&self, _input: &Value) -> bool {
        false
    }

    /// Custom output formatting (optional)
    /// If not implemented, uses default conversion (string → text, object → json)
    fn to_model_output(&self, output: JsonValue) -> ToolResultOutput {
        // Default implementation
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
}

/// Executor that manages tool execution
pub struct ToolExecutor {
    tools: Vec<Arc<dyn Tool>>,
}

impl ToolExecutor {
    /// Creates a new ToolExecutor with the given tools
    pub fn new(tools: Vec<Arc<dyn Tool>>) -> Self {
        Self { tools }
    }

    /// Get tool definitions for the model
    pub fn tool_definitions(&self) -> Vec<FunctionTool> {
        self.tools
            .iter()
            .map(|tool| FunctionTool {
                name: tool.name().to_string(),
                description: Some(tool.description().to_string()),
                input_schema: tool.input_schema(),
                provider_options: None,
            })
            .collect()
    }

    /// Execute multiple tool calls in parallel
    pub async fn execute_tools(&self, tool_calls: Vec<ToolCallPart>) -> Vec<ToolResultPart> {
        let mut futures = Vec::new();

        for tool_call in tool_calls {
            let tool_opt = self.find_tool(&tool_call.tool_name);
            let tool_call_id = tool_call.tool_call_id.clone();
            let tool_name = tool_call.tool_name.clone();
            let input_str = tool_call.input.clone();

            let future = async move {
                // Handle tool not found
                let tool = match tool_opt {
                    Some(t) => t,
                    None => {
                        return ToolResultPart {
                            tool_call_id,
                            tool_name: tool_name.clone(),
                            output: ToolResultOutput::ErrorText {
                                value: format!("Tool '{}' not found", tool_name),
                                provider_metadata: None,
                            },
                            preliminary: None,
                            provider_metadata: None,
                        };
                    }
                };

                let context = ToolContext {
                    tool_call_id: tool_call_id.clone(),
                    messages: vec![], // TODO: pass actual messages
                };

                // Parse input
                let input: Value = match serde_json::from_str(&input_str) {
                    Ok(v) => v,
                    Err(e) => {
                        // Return error result instead of propagating
                        return ToolResultPart {
                            tool_call_id,
                            tool_name,
                            output: ToolResultOutput::ErrorText {
                                value: format!("Invalid input: {}", e),
                                provider_metadata: None,
                            },
                            preliminary: None,
                            provider_metadata: None,
                        };
                    }
                };

                // Check approval
                if tool.needs_approval(&input) {
                    // Return denial result
                    return ToolResultPart {
                        tool_call_id,
                        tool_name,
                        output: ToolResultOutput::ExecutionDenied {
                            reason: Some("Execution denied by user".to_string()),
                            provider_metadata: None,
                        },
                        preliminary: None,
                        provider_metadata: None,
                    };
                }

                // Execute tool and convert to structured output
                let output = match tool.execute(input, &context).await {
                    Ok(raw_output) => {
                        // Success - convert to structured output
                        tool.to_model_output(raw_output)
                    }
                    Err(error) => {
                        // Execution error - return error output
                        ToolResultOutput::ErrorText {
                            value: error.to_string(),
                            provider_metadata: None,
                        }
                    }
                };

                ToolResultPart {
                    tool_call_id,
                    tool_name,
                    output,
                    preliminary: None,
                    provider_metadata: None,
                }
            };

            futures.push(future);
        }

        // Execute all tools in parallel
        futures::future::join_all(futures).await
    }

    /// Find a tool by name
    fn find_tool(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.iter().find(|t| t.name() == name).cloned()
    }

    /// Returns the list of available tools
    pub fn tools(&self) -> &[Arc<dyn Tool>] {
        &self.tools
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestTool {
        name: String,
        result: String,
    }

    #[async_trait]
    impl Tool for TestTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "A test tool"
        }

        fn input_schema(&self) -> Value {
            serde_json::json!({
                "type": "object",
                "properties": {}
            })
        }

        async fn execute(
            &self,
            _input: Value,
            _context: &ToolContext,
        ) -> Result<JsonValue, ToolError> {
            Ok(JsonValue::String(self.result.clone()))
        }
    }

    #[tokio::test]
    async fn test_tool_executor_find_tool() {
        let tool = Arc::new(TestTool {
            name: "test".to_string(),
            result: "success".to_string(),
        });

        let executor = ToolExecutor::new(vec![tool]);
        assert!(executor.find_tool("test").is_some());
        assert!(executor.find_tool("nonexistent").is_none());
    }

    #[tokio::test]
    async fn test_tool_executor_execute() {
        let tool = Arc::new(TestTool {
            name: "test".to_string(),
            result: "success".to_string(),
        });

        let executor = ToolExecutor::new(vec![tool]);

        let tool_call = ToolCallPart {
            tool_call_id: "call_123".to_string(),
            tool_name: "test".to_string(),
            input: "{}".to_string(),
            provider_executed: None,
            dynamic: None,
            provider_metadata: None,
        };

        let results = executor.execute_tools(vec![tool_call]).await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].tool_call_id, "call_123");
        assert_eq!(results[0].tool_name, "test");

        // Check that output is Text variant
        match &results[0].output {
            ToolResultOutput::Text { value, .. } => {
                assert_eq!(value, "success");
            }
            _ => panic!("Expected Text output variant"),
        }
    }

    #[tokio::test]
    async fn test_tool_executor_tool_not_found() {
        let executor = ToolExecutor::new(vec![]);

        let tool_call = ToolCallPart {
            tool_call_id: "call_123".to_string(),
            tool_name: "nonexistent".to_string(),
            input: "{}".to_string(),
            provider_executed: None,
            dynamic: None,
            provider_metadata: None,
        };

        let results = executor.execute_tools(vec![tool_call]).await;
        assert_eq!(results.len(), 1);

        // Check that output is ErrorText variant
        match &results[0].output {
            ToolResultOutput::ErrorText { value, .. } => {
                assert!(value.contains("not found"));
            }
            _ => panic!("Expected ErrorText output variant"),
        }
    }
}
