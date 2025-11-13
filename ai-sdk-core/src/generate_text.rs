use ai_sdk_provider::language_model::{
    AssistantContentPart, CallOptions, Content, FinishReason, LanguageModel, Message, TextPart,
    Tool as ProviderTool, ToolCallPart, ToolChoice, Usage, UserContentPart,
};
use crate::error::GenerateTextError;
use crate::retry::RetryPolicy;
use crate::tool::{Tool, ToolExecutor};
use std::sync::Arc;

/// Builder for text generation with optional tool calling
pub struct GenerateTextBuilder {
    model: Option<Arc<dyn LanguageModel>>,
    prompt: Option<Vec<Message>>,
    tools: Vec<Arc<dyn Tool>>,
    max_steps: u32,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    retry_policy: RetryPolicy,
}

impl GenerateTextBuilder {
    pub fn new() -> Self {
        Self {
            model: None,
            prompt: None,
            tools: Vec::new(),
            max_steps: 1,
            temperature: None,
            max_tokens: None,
            retry_policy: RetryPolicy::default(),
        }
    }

    /// Set the language model to use
    pub fn model<M: LanguageModel + 'static>(mut self, model: M) -> Self {
        self.model = Some(Arc::new(model));
        self
    }

    /// Set the prompt from a string
    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        let text = prompt.into();
        self.prompt = Some(vec![Message::User {
            content: vec![UserContentPart::Text { text }],
        }]);
        self
    }

    /// Set the prompt from messages
    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.prompt = Some(messages);
        self
    }

    /// Add tools that the model can call
    pub fn tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools = tools;
        self
    }

    /// Set maximum number of steps (tool call rounds)
    pub fn max_steps(mut self, max_steps: u32) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Set temperature (0.0 to 2.0)
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set maximum tokens to generate
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set retry policy
    pub fn retry_policy(mut self, retry_policy: RetryPolicy) -> Self {
        self.retry_policy = retry_policy;
        self
    }

    /// Execute the text generation
    pub async fn execute(self) -> Result<GenerateTextResult, GenerateTextError> {
        let model = self.model.ok_or(GenerateTextError::MissingModel)?;
        let mut messages = self.prompt.ok_or(GenerateTextError::MissingPrompt)?;

        let tool_executor = ToolExecutor::new(self.tools);
        let mut steps = Vec::new();
        let mut total_usage = Usage {
            input_tokens: Some(0),
            output_tokens: Some(0),
            total_tokens: Some(0),
            reasoning_tokens: None,
            cached_input_tokens: None,
        };

        for step_index in 0..self.max_steps {
            // Prepare call options
            let mut options = CallOptions {
                prompt: messages.clone(),
                temperature: self.temperature,
                max_output_tokens: self.max_tokens,
                ..Default::default()
            };

            // Add tools if available
            if !tool_executor.tools().is_empty() {
                let tool_defs = tool_executor.tool_definitions();
                options.tools = Some(
                    tool_defs
                        .into_iter()
                        .map(ProviderTool::Function)
                        .collect(),
                );
                options.tool_choice = Some(ToolChoice::Auto);
            }

            // Call model with retry
            let response = self
                .retry_policy
                .retry(|| {
                    let model = model.clone();
                    let options = options.clone();
                    async move { model.do_generate(options).await }
                })
                .await
                .map_err(GenerateTextError::ModelError)?;

            // Track usage
            if let (Some(total_input), Some(input)) =
                (total_usage.input_tokens, response.usage.input_tokens)
            {
                total_usage.input_tokens = Some(total_input + input);
            }
            if let (Some(total_output), Some(output)) =
                (total_usage.output_tokens, response.usage.output_tokens)
            {
                total_usage.output_tokens = Some(total_output + output);
            }
            if let (Some(total), Some(step_total)) =
                (total_usage.total_tokens, response.usage.total_tokens)
            {
                total_usage.total_tokens = Some(total + step_total);
            }

            // Extract tool calls
            let tool_calls = extract_tool_calls(&response.content);

            // Store step result
            let step_result = StepResult {
                step_index,
                response_content: response.content.clone(),
                tool_calls: tool_calls.clone(),
                finish_reason: response.finish_reason,
                usage: response.usage.clone(),
            };
            steps.push(step_result);

            // Check if we should continue
            if tool_calls.is_empty() || response.finish_reason != FinishReason::ToolCalls {
                break;
            }

            // Execute tools
            let tool_results = tool_executor.execute_tools(tool_calls).await?;

            // Append assistant message with tool calls
            messages.push(Message::Assistant {
                content: response
                    .content
                    .into_iter()
                    .map(|c| match c {
                        Content::Text(tp) => AssistantContentPart::Text(tp),
                        Content::ToolCall(tc) => AssistantContentPart::ToolCall(tc),
                        Content::Reasoning(rp) => AssistantContentPart::Reasoning(rp),
                        Content::File(fp) => AssistantContentPart::File(fp),
                        Content::ToolResult(tr) => AssistantContentPart::ToolResult(tr),
                        Content::Source(_) => {
                            // Source is not an AssistantContentPart, skip it
                            // This shouldn't happen in normal generation
                            AssistantContentPart::Text(TextPart {
                                text: String::new(),
                                provider_metadata: None,
                            })
                        }
                    })
                    .collect(),
            });

            // Append tool results
            messages.push(Message::Tool {
                content: tool_results,
            });
        }

        Ok(GenerateTextResult {
            steps,
            total_usage,
        })
    }
}

impl Default for GenerateTextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of text generation
pub struct GenerateTextResult {
    steps: Vec<StepResult>,
    total_usage: Usage,
}

impl GenerateTextResult {
    /// Get the final text response
    pub fn text(&self) -> String {
        self.steps
            .last()
            .map(|step| extract_text(&step.response_content))
            .unwrap_or_default()
    }

    /// Get all steps
    pub fn steps(&self) -> &[StepResult] {
        &self.steps
    }

    /// Get total token usage across all steps
    pub fn usage(&self) -> &Usage {
        &self.total_usage
    }

    /// Get finish reason from the last step
    pub fn finish_reason(&self) -> &FinishReason {
        self.steps
            .last()
            .map(|s| &s.finish_reason)
            .unwrap_or(&FinishReason::Stop)
    }
}

/// Result of a single generation step
#[derive(Debug, Clone)]
pub struct StepResult {
    pub step_index: u32,
    pub response_content: Vec<Content>,
    pub tool_calls: Vec<ToolCallPart>,
    pub finish_reason: FinishReason,
    pub usage: Usage,
}

/// Entry point function
pub fn generate_text() -> GenerateTextBuilder {
    GenerateTextBuilder::new()
}

// Helper functions
fn extract_tool_calls(content: &[Content]) -> Vec<ToolCallPart> {
    content
        .iter()
        .filter_map(|c| match c {
            Content::ToolCall(tc) => Some(tc.clone()),
            _ => None,
        })
        .collect()
}

fn extract_text(content: &[Content]) -> String {
    content
        .iter()
        .filter_map(|c| match c {
            Content::Text(text_part) => Some(text_part.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_tool_calls() {
        let content = vec![
            Content::Text(TextPart {
                text: "Hello".to_string(),
                provider_metadata: None,
            }),
            Content::ToolCall(ToolCallPart {
                tool_call_id: "call_123".to_string(),
                tool_name: "weather".to_string(),
                input: r#"{"location":"Tokyo"}"#.to_string(),
                provider_executed: None,
                dynamic: None,
                provider_metadata: None,
            }),
        ];

        let tool_calls = extract_tool_calls(&content);
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].tool_name, "weather");
    }

    #[test]
    fn test_extract_text() {
        let content = vec![
            Content::Text(TextPart {
                text: "Hello".to_string(),
                provider_metadata: None,
            }),
            Content::Text(TextPart {
                text: " world".to_string(),
                provider_metadata: None,
            }),
        ];

        let text = extract_text(&content);
        assert_eq!(text, "Hello world");
    }

    #[test]
    fn test_builder_defaults() {
        let builder = GenerateTextBuilder::new();
        assert_eq!(builder.max_steps, 1);
        assert!(builder.temperature.is_none());
        assert!(builder.max_tokens.is_none());
    }
}
