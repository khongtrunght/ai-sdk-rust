use super::agent::{Agent, AgentCallParameters};
use super::step_result::StepResult;
use super::stop_condition::is_stop_condition_met;
use super::tool_loop_agent_settings::{FinishContext, PrepareCallContext, ToolLoopAgentSettings};
use crate::error::GenerateTextError;
use crate::tool::{Tool, ToolExecutor};
use crate::{GenerateTextResult, StreamTextResult};
use ai_sdk_provider::language_model::{
    AssistantContentPart, CallOptions, Content, FinishReason, FunctionTool, Message,
    Tool as ProviderTool, Usage, UserContentPart,
};
use async_trait::async_trait;
use std::sync::Arc;

/// Tool loop agent that autonomously executes tools
pub struct ToolLoopAgent {
    settings: ToolLoopAgentSettings,
}

impl ToolLoopAgent {
    /// Create a new ToolLoopAgent with the given settings
    pub fn new(settings: ToolLoopAgentSettings) -> Self {
        Self { settings }
    }

    /// Prepare the call before execution
    fn prepare_call(&self, params: AgentCallParameters) -> Result<PreparedCall, GenerateTextError> {
        // Convert prompt/messages
        let mut messages = match (params.prompt, params.messages) {
            (Some(prompt), None) => vec![Message::User {
                content: vec![UserContentPart::Text { text: prompt }],
            }],
            (None, Some(messages)) => messages,
            (Some(_), Some(_)) => {
                return Err(GenerateTextError::InvalidParameters(
                    "Cannot specify both prompt and messages".into(),
                ))
            }
            (None, None) => {
                return Err(GenerateTextError::InvalidParameters(
                    "Must specify either prompt or messages".into(),
                ))
            }
        };

        // Prepend instructions as system message
        if let Some(instructions) = &self.settings.instructions {
            messages.insert(
                0,
                Message::System {
                    content: instructions.clone(),
                },
            );
        }

        Ok(PreparedCall { messages })
    }

    /// Execute the tool loop
    async fn execute_loop(
        &self,
        mut messages: Vec<Message>,
    ) -> Result<GenerateTextResult, GenerateTextError> {
        // Create tool executor with executable tools
        let tool_executor = ToolExecutor::new(self.settings.tools.clone());

        // Convert executable tools to provider tool definitions
        let provider_tools: Vec<ProviderTool> = self
            .settings
            .tools
            .iter()
            .map(|tool| {
                ProviderTool::Function(FunctionTool {
                    name: tool.name().to_string(),
                    description: Some(tool.description().to_string()),
                    input_schema: tool.input_schema(),
                    provider_options: None,
                })
            })
            .collect();

        let mut steps = Vec::new();
        let mut total_usage = Usage {
            input_tokens: Some(0),
            output_tokens: Some(0),
            total_tokens: Some(0),
            reasoning_tokens: None,
            cached_input_tokens: None,
        };

        // Use a default stop condition of 20 steps if none provided
        let default_max_steps = 20;
        let mut step_count = 0;

        loop {
            // Check stop conditions before each step
            if !steps.is_empty()
                && is_stop_condition_met(&self.settings.stop_conditions, &steps).await
            {
                break;
            }

            // Default: stop after 20 steps if no stop conditions provided
            if self.settings.stop_conditions.is_empty() && step_count >= default_max_steps {
                break;
            }

            step_count += 1;

            // Prepare call options
            let options = CallOptions {
                prompt: messages.clone(),
                tools: if !provider_tools.is_empty() {
                    Some(provider_tools.clone())
                } else {
                    None
                },
                tool_choice: if !provider_tools.is_empty() {
                    Some(self.settings.tool_choice.clone())
                } else {
                    None
                },
                ..Default::default()
            };

            // Call model
            let response = self
                .settings
                .model
                .do_generate(options)
                .await
                .map_err(GenerateTextError::ModelError)?;

            // Update usage
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

            // Create step result
            let step_result = StepResult {
                content: response.content.clone(),
                tool_calls: StepResult::extract_tool_calls(&response.content),
                tool_results: None,
                text: StepResult::extract_text(&response.content),
                reasoning_text: StepResult::extract_reasoning(&response.content),
                finish_reason: response.finish_reason,
                usage: response.usage.clone(),
                warnings: response.warnings.clone(),
                request: response.request,
                response: response.response,
                provider_metadata: response.provider_metadata,
            };

            steps.push(step_result.clone());

            // Call on_step_finish callback
            if let Some(callback) = &self.settings.on_step_finish {
                callback(step_result.clone()).await;
            }

            // Check if we should continue (no tool calls or not finishing with tool calls)
            if step_result.tool_calls.is_none() || response.finish_reason != FinishReason::ToolCalls
            {
                break;
            }

            // Execute tools
            let tool_calls = step_result.tool_calls.as_ref().unwrap();
            let tool_results = tool_executor.execute_tools(tool_calls.clone()).await?;

            // Append assistant message with tool calls
            messages.push(Message::Assistant {
                content: response
                    .content
                    .into_iter()
                    .filter_map(|c| match c {
                        Content::Text(tp) => Some(AssistantContentPart::Text(tp)),
                        Content::ToolCall(tc) => Some(AssistantContentPart::ToolCall(tc)),
                        Content::Reasoning(rp) => Some(AssistantContentPart::Reasoning(rp)),
                        _ => None,
                    })
                    .collect(),
            });

            // Append tool results
            messages.push(Message::Tool {
                content: tool_results,
            });
        }

        // Call on_finish callback
        if let Some(callback) = &self.settings.on_finish {
            let last_step = steps.last().cloned().unwrap_or(StepResult {
                content: vec![],
                tool_calls: None,
                tool_results: None,
                text: String::new(),
                reasoning_text: None,
                finish_reason: FinishReason::Stop,
                usage: Usage {
                    input_tokens: Some(0),
                    output_tokens: Some(0),
                    total_tokens: Some(0),
                    reasoning_tokens: None,
                    cached_input_tokens: None,
                },
                warnings: vec![],
                request: None,
                response: None,
                provider_metadata: None,
            });

            callback(FinishContext {
                steps: steps.clone(),
                total_usage: total_usage.clone(),
                last_step,
            })
            .await;
        }

        // Convert agent steps to generate_text steps
        let generate_text_steps = convert_agent_steps_to_generate_text_steps(steps);
        Ok(GenerateTextResult::new(generate_text_steps, total_usage))
    }
}

#[async_trait]
impl Agent for ToolLoopAgent {
    fn id(&self) -> Option<&str> {
        self.settings.id.as_deref()
    }

    fn tools(&self) -> &[Arc<dyn Tool>] {
        &self.settings.tools
    }

    async fn generate(
        &self,
        params: AgentCallParameters,
    ) -> Result<GenerateTextResult, GenerateTextError> {
        // Prepare call
        let mut prepared = self.prepare_call(params)?;

        // Call prepare_call hook if provided
        if let Some(prepare_call) = &self.settings.prepare_call {
            prepared.messages = prepare_call(PrepareCallContext {
                messages: prepared.messages,
                agent: self,
            })
            .await?;
        }

        // Execute the loop
        self.execute_loop(prepared.messages).await
    }

    async fn stream(
        &self,
        _params: AgentCallParameters,
    ) -> Result<StreamTextResult, GenerateTextError> {
        // Streaming not yet implemented for agents
        Err(GenerateTextError::InvalidParameters(
            "Streaming not yet supported for agents".into(),
        ))
    }
}

struct PreparedCall {
    messages: Vec<Message>,
}

// Helper to convert agent StepResults to generate_text StepResults
fn convert_agent_steps_to_generate_text_steps(
    steps: Vec<StepResult>,
) -> Vec<crate::generate_text::StepResult> {
    steps
        .into_iter()
        .enumerate()
        .map(|(index, step)| crate::generate_text::StepResult {
            step_index: index as u32,
            response_content: step.content,
            tool_calls: step.tool_calls.unwrap_or_default(),
            finish_reason: step.finish_reason,
            usage: step.usage,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_sdk_provider::language_model::TextPart;
    use std::sync::Arc;

    // Mock model for testing
    struct MockModel;

    #[async_trait]
    impl ai_sdk_provider::language_model::LanguageModel for MockModel {
        fn specification_version(&self) -> &str {
            "v3"
        }

        fn provider(&self) -> &str {
            "mock"
        }

        fn model_id(&self) -> &str {
            "mock-1"
        }

        async fn supported_urls(&self) -> std::collections::HashMap<String, Vec<String>> {
            std::collections::HashMap::new()
        }

        async fn do_generate(
            &self,
            _options: CallOptions,
        ) -> Result<
            ai_sdk_provider::language_model::GenerateResponse,
            Box<dyn std::error::Error + Send + Sync>,
        > {
            Ok(ai_sdk_provider::language_model::GenerateResponse {
                content: vec![Content::Text(TextPart {
                    text: "Hello".to_string(),
                    provider_metadata: None,
                })],
                finish_reason: FinishReason::Stop,
                usage: Usage {
                    input_tokens: Some(10),
                    output_tokens: Some(20),
                    total_tokens: Some(30),
                    reasoning_tokens: None,
                    cached_input_tokens: None,
                },
                provider_metadata: None,
                request: None,
                response: None,
                warnings: vec![],
            })
        }

        async fn do_stream(
            &self,
            _options: CallOptions,
        ) -> Result<
            ai_sdk_provider::language_model::StreamResponse,
            Box<dyn std::error::Error + Send + Sync + 'static>,
        > {
            unimplemented!()
        }
    }

    #[tokio::test]
    async fn test_prepare_call_with_prompt() {
        let settings = ToolLoopAgentSettings::builder(Arc::new(MockModel)).build();
        let agent = ToolLoopAgent::new(settings);

        let params = AgentCallParameters::from_prompt("Hello");
        let prepared = agent.prepare_call(params).unwrap();

        assert_eq!(prepared.messages.len(), 1);
    }

    #[tokio::test]
    async fn test_prepare_call_with_instructions() {
        let settings = ToolLoopAgentSettings::builder(Arc::new(MockModel))
            .instructions("You are a helpful assistant")
            .build();
        let agent = ToolLoopAgent::new(settings);

        let params = AgentCallParameters::from_prompt("Hello");
        let prepared = agent.prepare_call(params).unwrap();

        // Should have system message + user message
        assert_eq!(prepared.messages.len(), 2);
        assert!(matches!(prepared.messages[0], Message::System { .. }));
    }
}
