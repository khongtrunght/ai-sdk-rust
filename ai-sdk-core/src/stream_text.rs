use ai_sdk_provider::language_model::{
    AssistantContentPart, CallOptions, Content, FinishReason, LanguageModel, Message, StreamPart,
    TextPart, Tool as ProviderTool, ToolCallPart, ToolChoice, ToolResultPart, Usage,
    UserContentPart,
};
use crate::error::StreamTextError;
use crate::tool::{Tool, ToolExecutor};
use async_stream::stream;
use std::pin::Pin;
use std::sync::Arc;
use tokio_stream::{Stream, StreamExt};

/// Builder for streaming text generation
pub struct StreamTextBuilder {
    model: Option<Arc<dyn LanguageModel>>,
    prompt: Option<Vec<Message>>,
    tools: Vec<Arc<dyn Tool>>,
    max_steps: u32,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
}

impl StreamTextBuilder {
    pub fn new() -> Self {
        Self {
            model: None,
            prompt: None,
            tools: Vec::new(),
            max_steps: 5,
            temperature: None,
            max_tokens: None,
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

    /// Set temperature
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set maximum tokens
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Execute and return streaming result
    pub async fn execute(self) -> Result<StreamTextResult, StreamTextError> {
        let model = self.model.ok_or(StreamTextError::MissingModel)?;
        let messages = self.prompt.ok_or(StreamTextError::MissingPrompt)?;

        let tool_executor = ToolExecutor::new(self.tools);

        // Create the stream
        let stream_impl = create_multi_step_stream(
            model,
            messages,
            tool_executor,
            self.max_steps,
            self.temperature,
            self.max_tokens,
        );

        Ok(StreamTextResult {
            stream: Box::pin(stream_impl),
        })
    }
}

impl Default for StreamTextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Result containing the text stream
pub struct StreamTextResult {
    stream: Pin<Box<dyn Stream<Item = Result<TextStreamPart, StreamTextError>> + Send>>,
}

impl StreamTextResult {
    /// Get a mutable reference to the stream
    pub fn stream_mut(
        &mut self,
    ) -> Pin<&mut (dyn Stream<Item = Result<TextStreamPart, StreamTextError>> + Send)> {
        self.stream.as_mut()
    }

    /// Consume self and return the underlying stream
    pub fn into_stream(
        self,
    ) -> Pin<Box<dyn Stream<Item = Result<TextStreamPart, StreamTextError>> + Send>> {
        self.stream
    }
}

/// Parts emitted by the text stream
#[derive(Debug, Clone)]
pub enum TextStreamPart {
    TextDelta(String),
    ToolCall(ToolCallPart),
    ToolResult(ToolResultPart),
    StepFinish {
        step_index: u32,
        finish_reason: FinishReason,
    },
    Finish {
        total_usage: Usage,
    },
}

/// Create a stream that handles multiple steps with tool calling
fn create_multi_step_stream(
    model: Arc<dyn LanguageModel>,
    initial_messages: Vec<Message>,
    tool_executor: ToolExecutor,
    max_steps: u32,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
) -> impl Stream<Item = Result<TextStreamPart, StreamTextError>> {
    stream! {
        let mut messages = initial_messages;
        let mut total_usage = Usage {
            input_tokens: Some(0),
            output_tokens: Some(0),
            total_tokens: Some(0),
            reasoning_tokens: None,
            cached_input_tokens: None,
        };

        for step_index in 0..max_steps {
            // Prepare options
            let mut options = CallOptions {
                prompt: messages.clone(),
                temperature,
                max_output_tokens: max_tokens,
                ..Default::default()
            };

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

            // Call model streaming
            let stream_response = match model.do_stream(options).await {
                Ok(resp) => resp,
                Err(e) => {
                    yield Err(StreamTextError::ModelError(format!("{:?}", e)));
                    return;
                }
            };

            let mut step_stream = stream_response.stream;
            let mut step_content: Vec<Content> = Vec::new();
            let mut text_accumulator = String::new();
            let mut tool_calls: Vec<ToolCallPart> = Vec::new();
            let mut finish_reason = None;

            // Process stream
            while let Some(part_result) = step_stream.next().await {
                let part = match part_result {
                    Ok(p) => p,
                    Err(e) => {
                        yield Err(StreamTextError::StreamError(format!("{:?}", e)));
                        return;
                    }
                };

                match part {
                    StreamPart::TextDelta { delta, .. } => {
                        yield Ok(TextStreamPart::TextDelta(delta.clone()));
                        text_accumulator.push_str(&delta);
                    }
                    StreamPart::ToolCall(tc) => {
                        tool_calls.push(tc.clone());
                        step_content.push(Content::ToolCall(tc.clone()));
                        yield Ok(TextStreamPart::ToolCall(tc));
                    }
                    StreamPart::Finish { finish_reason: fr, usage, .. } => {
                        finish_reason = Some(fr);
                        // Accumulate usage
                        if let (Some(total_input), Some(input)) =
                            (total_usage.input_tokens, usage.input_tokens)
                        {
                            total_usage.input_tokens = Some(total_input + input);
                        }
                        if let (Some(total_output), Some(output)) =
                            (total_usage.output_tokens, usage.output_tokens)
                        {
                            total_usage.output_tokens = Some(total_output + output);
                        }
                        if let (Some(total), Some(step_total)) =
                            (total_usage.total_tokens, usage.total_tokens)
                        {
                            total_usage.total_tokens = Some(total + step_total);
                        }
                    }
                    _ => {
                        // Ignore other stream parts for now
                    }
                }
            }

            // Add accumulated text to content
            if !text_accumulator.is_empty() {
                step_content.push(Content::Text(TextPart {
                    text: text_accumulator,
                    provider_metadata: None,
                }));
            }

            // Emit step finish
            let fr = finish_reason.unwrap_or(FinishReason::Stop);
            yield Ok(TextStreamPart::StepFinish {
                step_index,
                finish_reason: fr,
            });

            // Check if we should continue
            if tool_calls.is_empty() || fr != FinishReason::ToolCalls {
                break;
            }

            // Execute tools
            let tool_results = match tool_executor.execute_tools(tool_calls.clone()).await {
                Ok(results) => results,
                Err(e) => {
                    yield Err(StreamTextError::ToolError(e));
                    return;
                }
            };

            // Emit tool results
            for result in &tool_results {
                yield Ok(TextStreamPart::ToolResult(result.clone()));
            }

            // Append messages
            messages.push(Message::Assistant {
                content: step_content
                    .into_iter()
                    .filter_map(|c| match c {
                        Content::Text(tp) => Some(AssistantContentPart::Text(tp)),
                        Content::ToolCall(tc) => Some(AssistantContentPart::ToolCall(tc)),
                        Content::Reasoning(rp) => Some(AssistantContentPart::Reasoning(rp)),
                        Content::File(fp) => Some(AssistantContentPart::File(fp)),
                        Content::ToolResult(tr) => Some(AssistantContentPart::ToolResult(tr)),
                        Content::Source(_) => None,
                    })
                    .collect(),
            });
            messages.push(Message::Tool {
                content: tool_results,
            });
        }

        yield Ok(TextStreamPart::Finish { total_usage });
    }
}

/// Entry point function
pub fn stream_text() -> StreamTextBuilder {
    StreamTextBuilder::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_defaults() {
        let builder = StreamTextBuilder::new();
        assert_eq!(builder.max_steps, 5);
        assert!(builder.temperature.is_none());
        assert!(builder.max_tokens.is_none());
    }

    #[test]
    fn test_builder_configuration() {
        let builder = StreamTextBuilder::new()
            .max_steps(10)
            .temperature(0.7)
            .max_tokens(1000);

        assert_eq!(builder.max_steps, 10);
        assert_eq!(builder.temperature, Some(0.7));
        assert_eq!(builder.max_tokens, Some(1000));
    }
}
