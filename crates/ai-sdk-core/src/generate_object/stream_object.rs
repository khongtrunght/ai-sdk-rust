//! Stream structured objects from language models.
//!
//! This module provides the `stream_object` function that generates validated,
//! schema-based outputs from language models with streaming support for partial results.

use super::output_strategy::{
    OutputStrategy, PartialValidation, ValidationContext, ValidationResult,
};
use crate::retry::RetryPolicy;
use crate::util::{is_deep_equal, parse_partial_json, ParseState};
use ai_sdk_provider::language_model::{
    CallOptions, FinishReason, LanguageModel, Message, ResponseFormat, StreamError, StreamPart,
    Usage,
};
use futures::stream::{Stream, StreamExt};
use serde_json::Value;
use std::pin::Pin;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::oneshot;

/// Errors that can occur during streaming object generation.
#[derive(Debug, Error)]
pub enum StreamObjectError {
    /// Missing model
    #[error("Model is required")]
    MissingModel,

    /// Missing prompt
    #[error("Prompt is required")]
    MissingPrompt,

    /// Missing output strategy
    #[error("Output strategy is required")]
    MissingOutputStrategy,

    /// Model error
    #[error("Model error: {0}")]
    ModelError(Box<dyn std::error::Error + Send + Sync>),

    /// Validation error
    #[error("Validation failed: {0}")]
    ValidationFailed(String),

    /// Stream error
    #[error("Stream error: {0}")]
    StreamError(String),
}

/// Builder for streaming structured objects.
pub struct StreamObjectBuilder<S: OutputStrategy> {
    model: Option<Arc<dyn LanguageModel>>,
    prompt: Option<Vec<Message>>,
    output_strategy: Option<Arc<S>>,
    schema_name: Option<String>,
    schema_description: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    retry_policy: RetryPolicy,
}

impl<S: OutputStrategy + 'static> StreamObjectBuilder<S> {
    /// Creates a new StreamObjectBuilder
    pub fn new() -> Self {
        Self {
            model: None,
            prompt: None,
            output_strategy: None,
            schema_name: None,
            schema_description: None,
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
            content: vec![ai_sdk_provider::language_model::UserContentPart::Text { text }],
        }]);
        self
    }

    /// Set the prompt from messages
    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.prompt = Some(messages);
        self
    }

    /// Set the output strategy
    pub fn output_strategy(mut self, strategy: S) -> Self {
        self.output_strategy = Some(Arc::new(strategy));
        self
    }

    /// Set the schema name
    pub fn schema_name(mut self, name: impl Into<String>) -> Self {
        self.schema_name = Some(name.into());
        self
    }

    /// Set the schema description
    pub fn schema_description(mut self, description: impl Into<String>) -> Self {
        self.schema_description = Some(description.into());
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

    /// Execute the streaming object generation
    pub async fn execute(
        self,
    ) -> Result<StreamObjectResult<S::Partial, S::Result>, StreamObjectError> {
        let model = self.model.ok_or(StreamObjectError::MissingModel)?;
        let messages = self.prompt.ok_or(StreamObjectError::MissingPrompt)?;
        let strategy = self
            .output_strategy
            .ok_or(StreamObjectError::MissingOutputStrategy)?;

        // Get schema from strategy
        let schema = strategy.json_schema().await;

        // Prepare call options
        let options = CallOptions {
            prompt: messages.clone(),
            response_format: Some(ResponseFormat::Json {
                schema: schema.clone(),
                name: self.schema_name,
                description: self.schema_description,
            }),
            temperature: self.temperature,
            max_output_tokens: self.max_tokens,
            tools: None, // No tools in object mode
            tool_choice: None,
            ..Default::default()
        };

        // Call model streaming with retry
        let stream_response = self
            .retry_policy
            .retry(|| {
                let model = model.clone();
                let options = options.clone();
                async move { model.do_stream(options).await }
            })
            .await
            .map_err(StreamObjectError::ModelError)?;

        // Create channels for final results
        let (object_tx, object_rx) = oneshot::channel();
        let (usage_tx, usage_rx) = oneshot::channel();

        // Transform stream
        let partial_stream =
            create_partial_stream(stream_response.stream, strategy, object_tx, usage_tx);

        Ok(StreamObjectResult {
            partial_object_stream: Box::pin(partial_stream),
            object: object_rx,
            usage: usage_rx,
        })
    }
}

impl<S: OutputStrategy + 'static> Default for StreamObjectBuilder<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// Creates a new StreamObjectBuilder
pub fn stream_object<S: OutputStrategy + 'static>() -> StreamObjectBuilder<S> {
    StreamObjectBuilder::new()
}

/// Part of a streaming object result.
#[derive(Debug)]
pub enum ObjectStreamPart<P> {
    /// Partial object update
    Object {
        /// The partial object
        object: P,
    },
    /// Text delta that was processed
    TextDelta {
        /// The text delta
        text_delta: String,
    },
    /// Stream finished
    Finish {
        /// Why the stream finished
        finish_reason: FinishReason,
    },
    /// Error during streaming
    Error {
        /// The error message
        error: String,
    },
}

/// Result of streaming object generation.
pub struct StreamObjectResult<P, R> {
    /// Stream of partial objects as they're generated
    pub partial_object_stream: Pin<Box<dyn Stream<Item = ObjectStreamPart<P>> + Send>>,
    /// Receiver for the final object (once complete)
    pub object: oneshot::Receiver<R>,
    /// Receiver for the final usage statistics
    pub usage: oneshot::Receiver<Usage>,
}

/// Creates a partial object stream from the model's stream.
fn create_partial_stream<S: OutputStrategy + 'static>(
    mut model_stream: Pin<Box<dyn Stream<Item = Result<StreamPart, StreamError>> + Send>>,
    strategy: Arc<S>,
    object_tx: oneshot::Sender<S::Result>,
    usage_tx: oneshot::Sender<Usage>,
) -> impl Stream<Item = ObjectStreamPart<S::Partial>> {
    async_stream::stream! {
        let mut accumulated_text = String::new();
        let mut text_delta = String::new();
        let mut latest_object_json: Option<Value> = None;
        let mut latest_object: Option<S::Partial> = None;
        let mut is_first_delta = true;

        while let Some(chunk_result) = model_stream.next().await {
            match chunk_result {
                Ok(StreamPart::TextDelta { delta, .. }) => {
                    accumulated_text.push_str(&delta);
                    text_delta.push_str(&delta);

                    // Parse partial JSON
                    let parse_result = parse_partial_json(Some(&accumulated_text));

                    if let Some(current_json) = parse_result.value {
                        // Check if JSON changed
                        let json_changed = match &latest_object_json {
                            Some(prev) => !is_deep_equal(prev, &current_json),
                            None => true,
                        };

                        if json_changed {
                            // Validate partial
                            let is_final_delta = parse_result.state == ParseState::SuccessfulParse;
                            let validation = strategy
                                .validate_partial_result(
                                    current_json.clone(),
                                    text_delta.clone(),
                                    is_first_delta,
                                    is_final_delta,
                                    latest_object.as_ref(),
                                )
                                .await;

                            match validation {
                                ValidationResult::Success {
                                    value: PartialValidation {
                                        partial,
                                        text_delta: processed_delta,
                                    },
                                    ..
                                } => {
                                    // Check if object changed (deep equality for the partial object)
                                    let object_changed = match &latest_object {
                                        Some(_prev) => {
                                            // For now, assume changed if JSON changed
                                            // TODO: Implement deep equality for generic types
                                            true
                                        }
                                        None => true,
                                    };

                                    if object_changed {
                                        latest_object_json = Some(current_json);
                                        latest_object = Some(partial.clone());
                                        is_first_delta = false;

                                        // Emit object and text delta
                                        yield ObjectStreamPart::Object { object: partial };
                                        if !processed_delta.is_empty() {
                                            yield ObjectStreamPart::TextDelta {
                                                text_delta: processed_delta,
                                            };
                                        }

                                        text_delta.clear();
                                    }
                                }
                                ValidationResult::Failure { .. } => {
                                    // Don't emit error for partial validation failures
                                    // Just skip this update and wait for more data
                                }
                            }
                        }
                    }
                }
                Ok(StreamPart::Finish {
                    usage,
                    finish_reason,
                    ..
                }) => {
                    // Validate final result
                    let final_json = parse_partial_json(Some(&accumulated_text)).value;
                    let context = ValidationContext {
                        text: accumulated_text.clone(),
                        response: None,
                        usage: usage.clone(),
                    };

                    let validation = strategy.validate_final_result(final_json, context).await;

                    match validation {
                        ValidationResult::Success { value: result, .. } => {
                            let _ = object_tx.send(result);
                        }
                        ValidationResult::Failure { error, .. } => {
                            yield ObjectStreamPart::Error {
                                error: error.to_string(),
                            };
                        }
                    }

                    let _ = usage_tx.send(usage);
                    yield ObjectStreamPart::Finish { finish_reason };
                    break;
                }
                Err(e) => {
                    yield ObjectStreamPart::Error {
                        error: e.to_string(),
                    };
                    break;
                }
                _ => {
                    // Ignore other stream parts (metadata, etc.)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    // Note: These tests would require a mock LanguageModel implementation
    // For now, they serve as documentation of the expected API
}
