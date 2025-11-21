//! Generate structured objects from language models.
//!
//! This module provides the `generate_object` function that generates validated,
//! schema-based outputs from language models.

use super::output_strategy::{OutputStrategy, ValidationContext, ValidationResult};
use crate::retry::RetryPolicy;
use ai_sdk_provider::language_model::{
    CallOptions, CallWarning, Content, FinishReason, LanguageModel, Message, ResponseFormat,
    ResponseMetadata, TextPart, Usage,
};
use serde_json::Value;
use std::sync::Arc;
use thiserror::Error;

/// Errors that can occur during object generation.
#[derive(Debug, Error)]
pub enum GenerateObjectError {
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

    /// JSON parsing error
    #[error("JSON parsing error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// No text content in response
    #[error("No text content in model response")]
    NoTextContent,
}

/// Builder for generating structured objects.
pub struct GenerateObjectBuilder<S: OutputStrategy> {
    model: Option<Arc<dyn LanguageModel>>,
    prompt: Option<Vec<Message>>,
    output_strategy: Option<Arc<S>>,
    schema_name: Option<String>,
    schema_description: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    retry_policy: RetryPolicy,
}

impl<S: OutputStrategy + 'static> GenerateObjectBuilder<S> {
    /// Creates a new GenerateObjectBuilder
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

    /// Execute the object generation
    pub async fn execute(self) -> Result<GenerateObjectResult<S::Result>, GenerateObjectError> {
        let model = self.model.ok_or(GenerateObjectError::MissingModel)?;
        let messages = self.prompt.ok_or(GenerateObjectError::MissingPrompt)?;
        let strategy = self
            .output_strategy
            .ok_or(GenerateObjectError::MissingOutputStrategy)?;

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

        // Call model with retry
        let response = self
            .retry_policy
            .retry(|| {
                let model = model.clone();
                let options = options.clone();
                async move { model.do_generate(options).await }
            })
            .await
            .map_err(GenerateObjectError::ModelError)?;

        // Extract text from response
        let text = extract_text_from_content(&response.content)
            .ok_or(GenerateObjectError::NoTextContent)?;

        // Parse JSON
        let value: Value = serde_json::from_str(&text)?;

        // Validate final result
        let context = ValidationContext {
            text: text.clone(),
            response: response.response.clone().map(|r| ResponseMetadata {
                id: r.id,
                timestamp: r.timestamp,
                model_id: r.model_id,
            }),
            usage: response.usage.clone(),
        };

        let validation = strategy.validate_final_result(Some(value), context).await;

        match validation {
            ValidationResult::Success { value: result, .. } => Ok(GenerateObjectResult {
                object: result,
                usage: response.usage.clone(),
                finish_reason: response.finish_reason,
                warnings: response.warnings.clone(),
                raw_response: response,
            }),
            ValidationResult::Failure { error, .. } => {
                Err(GenerateObjectError::ValidationFailed(error.to_string()))
            }
        }
    }
}

impl<S: OutputStrategy + 'static> Default for GenerateObjectBuilder<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of generating a structured object.
#[derive(Debug)]
pub struct GenerateObjectResult<R> {
    /// The generated and validated object
    pub object: R,
    /// Token usage statistics
    pub usage: Usage,
    /// Finish reason
    pub finish_reason: FinishReason,
    /// Warnings from the provider
    pub warnings: Vec<CallWarning>,
    /// Raw response from the model
    pub raw_response: ai_sdk_provider::language_model::GenerateResponse,
}

/// Creates a new GenerateObjectBuilder
pub fn generate_object<S: OutputStrategy + 'static>() -> GenerateObjectBuilder<S> {
    GenerateObjectBuilder::new()
}

/// Extracts text content from a list of content items.
fn extract_text_from_content(content: &[Content]) -> Option<String> {
    let mut text = String::new();
    for item in content {
        if let Content::Text(TextPart { text: t, .. }) = item {
            text.push_str(t);
        }
    }
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

#[cfg(test)]
mod tests {
    // Note: These tests would require a mock LanguageModel implementation
    // For now, they serve as documentation of the expected API
}
