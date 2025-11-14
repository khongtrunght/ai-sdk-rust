//! # AI SDK Core
//!
//! High-level APIs for working with AI models.
//!
//! This crate provides ergonomic, high-level functions for:
//! - Text generation with `generate_text()` and `stream_text()`
//! - Tool/function calling with automatic execution loops
//! - Embeddings with `embed()` and `embed_many()`
//!
//! ## Example: Text Generation
//!
//! ```rust,ignore
//! use ai_sdk_core::generate_text;
//! use ai_sdk_openai::openai;
//!
//! let result = generate_text()
//!     .model(openai("gpt-4").api_key(api_key))
//!     .prompt("Explain quantum computing")
//!     .execute()
//!     .await?;
//!
//! println!("{}", result.text());
//! ```
//!
//! ## Example: Tool Calling
//!
//! ```rust,ignore
//! use ai_sdk_core::{generate_text, Tool, ToolContext};
//! use ai_sdk_openai::openai;
//! use async_trait::async_trait;
//! use std::sync::Arc;
//!
//! struct WeatherTool;
//!
//! #[async_trait]
//! impl Tool for WeatherTool {
//!     fn name(&self) -> &str { "get_weather" }
//!     fn description(&self) -> &str { "Get weather for a location" }
//!     fn input_schema(&self) -> serde_json::Value {
//!         serde_json::json!({
//!             "type": "object",
//!             "properties": {
//!                 "location": {"type": "string"}
//!             }
//!         })
//!     }
//!     async fn execute(&self, input: serde_json::Value, _ctx: &ToolContext)
//!         -> Result<serde_json::Value, ai_sdk_core::ToolError> {
//!         Ok(serde_json::json!({"temperature": 72}))
//!     }
//! }
//!
//! let result = generate_text()
//!     .model(openai("gpt-4").api_key(api_key))
//!     .prompt("What's the weather in Tokyo?")
//!     .tools(vec![Arc::new(WeatherTool)])
//!     .max_steps(5)
//!     .execute()
//!     .await?;
//! ```
//!
//! ## Example: Streaming
//!
//! ```rust,ignore
//! use ai_sdk_core::stream_text;
//! use tokio_stream::StreamExt;
//!
//! let mut result = stream_text()
//!     .model(openai("gpt-4").api_key(api_key))
//!     .prompt("Write a story")
//!     .execute()
//!     .await?;
//!
//! let mut stream = result.into_stream();
//! while let Some(part) = stream.next().await {
//!     match part? {
//!         TextStreamPart::TextDelta(delta) => print!("{}", delta),
//!         _ => {}
//!     }
//! }
//! ```

#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]

mod embed;
mod embed_many;
mod error;
mod generate_text;
mod retry;
mod stop_condition;
mod stream_text;
mod tool;

/// Utility functions for media type detection, file download, and base64 encoding
pub mod util;

/// Generate structured objects with schema validation
pub mod generate_object;

/// Agent framework for autonomous tool-using agents
pub mod agent;

// Re-export commonly used types from ai-sdk-provider
pub use ai_sdk_provider::language_model::{
    CallOptions, Content, FinishReason, LanguageModel, Message, ToolCallPart, ToolResultPart, Usage,
};
pub use ai_sdk_provider::{EmbeddingModel, EmbeddingUsage, JsonValue};

// Re-export core functionality
pub use embed::{embed, EmbedBuilder, EmbedResult};
pub use embed_many::{embed_many, EmbedManyBuilder, EmbedManyResult};
pub use error::{EmbedError, GenerateTextError, StreamTextError, ToolError};
pub use generate_text::{generate_text, GenerateTextBuilder, GenerateTextResult, StepResult};
pub use retry::RetryPolicy;
pub use stop_condition::{stop_after_steps, stop_on_finish, StopCondition};
pub use stream_text::{stream_text, StreamTextBuilder, StreamTextResult, TextStreamPart};
pub use tool::{Tool, ToolContext, ToolExecutor};
