//! AI SDK Provider Specification (v3)
//!
//! This crate defines the provider interface specification that
//! all AI model providers must implement.

pub mod json_value;
pub mod language_model;
pub mod shared;

// Re-export commonly used types
pub use json_value::{JsonArray, JsonObject, JsonValue};
pub use language_model::{
    CallOptions, Content, FinishReason, GenerateResponse, LanguageModel, StreamPart,
    StreamResponse, Usage,
};
pub use shared::{SharedHeaders, SharedProviderMetadata, SharedProviderOptions};
