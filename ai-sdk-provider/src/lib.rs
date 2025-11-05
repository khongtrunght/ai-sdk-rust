//! AI SDK Provider Specification (v3)
//!
//! This crate defines the provider interface specification that
//! all AI model providers must implement.

pub mod embedding_model;
pub mod image_model;
pub mod json_value;
pub mod language_model;
pub mod shared;
pub mod speech_model;

// Re-export commonly used types
pub use embedding_model::{EmbedOptions, EmbedResponse, Embedding, EmbeddingModel, EmbeddingUsage};
pub use image_model::{
    CallWarning as ImageCallWarning, ImageData, ImageGenerateOptions, ImageGenerateResponse,
    ImageModel, ImageProviderMetadata,
};
pub use json_value::{JsonArray, JsonObject, JsonValue};
pub use language_model::{
    CallOptions, Content, FinishReason, GenerateResponse, LanguageModel, StreamPart,
    StreamResponse, Usage,
};
pub use shared::{SharedHeaders, SharedProviderMetadata, SharedProviderOptions};
pub use speech_model::{
    AudioData, CallWarning as SpeechCallWarning, SpeechGenerateOptions, SpeechGenerateResponse,
    SpeechModel,
};
