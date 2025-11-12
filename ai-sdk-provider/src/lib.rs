//! # AI SDK Provider Specification (v3)
//!
//! This crate defines the provider interface specification that all AI model
//! providers must implement. It provides trait-based interfaces for different
//! types of AI models.
//!
//! ## Features
//!
//! - **Language Models** - Text generation and chat completion
//! - **Embedding Models** - Text embedding generation
//! - **Image Models** - Image generation
//! - **Speech Models** - Text-to-speech synthesis
//! - **Transcription Models** - Speech-to-text transcription
//! - **Reranking Models** - Document reranking by relevance
//!
//! ## Example
//!
//! ```rust,ignore
//! use ai_sdk_provider::{LanguageModel, CallOptions, Message, UserContentPart};
//!
//! async fn generate_text<M: LanguageModel>(model: &M) {
//!     let options = CallOptions {
//!         prompt: vec![Message::User {
//!             content: vec![UserContentPart::Text {
//!                 text: "Hello!".to_string(),
//!             }],
//!         }],
//!         ..Default::default()
//!     };
//!
//!     let response = model.do_generate(options).await.unwrap();
//!     println!("Generated: {:?}", response.content);
//! }
//! ```
//!
//! ## Provider Implementations
//!
//! Provider implementations are maintained in separate crates:
//!
//! - [`ai-sdk-openai`] - OpenAI provider
//!
//! [`ai-sdk-openai`]: https://crates.io/crates/ai-sdk-openai

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]

/// Embedding model interfaces and types for text embedding generation.
pub mod embedding_model;
/// Image model interfaces and types for image generation.
pub mod image_model;
/// JSON value types for provider metadata and structured data.
pub mod json_value;
/// Language model interfaces and types for text generation and chat completion.
pub mod language_model;
/// Reranking model interfaces and types for document reranking.
pub mod reranking_model;
/// Shared types and utilities used across all model types.
pub mod shared;
/// Speech model interfaces and types for text-to-speech synthesis.
pub mod speech_model;
/// Transcription model interfaces and types for speech-to-text transcription.
pub mod transcription_model;

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
pub use reranking_model::{
    Documents, RankingItem, RerankOptions, RerankResponse, RerankingModel,
    ResponseInfo as RerankingResponseInfo,
};
pub use shared::{SharedHeaders, SharedProviderMetadata, SharedProviderOptions, SharedWarning};
pub use speech_model::{
    AudioData, CallWarning as SpeechCallWarning, SpeechGenerateOptions, SpeechGenerateResponse,
    SpeechModel,
};
pub use transcription_model::{
    AudioInput, CallWarning as TranscriptionCallWarning, RequestInfo as TranscriptionRequestInfo,
    ResponseInfo as TranscriptionResponseInfo, TranscriptionModel, TranscriptionOptions,
    TranscriptionResponse, TranscriptionSegment,
};
