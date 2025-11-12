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
//! ```rust,no_run
//! use ai_sdk_provider::{LanguageModel, CallOptions};
//!
//! async fn generate_text<M: LanguageModel>(model: &M) {
//!     let options = CallOptions {
//!         // ... configure options
//!         ..Default::default()
//!     };
//!
//!     let response = model.do_generate(options).await.unwrap();
//!     println!("Generated: {}", response.text);
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

pub mod embedding_model;
pub mod image_model;
pub mod json_value;
pub mod language_model;
pub mod reranking_model;
pub mod shared;
pub mod speech_model;
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
