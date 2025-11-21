//! # OpenAI Provider for AI SDK
//!
//! This crate provides OpenAI implementations for the AI SDK provider
//! specification. It supports GPT language models, DALL-E image generation,
//! Whisper transcription, embeddings, and text-to-speech.
//!
//! ## Features
//!
//! - **GPT Models** - GPT-4, GPT-3.5 Turbo chat completion
//! - **Embeddings** - text-embedding-3-small, text-embedding-3-large
//! - **Image Generation** - DALL-E 2, DALL-E 3
//! - **Speech Synthesis** - TTS-1, TTS-1-HD
//! - **Transcription** - Whisper-1
//!
//! ## Example
//!
//! ```rust,ignore
//! use ai_sdk_openai::OpenAIChatModel;
//! use ai_sdk_provider::{LanguageModel, CallOptions, Message, UserContentPart};
//!
//! #[tokio::main]
//! async fn main() {
//!     let api_key = std::env::var("OPENAI_API_KEY").unwrap();
//!     let model = OpenAIChatModel::new("gpt-4", api_key);
//!
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
//!     println!("{}", response.text);
//! }
//! ```
//!
//! ## API Key
//!
//! You'll need an OpenAI API key from https://platform.openai.com/api-keys

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]

mod api_types;
mod chat;
mod embedding;
mod error;
mod image;
pub mod model_detection;
mod multimodal;
mod openai_config;
mod provider;
pub mod responses;
mod speech;
mod transcription;

pub use chat::OpenAIChatModel;
pub use embedding::OpenAIEmbeddingModel;
pub use error::OpenAIError;
pub use image::OpenAIImageModel;
pub use multimodal::{convert_audio_part, convert_image_part, MultimodalError, OpenAIContentPart};
pub use openai_config::{OpenAIConfig, OpenAIUrlOptions};
pub use provider::OpenAIProvider;
pub use speech::OpenAISpeechModel;
pub use transcription::OpenAITranscriptionModel;
