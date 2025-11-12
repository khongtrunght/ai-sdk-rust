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
//! ```rust,no_run
//! use ai_sdk_openai::OpenAIChatModel;
//! use ai_sdk_provider::{LanguageModel, CallOptions, Message};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let api_key = std::env::var("OPENAI_API_KEY")?;
//!     let model = OpenAIChatModel::new("gpt-4", api_key);
//!
//!     let options = CallOptions {
//!         messages: vec![Message::user("Hello!")],
//!         ..Default::default()
//!     };
//!
//!     let response = model.do_generate(options).await?;
//!     println!("{}", response.text);
//!
//!     Ok(())
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
mod speech;
mod transcription;

pub use chat::OpenAIChatModel;
pub use embedding::OpenAIEmbeddingModel;
pub use error::OpenAIError;
pub use image::OpenAIImageModel;
pub use speech::OpenAISpeechModel;
pub use transcription::OpenAITranscriptionModel;

// Factory functions
pub fn openai(model_id: impl Into<String>, api_key: impl Into<String>) -> OpenAIChatModel {
    OpenAIChatModel::new(model_id, api_key)
}

pub fn openai_embedding(
    model_id: impl Into<String>,
    api_key: impl Into<String>,
) -> OpenAIEmbeddingModel {
    OpenAIEmbeddingModel::new(model_id, api_key)
}

pub fn openai_image(
    model_id: impl Into<String>,
    api_key: impl Into<String>,
) -> OpenAIImageModel {
    OpenAIImageModel::new(model_id, api_key)
}

pub fn openai_speech(
    model_id: impl Into<String>,
    api_key: impl Into<String>,
) -> OpenAISpeechModel {
    OpenAISpeechModel::new(model_id, api_key)
}

pub fn openai_transcription(
    model_id: impl Into<String>,
    api_key: impl Into<String>,
) -> OpenAITranscriptionModel {
    OpenAITranscriptionModel::new(model_id, api_key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_sdk_provider::LanguageModel;

    #[test]
    fn test_model_creation() {
        let model = openai("gpt-4", "test-key");
        assert_eq!(model.provider(), "openai");
        assert_eq!(model.model_id(), "gpt-4");
    }
}
