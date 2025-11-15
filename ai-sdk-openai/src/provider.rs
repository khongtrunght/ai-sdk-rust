//! OpenAI provider implementation for ProviderV3 trait.
//!
//! This module provides the `OpenAIProvider` struct which implements the
//! `ProviderV3` trait, allowing creation of OpenAI model instances by model ID.

use ai_sdk_provider::{EmbeddingModel, ImageModel, LanguageModel, ProviderV3, TranscriptionModel};
use std::sync::Arc;

use crate::{
    OpenAIChatModel, OpenAIEmbeddingModel, OpenAIImageModel, OpenAISpeechModel,
    OpenAITranscriptionModel,
};

/// OpenAI provider for creating model instances by model ID.
///
/// This provider supports:
/// - Language models: GPT-4, GPT-4 Turbo, GPT-3.5 Turbo
/// - Embedding models: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
/// - Image models: DALL-E 2, DALL-E 3
/// - Transcription models: Whisper-1
/// - Speech models: TTS-1, TTS-1-HD
///
/// # Example
///
/// ```rust,ignore
/// use ai_sdk_openai::OpenAIProvider;
/// use ai_sdk_provider::ProviderV3;
///
/// let provider = OpenAIProvider::new("your-api-key");
/// let gpt4 = provider.language_model("gpt-4").unwrap();
/// let embedder = provider.text_embedding_model("text-embedding-3-small").unwrap();
/// ```
pub struct OpenAIProvider {
    api_key: String,
}

impl OpenAIProvider {
    /// Create a new OpenAI provider with the given API key.
    ///
    /// # Arguments
    /// * `api_key` - Your OpenAI API key
    ///
    /// # Example
    /// ```rust,ignore
    /// let provider = OpenAIProvider::new("sk-...");
    /// ```
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
        }
    }
}

impl ProviderV3 for OpenAIProvider {
    fn language_model(&self, model_id: &str) -> Option<Arc<dyn LanguageModel>> {
        // Validate model ID and return if supported
        match model_id {
            // GPT-4 models
            "gpt-4" | "gpt-4-turbo" | "gpt-4-turbo-preview" | "gpt-4-0125-preview"
            | "gpt-4-1106-preview" | "gpt-4-vision-preview" | "gpt-4-0314" | "gpt-4-0613"
            | "gpt-4-32k" | "gpt-4-32k-0314" | "gpt-4-32k-0613"
            // GPT-4o models
            | "gpt-4o" | "gpt-4o-mini" | "gpt-4o-2024-08-06" | "gpt-4o-2024-05-13"
            | "gpt-4o-mini-2024-07-18"
            // GPT-3.5 models
            | "gpt-3.5-turbo" | "gpt-3.5-turbo-0125" | "gpt-3.5-turbo-1106"
            | "gpt-3.5-turbo-0613" | "gpt-3.5-turbo-16k" | "gpt-3.5-turbo-16k-0613" => {
                Some(Arc::new(OpenAIChatModel::new(model_id, self.api_key.clone())))
            }
            _ => None,
        }
    }

    fn text_embedding_model(&self, model_id: &str) -> Option<Arc<dyn EmbeddingModel<String>>> {
        match model_id {
            "text-embedding-3-small" | "text-embedding-3-large" | "text-embedding-ada-002" => Some(
                Arc::new(OpenAIEmbeddingModel::new(model_id, self.api_key.clone())),
            ),
            _ => None,
        }
    }

    fn image_model(&self, model_id: &str) -> Option<Arc<dyn ImageModel>> {
        match model_id {
            "dall-e-3" | "dall-e-2" => Some(Arc::new(OpenAIImageModel::new(
                model_id,
                self.api_key.clone(),
            ))),
            _ => None,
        }
    }

    fn transcription_model(&self, model_id: &str) -> Option<Arc<dyn TranscriptionModel>> {
        match model_id {
            "whisper-1" => Some(Arc::new(OpenAITranscriptionModel::new(
                model_id,
                self.api_key.clone(),
            ))),
            _ => None,
        }
    }

    fn speech_model(&self, model_id: &str) -> Option<Arc<dyn ai_sdk_provider::SpeechModel>> {
        match model_id {
            "tts-1" | "tts-1-hd" => Some(Arc::new(OpenAISpeechModel::new(
                model_id,
                self.api_key.clone(),
            ))),
            _ => None,
        }
    }

    // reranking_model uses default implementation (returns None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = OpenAIProvider::new("test-key");
        assert!(provider.language_model("gpt-4").is_some());
        assert!(provider.language_model("gpt-3.5-turbo").is_some());
        assert!(provider.language_model("invalid-model").is_none());
    }

    #[test]
    fn test_embedding_models() {
        let provider = OpenAIProvider::new("test-key");
        assert!(provider
            .text_embedding_model("text-embedding-3-small")
            .is_some());
        assert!(provider
            .text_embedding_model("text-embedding-3-large")
            .is_some());
        assert!(provider.text_embedding_model("invalid-model").is_none());
    }

    #[test]
    fn test_image_models() {
        let provider = OpenAIProvider::new("test-key");
        assert!(provider.image_model("dall-e-3").is_some());
        assert!(provider.image_model("dall-e-2").is_some());
        assert!(provider.image_model("invalid-model").is_none());
    }

    #[test]
    fn test_transcription_models() {
        let provider = OpenAIProvider::new("test-key");
        assert!(provider.transcription_model("whisper-1").is_some());
        assert!(provider.transcription_model("invalid-model").is_none());
    }

    #[test]
    fn test_speech_models() {
        let provider = OpenAIProvider::new("test-key");
        assert!(provider.speech_model("tts-1").is_some());
        assert!(provider.speech_model("tts-1-hd").is_some());
        assert!(provider.speech_model("invalid-model").is_none());
    }

    #[test]
    fn test_specification_version() {
        let provider = OpenAIProvider::new("test-key");
        assert_eq!(provider.specification_version(), "v3");
    }
}
