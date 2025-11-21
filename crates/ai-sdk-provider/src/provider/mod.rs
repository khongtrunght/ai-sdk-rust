//! Provider trait for AI model factories.
//!
//! This module defines the `ProviderV3` trait which is a factory interface
//! for creating model instances by their ID. Providers enable multi-provider
//! applications and consistent model instantiation across different AI services.

use std::sync::Arc;

use crate::embedding_model::EmbeddingModel;
use crate::image_model::ImageModel;
use crate::language_model::LanguageModel;
use crate::reranking_model::RerankingModel;
use crate::speech_model::SpeechModel;
use crate::transcription_model::TranscriptionModel;

/// Provider for language, embedding, image, and other AI models.
///
/// A provider is a factory that creates model instances by their ID.
/// For example, the OpenAI provider returns GPT-4 when asked for "gpt-4".
///
/// This trait enables:
/// - Multi-provider applications (OpenAI, Anthropic, Google)
/// - Provider registry systems
/// - Custom provider implementations
/// - Consistent model instantiation across providers
///
/// # Example
///
/// ```rust,ignore
/// use ai_sdk_provider::ProviderV3;
///
/// let openai = OpenAIProvider::new(api_key);
/// let gpt4 = openai.language_model("gpt-4").unwrap();
/// ```
pub trait ProviderV3: Send + Sync {
    /// Specification version (always "v3")
    fn specification_version(&self) -> &str {
        "v3"
    }

    /// Create a language model instance by model ID.
    ///
    /// # Arguments
    /// * `model_id` - The model identifier (e.g., "gpt-4", "claude-sonnet-4")
    ///
    /// # Returns
    /// * `Some(model)` if the model exists and is supported
    /// * `None` if no such model is available from this provider
    ///
    /// # Example
    /// ```rust,ignore
    /// let openai = OpenAIProvider::new(api_key);
    /// let gpt4 = openai.language_model("gpt-4").unwrap();
    /// ```
    fn language_model(&self, model_id: &str) -> Option<Arc<dyn LanguageModel>>;

    /// Create a text embedding model instance by model ID.
    ///
    /// # Arguments
    /// * `model_id` - The model identifier (e.g., "text-embedding-3-small")
    ///
    /// # Returns
    /// * `Some(model)` if the model exists
    /// * `None` if no such model is available
    fn text_embedding_model(&self, model_id: &str) -> Option<Arc<dyn EmbeddingModel<String>>>;

    /// Create an image generation model instance by model ID.
    ///
    /// # Arguments
    /// * `model_id` - The model identifier (e.g., "dall-e-3")
    ///
    /// # Returns
    /// * `Some(model)` if the model exists
    /// * `None` if no such model is available
    fn image_model(&self, model_id: &str) -> Option<Arc<dyn ImageModel>>;

    /// Create a transcription model instance by model ID (optional).
    ///
    /// # Arguments
    /// * `model_id` - The model identifier (e.g., "whisper-1")
    ///
    /// # Returns
    /// * `Some(model)` if the model exists
    /// * `None` if not supported by this provider
    fn transcription_model(&self, _model_id: &str) -> Option<Arc<dyn TranscriptionModel>> {
        None
    }

    /// Create a speech synthesis model instance by model ID (optional).
    ///
    /// # Arguments
    /// * `model_id` - The model identifier (e.g., "tts-1")
    ///
    /// # Returns
    /// * `Some(model)` if the model exists
    /// * `None` if not supported by this provider
    fn speech_model(&self, _model_id: &str) -> Option<Arc<dyn SpeechModel>> {
        None
    }

    /// Create a reranking model instance by model ID (optional).
    ///
    /// # Arguments
    /// * `model_id` - The model identifier
    ///
    /// # Returns
    /// * `Some(model)` if the model exists
    /// * `None` if not supported by this provider
    fn reranking_model(&self, _model_id: &str) -> Option<Arc<dyn RerankingModel>> {
        None
    }
}
