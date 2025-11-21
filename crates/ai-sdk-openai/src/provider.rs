//! OpenAI provider implementation for ProviderV3 trait.
//!
//! This module provides the `OpenAIProvider` struct which implements the
//! `ProviderV3` trait, allowing creation of OpenAI model instances by model ID.

use ai_sdk_provider::{EmbeddingModel, ImageModel, LanguageModel, ProviderV3, TranscriptionModel};
use std::{collections::HashMap, sync::Arc};

use crate::{
    openai_config::OpenAIConfig, OpenAIChatModel, OpenAIEmbeddingModel, OpenAIImageModel,
    OpenAISpeechModel, OpenAITranscriptionModel,
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
    settings: OpenAIProviderSettings,
}

#[derive(Default)]
struct OpenAIProviderSettings {
    base_url: String,
    api_key: String,
    organization: Option<String>,
    project: Option<String>,
    headers: HashMap<String, String>,
    name: String,
}

pub struct OpenAIProviderBuilder {
    base_url: Option<String>,
    api_key: Option<String>,
    organization: Option<String>,
    project: Option<String>,
    headers: HashMap<String, String>,
    name: Option<String>,
}

impl Default for OpenAIProviderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenAIProviderBuilder {
    pub fn new() -> Self {
        Self {
            base_url: None,
            api_key: None,
            organization: None,
            project: None,
            headers: HashMap::new(),
            name: None,
        }
    }

    /// Sets the API Key.
    /// If not set, the builder will look for the `OPENAI_API_KEY` environment variable.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Sets the Base URL.
    /// Defaults to "https://api.openai.com/v1" or `OPENAI_BASE_URL` env var.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Sets the Organization ID.
    pub fn with_organization(mut self, organization: impl Into<String>) -> Self {
        self.organization = Some(organization.into());
        self
    }

    /// Sets the Project ID.
    pub fn with_project(mut self, project: impl Into<String>) -> Self {
        self.project = Some(project.into());
        self
    }

    /// Adds a custom header to all requests.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }

    /// Sets the provider name (defaults to "openai").
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Consumes the builder and produces the `OpenAIProvider`.
    ///
    /// This performs environment variable lookups for missing fields.
    pub fn build(self) -> OpenAIProvider {
        // Resolve API Key
        let api_key = self
            .api_key
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .unwrap_or_default(); // Or panic/log warning if strict validation is required

        // Resolve Base URL
        let base_url = self
            .base_url
            .or_else(|| std::env::var("OPENAI_BASE_URL").ok())
            .map(|u| u.trim_end_matches('/').to_string())
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());

        OpenAIProvider {
            settings: OpenAIProviderSettings {
                api_key,
                base_url,
                organization: self.organization,
                project: self.project,
                headers: self.headers,
                name: self.name.unwrap_or_else(|| "openai".to_string()),
            },
        }
    }
}

impl OpenAIProvider {
    /// Creates a new builder for configuring the OpenAI provider.
    pub fn builder() -> OpenAIProviderBuilder {
        OpenAIProviderBuilder::new()
    }

    /// Convenience constructor: creates a provider from the environment variables.
    /// Looks for `OPENAI_API_KEY`.
    pub fn from_env() -> Self {
        Self::builder().build()
    }

    /// Convenience constructor: creates a provider with a specific API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::builder().with_api_key(api_key).build()
    }

    fn create_chat_model(&self, model_id: &str) -> Option<Arc<dyn LanguageModel>> {
        Some(Arc::new(OpenAIChatModel::new(
            model_id,
            OpenAIConfig::new(
                format!("{}.chat", self.provider_name()),
                // URL Factory
                {
                    let base = self.settings.base_url.clone();
                    move |options| format!("{}/{}", base, options.path.trim_start_matches('/'))
                },
                // Header Factory
                {
                    let headers = self.get_headers();
                    move || headers.clone()
                },
            ),
        )))
    }

    fn provider_name(&self) -> String {
        self.settings.name.clone()
    }

    fn get_headers(&self) -> HashMap<String, String> {
        let mut original_headers = self.settings.headers.clone();

        original_headers.insert(
            "Authorization".to_string(),
            format!("Bearer {}", self.settings.api_key),
        );

        if let Some(org) = &self.settings.organization {
            original_headers.insert("OpenAI-Organization".to_string(), org.clone());
        }

        if let Some(proj) = &self.settings.project {
            original_headers.insert("OpenAI-Project".to_string(), proj.clone());
        }

        original_headers
    }
}

/// Validates if a model ID is a valid OpenAI language model
fn is_valid_language_model(model_id: &str) -> bool {
    model_id.starts_with("gpt-")
        || model_id.starts_with("o1")
        || model_id.starts_with("o3")
        || model_id.starts_with("o4")
        || model_id.starts_with("chatgpt-")
}

/// Validates if a model ID is a valid OpenAI embedding model
fn is_valid_embedding_model(model_id: &str) -> bool {
    model_id.starts_with("text-embedding-")
}

/// Validates if a model ID is a valid OpenAI image model
fn is_valid_image_model(model_id: &str) -> bool {
    model_id.starts_with("dall-e-") || model_id.starts_with("gpt-image-")
}

/// Validates if a model ID is a valid OpenAI transcription model
fn is_valid_transcription_model(model_id: &str) -> bool {
    model_id.contains("whisper") || model_id.contains("transcribe")
}

/// Validates if a model ID is a valid OpenAI speech model
fn is_valid_speech_model(model_id: &str) -> bool {
    model_id.starts_with("tts-") || model_id == "chatgpt-4o-audio-preview"
}

impl ProviderV3 for OpenAIProvider {
    fn language_model(&self, model_id: &str) -> Option<Arc<dyn LanguageModel>> {
        if !is_valid_language_model(model_id) {
            return None;
        }
        self.create_chat_model(model_id)
    }

    fn text_embedding_model(&self, model_id: &str) -> Option<Arc<dyn EmbeddingModel<String>>> {
        if !is_valid_embedding_model(model_id) {
            return None;
        }
        Some(Arc::new(OpenAIEmbeddingModel::new(
            model_id,
            OpenAIConfig::new(
                format!("{}.embedding", self.provider_name()),
                // URL Factory
                {
                    let base = self.settings.base_url.clone();
                    move |options| format!("{}/{}", base, options.path.trim_start_matches('/'))
                },
                // Header Factory
                {
                    let headers = self.get_headers();
                    move || headers.clone()
                },
            ),
        )))
    }

    fn image_model(&self, model_id: &str) -> Option<Arc<dyn ImageModel>> {
        if !is_valid_image_model(model_id) {
            return None;
        }
        Some(Arc::new(OpenAIImageModel::new(
            model_id,
            OpenAIConfig::new(
                format!("{}.image", self.provider_name()),
                // URL Factory
                {
                    let base = self.settings.base_url.clone();
                    move |options| format!("{}/{}", base, options.path.trim_start_matches('/'))
                },
                // Header Factory
                {
                    let headers = self.get_headers();
                    move || headers.clone()
                },
            ),
        )))
    }

    fn transcription_model(&self, model_id: &str) -> Option<Arc<dyn TranscriptionModel>> {
        if !is_valid_transcription_model(model_id) {
            return None;
        }
        Some(Arc::new(OpenAITranscriptionModel::new(
            model_id,
            OpenAIConfig::new(
                format!("{}.transcription", self.provider_name()),
                // URL Factory
                {
                    let base = self.settings.base_url.clone();
                    move |options| format!("{}/{}", base, options.path.trim_start_matches('/'))
                },
                // Header Factory
                {
                    let headers = self.get_headers();
                    move || headers.clone()
                },
            ),
        )))
    }

    fn speech_model(&self, model_id: &str) -> Option<Arc<dyn ai_sdk_provider::SpeechModel>> {
        if !is_valid_speech_model(model_id) {
            return None;
        }
        Some(Arc::new(OpenAISpeechModel::new(
            model_id,
            OpenAIConfig::new(
                format!("{}.speech", self.provider_name()),
                // URL Factory
                {
                    let base = self.settings.base_url.clone();
                    move |options| format!("{}/{}", base, options.path.trim_start_matches('/'))
                },
                // Header Factory
                {
                    let headers = self.get_headers();
                    move || headers.clone()
                },
            ),
        )))
    }

    // reranking_model uses default implementation (returns None)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to construct a provider with a dummy API key for testing.
    /// This avoids relying on environment variables or repeating the builder setup.
    fn test_provider() -> OpenAIProvider {
        OpenAIProvider::builder().with_api_key("test-key").build()
    }

    #[test]
    fn test_provider_creation() {
        let provider = test_provider();

        // Assuming the implementation validates model IDs,
        // or simply returns a configured model for valid-looking strings.
        assert!(provider.language_model("gpt-4").is_some());
        assert!(provider.language_model("gpt-3.5-turbo").is_some());

        // Note: If your implementation is a "pass-through" factory (it accepts any string),
        // this assertion might need to be removed or the implementation needs an allow-list.
        assert!(provider.language_model("invalid-model").is_none());
    }

    #[test]
    fn test_embedding_models() {
        let provider = test_provider();

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
        let provider = test_provider();

        assert!(provider.image_model("dall-e-3").is_some());
        assert!(provider.image_model("dall-e-2").is_some());

        assert!(provider.image_model("invalid-model").is_none());
    }

    #[test]
    fn test_transcription_models() {
        let provider = test_provider();

        assert!(provider.transcription_model("whisper-1").is_some());

        assert!(provider.transcription_model("invalid-model").is_none());
    }

    #[test]
    fn test_speech_models() {
        let provider = test_provider();

        assert!(provider.speech_model("tts-1").is_some());
        assert!(provider.speech_model("tts-1-hd").is_some());

        assert!(provider.speech_model("invalid-model").is_none());
    }

    #[test]
    fn test_specification_version() {
        let provider = test_provider();
        assert_eq!(provider.specification_version(), "v3");
    }
}
