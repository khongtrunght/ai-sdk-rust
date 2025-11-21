//! Provider registry for multi-provider management.
//!
//! This module provides the `ProviderRegistry` trait and `DefaultProviderRegistry` implementation
//! for managing multiple AI providers in a single application.

use std::collections::HashMap;
use std::sync::Arc;

use ai_sdk_provider::{EmbeddingModel, ImageModel, LanguageModel, ProviderV3};

use super::error::RegistryError;

/// Provider registry trait for managing multiple providers.
///
/// A provider registry allows applications to work with multiple AI providers
/// (OpenAI, Anthropic, Google, etc.) through a unified interface. Models are
/// accessed using composite IDs in the format `provider_id:model_id`.
///
/// # Example
///
/// ```rust,ignore
/// use ai_sdk_core::registry::{ProviderRegistry, DefaultProviderRegistry};
///
/// let mut registry = DefaultProviderRegistry::new(":");
/// registry.register_provider("openai".to_string(), Arc::new(openai_provider));
///
/// let gpt4 = registry.language_model("openai:gpt-4")?;
/// ```
pub trait ProviderRegistry: Send + Sync {
    /// Register a provider with the registry.
    ///
    /// # Arguments
    /// * `id` - Unique identifier for the provider (e.g., "openai", "anthropic")
    /// * `provider` - The provider implementation
    fn register_provider(&mut self, id: String, provider: Arc<dyn ProviderV3>);

    /// Get a language model by composite ID (provider:model).
    ///
    /// # Arguments
    /// * `id` - Composite ID in format "provider_id:model_id"
    ///
    /// # Returns
    /// * `Ok(model)` if found
    /// * `Err(RegistryError)` if provider or model not found
    ///
    /// # Example
    /// ```rust,ignore
    /// let gpt4 = registry.language_model("openai:gpt-4")?;
    /// ```
    fn language_model(&self, id: &str) -> Result<Arc<dyn LanguageModel>, RegistryError>;

    /// Get an embedding model by composite ID (provider:model).
    ///
    /// # Arguments
    /// * `id` - Composite ID in format "provider_id:model_id"
    ///
    /// # Returns
    /// * `Ok(model)` if found
    /// * `Err(RegistryError)` if provider or model not found
    fn text_embedding_model(
        &self,
        id: &str,
    ) -> Result<Arc<dyn EmbeddingModel<String>>, RegistryError>;

    /// Get an image model by composite ID (provider:model).
    ///
    /// # Arguments
    /// * `id` - Composite ID in format "provider_id:model_id"
    ///
    /// # Returns
    /// * `Ok(model)` if found
    /// * `Err(RegistryError)` if provider or model not found
    fn image_model(&self, id: &str) -> Result<Arc<dyn ImageModel>, RegistryError>;

    /// List all registered provider IDs.
    ///
    /// # Returns
    /// Vector of provider IDs (e.g., ["openai", "anthropic"])
    fn list_providers(&self) -> Vec<String>;
}

/// Default implementation of provider registry.
///
/// This implementation uses a HashMap to store providers and supports
/// custom separators for composite IDs.
///
/// # Example
///
/// ```rust,ignore
/// let mut registry = DefaultProviderRegistry::new(":");
/// registry.register_provider("openai".to_string(), Arc::new(openai_provider));
/// registry.register_provider("anthropic".to_string(), Arc::new(anthropic_provider));
///
/// let gpt4 = registry.language_model("openai:gpt-4")?;
/// let claude = registry.language_model("anthropic:claude-sonnet-4")?;
/// ```
pub struct DefaultProviderRegistry {
    providers: HashMap<String, Arc<dyn ProviderV3>>,
    separator: String,
}

impl DefaultProviderRegistry {
    /// Create a new registry with a custom separator.
    ///
    /// # Arguments
    /// * `separator` - Separator string for composite IDs (e.g., ":", "-")
    ///
    /// # Example
    /// ```rust,ignore
    /// let registry = DefaultProviderRegistry::new(":");
    /// ```
    pub fn new(separator: impl Into<String>) -> Self {
        Self {
            providers: HashMap::new(),
            separator: separator.into(),
        }
    }

    /// Split composite ID into provider ID and model ID.
    ///
    /// # Arguments
    /// * `id` - Composite ID (e.g., "openai:gpt-4")
    /// * `model_type` - Model type name for error messages
    ///
    /// # Returns
    /// * `Ok((provider_id, model_id))` if valid format
    /// * `Err(RegistryError::InvalidModelId)` if invalid format
    fn split_id(&self, id: &str, model_type: &str) -> Result<(String, String), RegistryError> {
        match id.find(&self.separator) {
            Some(index) => {
                let provider_id = id[..index].to_string();
                let model_id = id[index + self.separator.len()..].to_string();
                Ok((provider_id, model_id))
            }
            None => Err(RegistryError::InvalidModelId {
                model_id: id.to_string(),
                model_type: model_type.to_string(),
                message: format!(
                    "Invalid {} id for registry: {} (must be in format 'providerId{}modelId')",
                    model_type, id, self.separator
                ),
            }),
        }
    }

    /// Get provider by ID.
    ///
    /// # Arguments
    /// * `id` - Provider ID
    /// * `model_type` - Model type name for error messages
    ///
    /// # Returns
    /// * `Ok(provider)` if found
    /// * `Err(RegistryError::NoSuchProvider)` if not found
    fn get_provider(
        &self,
        id: &str,
        model_type: &str,
    ) -> Result<&Arc<dyn ProviderV3>, RegistryError> {
        self.providers
            .get(id)
            .ok_or_else(|| RegistryError::NoSuchProvider {
                provider_id: id.to_string(),
                model_type: model_type.to_string(),
                available_providers: self.providers.keys().cloned().collect(),
            })
    }
}

impl ProviderRegistry for DefaultProviderRegistry {
    fn register_provider(&mut self, id: String, provider: Arc<dyn ProviderV3>) {
        self.providers.insert(id, provider);
    }

    fn language_model(&self, id: &str) -> Result<Arc<dyn LanguageModel>, RegistryError> {
        let (provider_id, model_id) = self.split_id(id, "languageModel")?;
        let provider = self.get_provider(&provider_id, "languageModel")?;

        provider
            .language_model(&model_id)
            .ok_or_else(|| RegistryError::NoSuchModel {
                model_id: id.to_string(),
                model_type: "languageModel".to_string(),
            })
    }

    fn text_embedding_model(
        &self,
        id: &str,
    ) -> Result<Arc<dyn EmbeddingModel<String>>, RegistryError> {
        let (provider_id, model_id) = self.split_id(id, "textEmbeddingModel")?;
        let provider = self.get_provider(&provider_id, "textEmbeddingModel")?;

        provider
            .text_embedding_model(&model_id)
            .ok_or_else(|| RegistryError::NoSuchModel {
                model_id: id.to_string(),
                model_type: "textEmbeddingModel".to_string(),
            })
    }

    fn image_model(&self, id: &str) -> Result<Arc<dyn ImageModel>, RegistryError> {
        let (provider_id, model_id) = self.split_id(id, "imageModel")?;
        let provider = self.get_provider(&provider_id, "imageModel")?;

        provider
            .image_model(&model_id)
            .ok_or_else(|| RegistryError::NoSuchModel {
                model_id: id.to_string(),
                model_type: "imageModel".to_string(),
            })
    }

    fn list_providers(&self) -> Vec<String> {
        self.providers.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_sdk_provider::{EmbeddingModel, ImageModel, LanguageModel, ProviderV3};
    use std::sync::Arc;

    // Mock provider for testing
    struct MockProvider;

    impl ProviderV3 for MockProvider {
        fn language_model(&self, model_id: &str) -> Option<Arc<dyn LanguageModel>> {
            if model_id == "test-model" {
                // Return a mock language model
                // For now, return None since we don't have a mock implementation
                // In real tests, you'd create a proper mock
                None
            } else {
                None
            }
        }

        fn text_embedding_model(&self, _model_id: &str) -> Option<Arc<dyn EmbeddingModel<String>>> {
            None
        }

        fn image_model(&self, _model_id: &str) -> Option<Arc<dyn ImageModel>> {
            None
        }
    }

    #[test]
    fn test_split_id() {
        let registry = DefaultProviderRegistry::new(":");

        let result = registry.split_id("openai:gpt-4", "languageModel");
        assert!(result.is_ok());
        let (provider_id, model_id) = result.unwrap();
        assert_eq!(provider_id, "openai");
        assert_eq!(model_id, "gpt-4");
    }

    #[test]
    fn test_split_id_invalid() {
        let registry = DefaultProviderRegistry::new(":");

        let result = registry.split_id("invalid-id", "languageModel");
        assert!(result.is_err());

        if let Err(RegistryError::InvalidModelId { model_id, .. }) = result {
            assert_eq!(model_id, "invalid-id");
        } else {
            panic!("Expected InvalidModelId error");
        }
    }

    #[test]
    fn test_register_and_list_providers() {
        let mut registry = DefaultProviderRegistry::new(":");

        let provider = Arc::new(MockProvider);
        registry.register_provider("test".to_string(), provider);

        let providers = registry.list_providers();
        assert_eq!(providers.len(), 1);
        assert!(providers.contains(&"test".to_string()));
    }

    #[test]
    fn test_get_provider_not_found() {
        let registry = DefaultProviderRegistry::new(":");

        let result = registry.get_provider("nonexistent", "languageModel");
        assert!(result.is_err());

        if let Err(RegistryError::NoSuchProvider {
            provider_id,
            available_providers,
            ..
        }) = result
        {
            assert_eq!(provider_id, "nonexistent");
            assert_eq!(available_providers.len(), 0);
        } else {
            panic!("Expected NoSuchProvider error");
        }
    }
}
