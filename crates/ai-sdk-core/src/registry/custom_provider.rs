//! Custom provider factory for creating providers with specific model instances.
//!
//! This module provides a builder for creating custom providers that combine
//! specific model instances with optional fallback to another provider.

use std::collections::HashMap;
use std::sync::Arc;

use ai_sdk_provider::{EmbeddingModel, ImageModel, LanguageModel, ProviderV3};

/// Builder for creating custom providers.
///
/// A custom provider allows you to:
/// - Map specific model IDs to model instances
/// - Provide a fallback provider for unmapped models
/// - Override specific models from a base provider
///
/// # Example
///
/// ```rust,ignore
/// use ai_sdk_core::registry::CustomProviderBuilder;
///
/// let custom = CustomProviderBuilder::new()
///     .language_model("my-gpt4", Arc::new(custom_gpt4_impl))
///     .fallback_provider(Arc::new(openai_provider))
///     .build();
/// ```
pub struct CustomProviderBuilder {
    language_models: HashMap<String, Arc<dyn LanguageModel>>,
    embedding_models: HashMap<String, Arc<dyn EmbeddingModel<String>>>,
    image_models: HashMap<String, Arc<dyn ImageModel>>,
    fallback_provider: Option<Arc<dyn ProviderV3>>,
}

impl CustomProviderBuilder {
    /// Create a new custom provider builder.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let builder = CustomProviderBuilder::new();
    /// ```
    pub fn new() -> Self {
        Self {
            language_models: HashMap::new(),
            embedding_models: HashMap::new(),
            image_models: HashMap::new(),
            fallback_provider: None,
        }
    }

    /// Add a language model to the provider.
    ///
    /// # Arguments
    /// * `model_id` - Model identifier (e.g., "custom-gpt4")
    /// * `model` - The model implementation
    ///
    /// # Returns
    /// Self for method chaining
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// builder.language_model("custom-gpt4", Arc::new(my_model));
    /// ```
    pub fn language_model(
        mut self,
        model_id: impl Into<String>,
        model: Arc<dyn LanguageModel>,
    ) -> Self {
        self.language_models.insert(model_id.into(), model);
        self
    }

    /// Add a text embedding model to the provider.
    ///
    /// # Arguments
    /// * `model_id` - Model identifier (e.g., "custom-embedder")
    /// * `model` - The model implementation
    ///
    /// # Returns
    /// Self for method chaining
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// builder.text_embedding_model("custom-embedder", Arc::new(my_embedder));
    /// ```
    pub fn text_embedding_model(
        mut self,
        model_id: impl Into<String>,
        model: Arc<dyn EmbeddingModel<String>>,
    ) -> Self {
        self.embedding_models.insert(model_id.into(), model);
        self
    }

    /// Add an image model to the provider.
    ///
    /// # Arguments
    /// * `model_id` - Model identifier (e.g., "custom-dalle")
    /// * `model` - The model implementation
    ///
    /// # Returns
    /// Self for method chaining
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// builder.image_model("custom-dalle", Arc::new(my_image_model));
    /// ```
    pub fn image_model(mut self, model_id: impl Into<String>, model: Arc<dyn ImageModel>) -> Self {
        self.image_models.insert(model_id.into(), model);
        self
    }

    /// Set a fallback provider for unmapped models.
    ///
    /// When a model is requested that isn't in the custom mappings,
    /// the fallback provider will be queried.
    ///
    /// # Arguments
    /// * `provider` - The fallback provider
    ///
    /// # Returns
    /// Self for method chaining
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// builder.fallback_provider(Arc::new(openai_provider));
    /// ```
    pub fn fallback_provider(mut self, provider: Arc<dyn ProviderV3>) -> Self {
        self.fallback_provider = Some(provider);
        self
    }

    /// Build the custom provider.
    ///
    /// # Returns
    /// A configured `ProviderV3` implementation
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let provider = builder.build();
    /// ```
    pub fn build(self) -> Arc<dyn ProviderV3> {
        Arc::new(CustomProvider {
            language_models: self.language_models,
            embedding_models: self.embedding_models,
            image_models: self.image_models,
            fallback_provider: self.fallback_provider,
        })
    }
}

impl Default for CustomProviderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Custom provider implementation.
///
/// This provider combines specific model mappings with optional fallback.
struct CustomProvider {
    language_models: HashMap<String, Arc<dyn LanguageModel>>,
    embedding_models: HashMap<String, Arc<dyn EmbeddingModel<String>>>,
    image_models: HashMap<String, Arc<dyn ImageModel>>,
    fallback_provider: Option<Arc<dyn ProviderV3>>,
}

impl ProviderV3 for CustomProvider {
    fn specification_version(&self) -> &str {
        "v3"
    }

    fn language_model(&self, model_id: &str) -> Option<Arc<dyn LanguageModel>> {
        // Check custom mappings first
        if let Some(model) = self.language_models.get(model_id) {
            return Some(model.clone());
        }

        // Fall back to fallback provider
        if let Some(fallback) = &self.fallback_provider {
            return fallback.language_model(model_id);
        }

        None
    }

    fn text_embedding_model(&self, model_id: &str) -> Option<Arc<dyn EmbeddingModel<String>>> {
        // Check custom mappings first
        if let Some(model) = self.embedding_models.get(model_id) {
            return Some(model.clone());
        }

        // Fall back to fallback provider
        if let Some(fallback) = &self.fallback_provider {
            return fallback.text_embedding_model(model_id);
        }

        None
    }

    fn image_model(&self, model_id: &str) -> Option<Arc<dyn ImageModel>> {
        // Check custom mappings first
        if let Some(model) = self.image_models.get(model_id) {
            return Some(model.clone());
        }

        // Fall back to fallback provider
        if let Some(fallback) = &self.fallback_provider {
            return fallback.image_model(model_id);
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockProvider;

    impl ProviderV3 for MockProvider {
        fn language_model(&self, model_id: &str) -> Option<Arc<dyn LanguageModel>> {
            if model_id == "fallback-model" {
                // Would return a real model in practice
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
    fn test_custom_provider_builder() {
        let builder = CustomProviderBuilder::new();
        let provider = builder.build();

        // Provider should exist but have no models
        assert!(provider.language_model("nonexistent").is_none());
    }

    #[test]
    fn test_custom_provider_with_fallback() {
        let fallback = Arc::new(MockProvider);
        let provider = CustomProviderBuilder::new()
            .fallback_provider(fallback)
            .build();

        // Should query fallback provider
        let _ = provider.language_model("fallback-model");
    }

    #[test]
    fn test_specification_version() {
        let provider = CustomProviderBuilder::new().build();
        assert_eq!(provider.specification_version(), "v3");
    }
}
