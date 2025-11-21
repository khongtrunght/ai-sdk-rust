//! Builder pattern for provider registry creation.
//!
//! This module provides a fluent builder API for creating and configuring
//! provider registries.

use std::collections::HashMap;
use std::sync::Arc;

use ai_sdk_provider::ProviderV3;

use super::provider_registry::{DefaultProviderRegistry, ProviderRegistry};

/// Create a new provider registry builder.
///
/// # Arguments
/// * `separator` - Optional separator for composite IDs (defaults to ":")
///
/// # Returns
/// A new `ProviderRegistryBuilder` instance
///
/// # Example
///
/// ```rust,ignore
/// use ai_sdk_core::registry::create_provider_registry;
///
/// let registry = create_provider_registry(Some(":"))
///     .with_provider("openai", openai_provider)
///     .with_provider("anthropic", anthropic_provider)
///     .build();
/// ```
pub fn create_provider_registry(separator: Option<&str>) -> ProviderRegistryBuilder {
    ProviderRegistryBuilder {
        separator: separator.unwrap_or(":").to_string(),
        providers: HashMap::new(),
    }
}

/// Builder for creating provider registries.
///
/// This builder provides a fluent API for constructing registries
/// with multiple providers.
///
/// # Example
///
/// ```rust,ignore
/// let registry = ProviderRegistryBuilder::new()
///     .with_provider("openai", openai_provider)
///     .with_provider("anthropic", anthropic_provider)
///     .build();
/// ```
pub struct ProviderRegistryBuilder {
    separator: String,
    providers: HashMap<String, Arc<dyn ProviderV3>>,
}

impl ProviderRegistryBuilder {
    /// Create a new builder with default settings (separator: ":").
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let builder = ProviderRegistryBuilder::new();
    /// ```
    pub fn new() -> Self {
        Self {
            separator: ":".to_string(),
            providers: HashMap::new(),
        }
    }

    /// Add a provider to the registry.
    ///
    /// # Arguments
    /// * `id` - Provider identifier (e.g., "openai", "anthropic")
    /// * `provider` - The provider implementation
    ///
    /// # Returns
    /// Self for method chaining
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// builder.with_provider("openai", Arc::new(openai_provider));
    /// ```
    pub fn with_provider(mut self, id: impl Into<String>, provider: Arc<dyn ProviderV3>) -> Self {
        self.providers.insert(id.into(), provider);
        self
    }

    /// Build the provider registry.
    ///
    /// # Returns
    /// A configured `DefaultProviderRegistry` instance
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let registry = builder.build();
    /// ```
    pub fn build(self) -> DefaultProviderRegistry {
        let mut registry = DefaultProviderRegistry::new(self.separator);

        for (id, provider) in self.providers {
            registry.register_provider(id, provider);
        }

        registry
    }
}

impl Default for ProviderRegistryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_sdk_provider::{EmbeddingModel, ImageModel, LanguageModel, ProviderV3};

    struct MockProvider;

    impl ProviderV3 for MockProvider {
        fn language_model(&self, _model_id: &str) -> Option<Arc<dyn LanguageModel>> {
            None
        }

        fn text_embedding_model(&self, _model_id: &str) -> Option<Arc<dyn EmbeddingModel<String>>> {
            None
        }

        fn image_model(&self, _model_id: &str) -> Option<Arc<dyn ImageModel>> {
            None
        }
    }

    #[test]
    fn test_builder_creation() {
        let builder = ProviderRegistryBuilder::new();
        assert_eq!(builder.separator, ":");
        assert_eq!(builder.providers.len(), 0);
    }

    #[test]
    fn test_builder_with_provider() {
        let provider = Arc::new(MockProvider);
        let builder = ProviderRegistryBuilder::new().with_provider("test", provider);

        assert_eq!(builder.providers.len(), 1);
        assert!(builder.providers.contains_key("test"));
    }

    #[test]
    fn test_create_provider_registry() {
        let provider = Arc::new(MockProvider);
        let registry = create_provider_registry(Some(":"))
            .with_provider("test", provider)
            .build();

        let providers = registry.list_providers();
        assert_eq!(providers.len(), 1);
        assert!(providers.contains(&"test".to_string()));
    }

    #[test]
    fn test_default_separator() {
        let registry = create_provider_registry(None).build();
        // Default separator should be ":"
        // We can't directly test the separator, but we can test that it works
        assert_eq!(registry.list_providers().len(), 0);
    }
}
