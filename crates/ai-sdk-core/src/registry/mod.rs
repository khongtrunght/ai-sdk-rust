//! Provider registry system for multi-provider management.
//!
//! This module provides a registry system for managing multiple AI providers
//! in a single application. It enables:
//!
//! - Registering multiple providers (OpenAI, Anthropic, Google, etc.)
//! - Accessing models via composite IDs (e.g., "openai:gpt-4")
//! - Custom provider implementations
//! - Fallback provider support
//!
//! # Example
//!
//! ```rust,ignore
//! use ai_sdk_core::registry::create_provider_registry;
//! use ai_sdk_openai::OpenAIProvider;
//! use std::sync::Arc;
//!
//! // Create providers
//! let openai = Arc::new(OpenAIProvider::new("api-key"));
//!
//! // Build registry
//! let registry = create_provider_registry(Some(":"))
//!     .with_provider("openai", openai)
//!     .build();
//!
//! // Use models via registry
//! let gpt4 = registry.language_model("openai:gpt-4")?;
//! ```

mod builder;
mod custom_provider;
mod error;
mod provider_registry;

pub use builder::{create_provider_registry, ProviderRegistryBuilder};
pub use custom_provider::CustomProviderBuilder;
pub use error::RegistryError;
pub use provider_registry::{DefaultProviderRegistry, ProviderRegistry};
