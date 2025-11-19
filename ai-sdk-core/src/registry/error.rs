//! Registry error types.
//!
//! This module defines error types for provider registry operations.

use thiserror::Error;

/// Registry-specific errors
#[derive(Debug, Error)]
pub enum RegistryError {
    /// Provider not found in registry
    #[error("No such provider: {provider_id} (available providers: {available_providers:?})")]
    NoSuchProvider {
        /// The provider ID that was not found
        provider_id: String,
        /// The model type being requested
        model_type: String,
        /// List of available provider IDs
        available_providers: Vec<String>,
    },

    /// Model not found in provider
    #[error("No such model: {model_id} (type: {model_type})")]
    NoSuchModel {
        /// The model ID that was not found
        model_id: String,
        /// The model type being requested
        model_type: String,
    },

    /// Invalid model ID format
    #[error("Invalid model ID: {message}")]
    InvalidModelId {
        /// The invalid model ID
        model_id: String,
        /// The model type being requested
        model_type: String,
        /// Error message explaining what's wrong
        message: String,
    },
}
