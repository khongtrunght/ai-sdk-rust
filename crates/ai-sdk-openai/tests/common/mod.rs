//! Common test utilities for ai-sdk-openai integration tests.
//!
//! This module is not a test itself - it's imported by test files using `mod common;`

pub mod fixtures;
pub mod mock_server;

// Re-export commonly used items for convenience
#[allow(unused_imports)]
pub use fixtures::{load_chunks_fixture, load_json_fixture};
pub use mock_server::TestServer;

use ai_sdk_openai::OpenAIProvider;
use ai_sdk_provider::{LanguageModel, ProviderV3};
use std::sync::Arc;

/// Helper to create a test model with a custom base URL
#[allow(dead_code)]
pub fn create_test_model(base_url: &str, model_id: &str) -> Arc<dyn LanguageModel> {
    let provider = OpenAIProvider::builder()
        .with_api_key("test-key")
        .with_base_url(format!("{}/v1", base_url))
        .build();

    provider.language_model(model_id).unwrap()
}
