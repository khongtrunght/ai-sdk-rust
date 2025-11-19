//! Common test utilities for ai-sdk-openai integration tests.
//!
//! This module is not a test itself - it's imported by test files using `mod common;`

pub mod fixtures;
pub mod mock_server;

// Re-export commonly used items for convenience
#[allow(unused_imports)]
pub use fixtures::{load_chunks_fixture, load_json_fixture};
pub use mock_server::TestServer;
