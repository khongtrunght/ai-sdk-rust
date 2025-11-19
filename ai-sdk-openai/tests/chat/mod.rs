//! Integration tests for OpenAI Chat Model
//!
//! These tests verify the chat model implementation against fixture-based
//! mock responses, ensuring compatibility with the OpenAI API without
//! requiring real API calls.
//!
//! ## Test Organization
//!
//! Tests are organized by functionality:
//! - `basic_test.rs` - Basic generation, streaming, usage tracking, metadata extraction (9 tests)
//! - `settings_test.rs` - Settings propagation and configuration (8 tests)
//! - `response_format_test.rs` - Response formats: JSON schema, structured outputs (9 tests)
//! - `tool_calling_test.rs` - Function/tool calling scenarios (4 tests)
//! - `model_specific_test.rs` - Model-specific behavior: o1/o3/o4, search models (7 tests)
//! - `extension_settings_test.rs` - OpenAI extension settings: store, metadata, etc. (10 tests)
//! - `advanced_features_test.rs` - Advanced features: annotations, reasoning tokens (5 tests)
//! - `streaming_test.rs` - Advanced streaming: tool deltas, usage, settings (19 tests)
//!
//! ## Fixtures
//!
//! Fixtures are stored in `fixtures/` and organized by category.
//! See `fixtures/README.md` for details.
//!
//! ## Test Coverage
//!
//! Current: 72 tests (58 passing, 14 ignored pending feature implementation)
//! Target: 72 tests (100% parity with TypeScript)
//!
//! ## Running Tests
//!
//! ```bash
//! # Run all chat tests
//! cargo test --package ai-sdk-openai --test chat
//!
//! # Run specific test module
//! cargo test --package ai-sdk-openai --test chat streaming
//!
//! # Run with output
//! cargo test --package ai-sdk-openai --test chat -- --nocapture
//! ```

mod advanced_features_test;
mod basic_test;
mod extension_settings_test;
mod model_specific_test;
mod response_format_test;
mod settings_test;
mod streaming_test;
mod tool_calling_test;
