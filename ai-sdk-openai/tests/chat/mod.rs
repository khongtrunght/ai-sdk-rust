//! Integration tests for OpenAI Chat Model
//!
//! These tests verify the chat model implementation against fixture-based
//! mock responses, ensuring compatibility with the OpenAI API without
//! requiring real API calls.
//!
//! ## Test Organization
//!
//! Tests are organized by functionality:
//! - `basic_test.rs` - Basic generation, streaming, and response extraction
//! - `tool_calling_test.rs` - Function/tool calling scenarios
//! - `settings_test.rs` - Settings propagation and configuration (TODO)
//! - `streaming_test.rs` - Advanced streaming scenarios (TODO)
//!
//! ## Fixtures
//!
//! Fixtures are stored in `fixtures/` and organized by category.
//! See `fixtures/README.md` for details.
//!
//! ## Test Coverage
//!
//! Current: 6 tests (basic functionality and tool calling)
//! Target: 72 tests (100% parity with TypeScript)

mod basic_test;
mod tool_calling_test;
