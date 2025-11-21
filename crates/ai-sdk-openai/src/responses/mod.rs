//! OpenAI Responses API implementation.
//!
//! This module contains the implementation of the OpenAI Responses API,
//! including type definitions and the language model implementation.

mod api_types;
mod api_types_test;
mod model;
mod options;

pub use model::OpenAIResponsesLanguageModel;

/// Maximum number of top logprobs (0-20)
pub const TOP_LOGPROBS_MAX: u8 = 20;
