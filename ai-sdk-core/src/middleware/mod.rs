//! Middleware system for language models
//!
//! Middlewares allow customization of language model behavior without modifying
//! the core implementation. They can:
//!
//! - Transform parameters before model calls
//! - Wrap generate/stream operations
//! - Override model metadata
//! - Compose in an "onion" pattern (first transforms input first, wraps output last)
//!
//! # Example
//!
//! ```no_run
//! use ai_sdk_core::middleware::{wrap_language_model, DefaultSettingsMiddleware};
//! use ai_sdk_provider::language_model::CallOptions;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! # let model = todo!();
//! let wrapped = wrap_language_model(
//!     model,
//!     vec![
//!         Box::new(DefaultSettingsMiddleware::new(CallOptions {
//!             temperature: Some(0.7),
//!             ..Default::default()
//!         })),
//!     ],
//! );
//! # Ok(())
//! # }
//! ```

mod default_settings_middleware;
mod language_model_middleware;
mod simulate_streaming_middleware;
mod wrap_language_model;

pub use default_settings_middleware::DefaultSettingsMiddleware;
pub use language_model_middleware::{CallType, GenerateFn, LanguageModelMiddleware, StreamFn};
pub use simulate_streaming_middleware::SimulateStreamingMiddleware;
pub use wrap_language_model::wrap_language_model;
