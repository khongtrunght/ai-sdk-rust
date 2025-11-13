//! Generate structured objects from language models.
//!
//! This module provides functionality for generating validated, schema-based outputs
//! from language models, with support for:
//! - Single structured objects
//! - Arrays of elements
//! - Enum values
//! - Streaming with partial results
//!
//! # Examples
//!
//! ## Basic Object Generation
//!
//! ```rust,ignore
//! use ai_sdk_core::generate_object::{generate_object, ObjectOutputStrategy};
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Serialize, Deserialize)]
//! struct Recipe {
//!     name: String,
//!     ingredients: Vec<String>,
//!     steps: Vec<String>,
//! }
//!
//! let result = generate_object::<ObjectOutputStrategy<Recipe>>()
//!     .model(model)
//!     .prompt("Generate a recipe for chocolate chip cookies")
//!     .output_strategy(ObjectOutputStrategy::new(recipe_schema))
//!     .execute()
//!     .await?;
//!
//! println!("Recipe: {:?}", result.object);
//! ```
//!
//! ## Streaming Object Generation
//!
//! ```rust,ignore
//! use ai_sdk_core::generate_object::{stream_object, ObjectOutputStrategy, ObjectStreamPart};
//! use tokio_stream::StreamExt;
//!
//! let mut result = stream_object::<ObjectOutputStrategy<Recipe>>()
//!     .model(model)
//!     .prompt("Generate a recipe for chocolate chip cookies")
//!     .output_strategy(ObjectOutputStrategy::new(recipe_schema))
//!     .execute()
//!     .await?;
//!
//! // Process streaming updates
//! while let Some(part) = result.partial_object_stream.next().await {
//!     match part {
//!         ObjectStreamPart::Object { object } => {
//!             println!("Partial recipe: {:?}", object);
//!         }
//!         ObjectStreamPart::Finish { .. } => break,
//!         _ => {}
//!     }
//! }
//! ```

mod builder;
mod output_strategy;
mod stream_object;

pub use builder::{
    generate_object, GenerateObjectBuilder, GenerateObjectError, GenerateObjectResult,
};
pub use output_strategy::{
    ArrayOutputStrategy, EnumOutputStrategy, NoSchemaOutputStrategy, ObjectOutputStrategy,
    OutputStrategy, OutputType, PartialValidation, ValidationContext, ValidationError,
    ValidationResult,
};
pub use stream_object::{
    stream_object, ObjectStreamPart, StreamObjectBuilder, StreamObjectError, StreamObjectResult,
};
