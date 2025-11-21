//! Embedding Model v3 specification types

/// Embedding generation options
pub mod embed_options;
/// Embedding generation response types
pub mod embed_response;
/// Embedding model trait definition
pub mod trait_def;

pub use embed_options::EmbedOptions;
pub use embed_response::{EmbedResponse, Embedding, EmbeddingUsage, ResponseInfo};
pub use trait_def::EmbeddingModel;
