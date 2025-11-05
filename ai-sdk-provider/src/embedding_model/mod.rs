//! Embedding Model v3 specification types

pub mod embed_options;
pub mod embed_response;
pub mod trait_def;

pub use embed_options::EmbedOptions;
pub use embed_response::{EmbedResponse, Embedding, EmbeddingUsage, ResponseInfo};
pub use trait_def::EmbeddingModel;
