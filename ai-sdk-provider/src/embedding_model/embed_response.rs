use crate::SharedHeaders;
use crate::SharedProviderMetadata;
use serde::{Deserialize, Serialize};

/// An embedding is a vector, i.e. an array of numbers.
/// It is e.g. used to represent a text as a vector of word embeddings.
pub type Embedding = Vec<f32>;

/// Response from the embedding model
#[derive(Debug, Clone, PartialEq)]
pub struct EmbedResponse {
    /// Generated embeddings. They are in the same order as the input values.
    pub embeddings: Vec<Embedding>,

    /// Token usage. We only have input tokens for embeddings.
    pub usage: Option<EmbeddingUsage>,

    /// Additional provider-specific metadata. They are passed through
    /// from the provider to the AI SDK and enable provider-specific
    /// results that can be fully encapsulated in the provider.
    pub provider_metadata: Option<SharedProviderMetadata>,

    /// Optional response information for debugging purposes.
    pub response: Option<ResponseInfo>,
}

/// Token usage information for embeddings
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    /// Number of tokens used
    pub tokens: u32,
}

/// Response metadata for debugging
#[derive(Debug, Clone, PartialEq)]
pub struct ResponseInfo {
    /// Response headers
    pub headers: Option<SharedHeaders>,

    /// The response body (for debugging)
    pub body: Option<serde_json::Value>,
}
