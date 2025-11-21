use crate::shared::{SharedHeaders, SharedProviderMetadata, SharedWarning};
use serde::{Deserialize, Serialize};

/// A single ranked item in the reranking results
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankingItem {
    /// The index of the document in the original list of documents before reranking
    pub index: usize,

    /// The relevance score of the document after reranking
    #[serde(rename = "relevanceScore")]
    pub relevance_score: f64,
}

/// Optional response information for debugging purposes
#[derive(Debug, Clone)]
pub struct ResponseInfo {
    /// ID for the generated response, if the provider sends one
    pub id: Option<String>,

    /// Timestamp for the start of the generated response, if the provider sends one
    pub timestamp: Option<std::time::SystemTime>,

    /// The ID of the response model that was used to generate the response, if the provider sends one
    pub model_id: Option<String>,

    /// Response headers
    pub headers: Option<SharedHeaders>,

    /// Response body for debugging
    pub body: Option<serde_json::Value>,
}

/// Response from a reranking model operation
#[derive(Debug, Clone)]
pub struct RerankResponse {
    /// Ordered list of reranked documents (via index before reranking).
    /// The documents are sorted by the descending order of relevance scores.
    pub ranking: Vec<RankingItem>,

    /// Additional provider-specific metadata
    pub provider_metadata: Option<SharedProviderMetadata>,

    /// Warnings for the call, e.g. unsupported settings
    pub warnings: Option<Vec<SharedWarning>>,

    /// Optional response information for debugging purposes
    pub response: Option<ResponseInfo>,
}
