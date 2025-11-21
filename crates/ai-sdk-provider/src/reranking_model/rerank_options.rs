use crate::json_value::JsonObject;
use crate::shared::{SharedHeaders, SharedProviderOptions};

/// Documents to rerank - either a list of text strings or JSON objects
#[derive(Debug, Clone)]
pub enum Documents {
    /// A list of text documents
    Text {
        /// Text document strings
        values: Vec<String>,
    },
    /// A list of JSON object documents
    Object {
        /// JSON object documents
        values: Vec<JsonObject>,
    },
}

/// Options for reranking documents
#[derive(Debug, Clone)]
pub struct RerankOptions {
    /// Documents to rerank. Either a list of texts or a list of JSON objects.
    pub documents: Documents,

    /// The query string to rerank the documents against
    pub query: String,

    /// Optional limit to return only the top N documents
    pub top_n: Option<usize>,

    /// Abort signal for cancelling the operation
    pub abort_signal: Option<tokio::sync::watch::Receiver<bool>>,

    /// Additional provider-specific options
    pub provider_options: Option<SharedProviderOptions>,

    /// Additional HTTP headers to be sent with the request
    pub headers: Option<SharedHeaders>,
}
