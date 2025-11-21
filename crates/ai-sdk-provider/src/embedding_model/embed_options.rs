use crate::{SharedHeaders, SharedProviderOptions};

/// Options for calling the embedding model
#[derive(Debug, Clone)]
pub struct EmbedOptions<VALUE>
where
    VALUE: Send + Sync,
{
    /// List of values to embed
    pub values: Vec<VALUE>,

    /// Additional provider-specific options. They are passed through
    /// to the provider from the AI SDK and enable provider-specific
    /// functionality that can be fully encapsulated in the provider.
    pub provider_options: Option<SharedProviderOptions>,

    /// Additional HTTP headers to be sent with the request.
    /// Only applicable for HTTP-based providers.
    pub headers: Option<SharedHeaders>,
}
