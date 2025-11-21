use std::{collections::HashMap, fmt, sync::Arc};

// Define aliases to make the struct readable
pub type UrlFactory = dyn Fn(OpenAIUrlOptions) -> String + Send + Sync;
pub type HeaderFactory = dyn Fn() -> HashMap<String, String> + Send + Sync;
pub type IdGenerator = dyn Fn() -> String + Send + Sync;

/// Configuration for OpenAI provider.
#[derive(Clone)]
pub struct OpenAIConfig {
    /// The provider name (e.g., "openai").
    pub provider: String,
    /// Factory function to generate the API URL.
    pub url: Arc<UrlFactory>,
    /// Factory function to generate headers.
    pub headers: Arc<HeaderFactory>,
    /// Optional factory function to generate IDs.
    pub generate_id: Option<Arc<IdGenerator>>,
    /// Optional list of file ID prefixes.
    pub file_id_prefixes: Option<Vec<String>>,
}

impl OpenAIConfig {
    /// Creates a new `OpenAIConfig`.
    ///
    /// # Arguments
    /// * `provider` - The provider name.
    /// * `url` - Function to generate the API URL.
    /// * `headers` - Function to generate headers.
    pub fn new(
        provider: impl Into<String>,
        url: impl Fn(OpenAIUrlOptions) -> String + Send + Sync + 'static,
        headers: impl Fn() -> HashMap<String, String> + Send + Sync + 'static,
    ) -> Self {
        Self {
            provider: provider.into(),
            url: Arc::new(url),
            headers: Arc::new(headers),
            generate_id: None,
            file_id_prefixes: None,
        }
    }

    /// Creates a new `OpenAIConfig` from an API key.
    ///
    /// This is a convenience method for creating a config with default settings.
    pub fn from_api_key(api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();
        Self::new(
            "openai",
            |opts| format!("https://api.openai.com/v1{}", opts.path),
            move || {
                let mut headers = HashMap::new();
                headers.insert("Authorization".to_string(), format!("Bearer {}", api_key));
                headers
            },
        )
    }
}

// Manual Debug implementation is required because Closures don't implement Debug
impl fmt::Debug for OpenAIConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAIConfig")
            .field("provider", &self.provider)
            .field("url", &"<closure>")
            .field("headers", &"<closure>")
            .field(
                "generate_id",
                &self.generate_id.as_ref().map(|_| "<closure>"),
            )
            .field("file_id_prefixes", &self.file_id_prefixes)
            .finish()
    }
}

/// URL builder input, equivalent to:
/// `{ modelId: string; path: string }`
#[derive(Debug, Clone)]
pub struct OpenAIUrlOptions {
    /// The model ID.
    pub model_id: String,
    /// The API path (e.g., "/chat/completions").
    pub path: String,
}
