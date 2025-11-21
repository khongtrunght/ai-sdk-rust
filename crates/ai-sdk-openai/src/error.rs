use thiserror::Error;

/// Errors that can occur when using the OpenAI provider.
#[derive(Error, Debug)]
pub enum OpenAIError {
    /// API error from OpenAI service.
    #[error("API error: {message}")]
    ApiError {
        /// Error message from the API.
        message: String,
        /// HTTP status code if available.
        status_code: Option<u16>,
    },

    /// Authentication failed with the provided API key.
    #[error("Authentication failed")]
    AuthenticationError,

    /// Rate limit exceeded for the API.
    #[error("Rate limit exceeded")]
    RateLimitError {
        /// Number of seconds to wait before retrying.
        retry_after: Option<u64>,
    },

    /// Network error occurred during request.
    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),

    /// JSON serialization/deserialization error.
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// Invalid response received from the API.
    #[error("Invalid response")]
    InvalidResponse,
}
