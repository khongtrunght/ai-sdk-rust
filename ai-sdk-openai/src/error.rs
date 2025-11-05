use thiserror::Error;

#[derive(Error, Debug)]
pub enum OpenAIError {
    #[error("API error: {message}")]
    ApiError {
        message: String,
        status_code: Option<u16>,
    },

    #[error("Authentication failed")]
    AuthenticationError,

    #[error("Rate limit exceeded")]
    RateLimitError { retry_after: Option<u64> },

    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Invalid response")]
    InvalidResponse,
}
