use thiserror::Error;

/// Error that can occur during text generation
#[derive(Error, Debug)]
pub enum GenerateTextError {
    /// Missing model - must call .model() before execute()
    #[error("Missing model - must call .model() before execute()")]
    MissingModel,

    /// Missing prompt - must call .prompt() before execute()
    #[error("Missing prompt - must call .prompt() before execute()")]
    MissingPrompt,

    /// Model error
    #[error("Model error: {0}")]
    ModelError(#[from] Box<dyn std::error::Error + Send + Sync>),

    /// Tool execution error
    #[error("Tool execution error: {0}")]
    ToolError(#[from] ToolError),

    /// Maximum steps reached without completion
    #[error("Maximum steps reached without completion")]
    MaxStepsReached,

    /// Invalid parameters provided
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
}

/// Error that can occur during streaming text generation
#[derive(Error, Debug)]
pub enum StreamTextError {
    /// Missing model - must call .model() before execute()
    #[error("Missing model - must call .model() before execute()")]
    MissingModel,

    /// Missing prompt - must call .prompt() before execute()
    #[error("Missing prompt - must call .prompt() before execute()")]
    MissingPrompt,

    /// Model error
    #[error("Model error: {0}")]
    ModelError(String),

    /// Stream error
    #[error("Stream error: {0}")]
    StreamError(String),

    /// Tool execution error
    #[error("Tool execution error: {0}")]
    ToolError(#[from] ToolError),
}

/// Error that can occur during embedding
#[derive(Error, Debug)]
#[allow(dead_code)] // Will be used in Phase 1.3
pub enum EmbedError {
    /// Missing model - must call .model() before execute()
    #[error("Missing model - must call .model() before execute()")]
    MissingModel,

    /// Missing value - must call .value() before execute()
    #[error("Missing value - must call .value() before execute()")]
    MissingValue,

    /// Empty response from embedding model
    #[error("Empty response from embedding model")]
    EmptyResponse,

    /// Model error
    #[error("Model error: {0}")]
    ModelError(#[from] Box<dyn std::error::Error + Send + Sync>),
}

/// Error that can occur during tool execution
#[derive(Error, Debug)]
pub enum ToolError {
    /// Tool execution failed
    #[error("Tool execution failed: {0}")]
    ExecutionError(String),

    /// Tool not found
    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    /// Invalid tool input
    #[error("Invalid tool input: {0}")]
    InvalidInput(String),

    /// Tool execution denied
    #[error("Tool execution denied")]
    ExecutionDenied,
}

impl ToolError {
    /// Create an execution error
    pub fn execution(msg: impl Into<String>) -> Self {
        ToolError::ExecutionError(msg.into())
    }

    /// Create a not found error
    pub fn not_found(name: impl Into<String>) -> Self {
        ToolError::ToolNotFound(name.into())
    }

    /// Create an invalid input error
    pub fn invalid_input(msg: impl Into<String>) -> Self {
        ToolError::InvalidInput(msg.into())
    }
}
