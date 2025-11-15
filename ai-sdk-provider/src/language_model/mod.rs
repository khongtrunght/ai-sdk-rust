//! Language Model v3 specification types

/// Call options for language model generation
pub mod call_options;
/// Content types for language model messages
pub mod content;
/// Data content types for binary data in messages
pub mod data_content;
/// Finish reason enumeration for generation completion
pub mod finish_reason;
/// Prompt and message types for language model input
pub mod prompt;
/// Response metadata for debugging and telemetry
pub mod response_metadata;
/// Streaming types for language model generation
pub mod stream;
/// Tool result output types for structured tool responses
pub mod tool_result_output;
/// Tool calling types for function calling
pub mod tools;
/// Language model trait definition
pub mod trait_def;
/// Token usage statistics
pub mod usage;

pub use call_options::{CallOptions, ResponseFormat};
pub use content::{
    Content, FilePart, ReasoningPart, SourcePart, SourceType, TextPart, ToolCallPart,
    ToolResultPart,
};
pub use data_content::DataContent;
pub use finish_reason::FinishReason;
pub use prompt::{AssistantContentPart, FileData, Message, Prompt, UserContentPart};
pub use response_metadata::ResponseMetadata;
pub use stream::{CallWarning, StreamPart};
pub use tool_result_output::{ContentPart, ToolResultOutput};
pub use tools::{FunctionTool, ProviderDefinedTool, Tool, ToolChoice};
pub use trait_def::{
    GenerateResponse, LanguageModel, RequestInfo, ResponseInfo, StreamError, StreamResponse,
};
pub use usage::Usage;
