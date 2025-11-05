//! Language Model v3 specification types

pub mod call_options;
pub mod content;
pub mod data_content;
pub mod finish_reason;
pub mod prompt;
pub mod response_metadata;
pub mod stream;
pub mod tools;
pub mod trait_def;
pub mod usage;

pub use call_options::{CallOptions, ResponseFormat};
pub use content::{
    Content, FilePart, ReasoningPart, SourcePart, SourceType, TextPart, ToolCallPart,
    ToolResultPart,
};
pub use data_content::DataContent;
pub use finish_reason::FinishReason;
pub use prompt::{AssistantContentPart, Message, Prompt, UserContentPart};
pub use response_metadata::ResponseMetadata;
pub use stream::{CallWarning, StreamPart};
pub use tools::{FunctionTool, ProviderDefinedTool, Tool, ToolChoice};
pub use trait_def::{
    GenerateResponse, LanguageModel, RequestInfo, ResponseInfo, StreamError, StreamResponse,
};
pub use usage::Usage;
