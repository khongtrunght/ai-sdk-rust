#![allow(missing_docs)]
#![allow(dead_code)]
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Request to the OpenAI Responses API
#[derive(Debug, Serialize)]
pub struct ResponsesRequest {
    pub model: String,
    pub input: Vec<ResponsesInputItem>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    // Provider-specific options
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tool_calls: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_retention: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

/// Input item for the Responses API
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum ResponsesInputItem {
    Message(ResponsesMessage),
    FunctionCall(ResponsesFunctionCall),
    FunctionCallOutput(ResponsesFunctionCallOutput),
    ComputerCall(ResponsesComputerCall),
    LocalShellCall(ResponsesLocalShellCall),
    LocalShellCallOutput(ResponsesLocalShellCallOutput),
    Reasoning(ResponsesReasoning),
    ItemReference(ResponsesItemReference),
}

/// Message in the Responses API
#[derive(Debug, Serialize)]
pub struct ResponsesMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<ResponsesContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

/// Content can be a string or array of content parts
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum ResponsesContent {
    Text(String),
    Parts(Vec<ResponsesContentPart>),
}

/// Content part for multimodal messages
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesContentPart {
    InputText {
        text: String,
    },
    InputImage {
        #[serde(skip_serializing_if = "Option::is_none")]
        image_url: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
    },
    InputFile {
        #[serde(skip_serializing_if = "Option::is_none")]
        file_url: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_data: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
    },
    OutputText {
        text: String,
    },
}

#[derive(Debug, Serialize)]
pub struct ResponsesFunctionCall {
    pub r#type: String, // "function_call"
    pub call_id: String,
    pub name: String,
    pub arguments: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ResponsesFunctionCallOutput {
    pub r#type: String, // "function_call_output"
    pub call_id: String,
    pub output: ResponsesFunctionCallOutputContent,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum ResponsesFunctionCallOutputContent {
    Text(String),
    Parts(Vec<ResponsesContentPart>),
}

#[derive(Debug, Serialize)]
pub struct ResponsesComputerCall {
    pub r#type: String, // "computer_call"
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ResponsesLocalShellCall {
    pub r#type: String, // "local_shell_call"
    pub id: String,
    pub call_id: String,
    pub action: ResponsesLocalShellAction,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesLocalShellAction {
    Exec {
        command: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timeout_ms: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        user: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        working_directory: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        env: Option<HashMap<String, String>>,
    },
}

#[derive(Debug, Serialize)]
pub struct ResponsesLocalShellCallOutput {
    pub r#type: String, // "local_shell_call_output"
    pub call_id: String,
    pub output: String,
}

#[derive(Debug, Serialize)]
pub struct ResponsesReasoning {
    pub r#type: String, // "reasoning"
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encrypted_content: Option<String>,
    pub summary: Vec<ResponsesReasoningSummary>,
}

/// Reference to a previous item
#[derive(Debug, Serialize)]
pub struct ResponsesItemReference {
    pub r#type: String, // "item_reference"
    pub id: String,
}

/// Response from the OpenAI Responses API
#[derive(Debug, Deserialize)]
pub struct ResponsesResponse {
    pub id: String,

    #[serde(default)]
    pub created_at: Option<i64>,

    pub model: String,

    #[serde(default)]
    pub output: Option<Vec<ResponsesOutputItem>>,

    #[serde(default)]
    pub error: Option<ResponsesError>,

    #[serde(default)]
    pub usage: Option<ResponsesUsage>,

    #[serde(default)]
    pub incomplete_details: Option<ResponsesIncompleteDetails>,

    #[serde(default)]
    pub service_tier: Option<String>,
}

/// Output item from the response
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesOutputItem {
    Message {
        id: String,
        role: String,
        content: Vec<ResponsesOutputContentPart>,
    },
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
    },
    Reasoning {
        id: String,
        #[serde(default)]
        encrypted_content: Option<String>,
        #[serde(default)]
        summary: Vec<ResponsesReasoningSummary>,
    },
    WebSearchCall {
        id: String,
        status: String,
        action: ResponsesWebSearchAction,
    },
    FileSearchCall {
        id: String,
        queries: Vec<String>,
        #[serde(default)]
        results: Option<Vec<ResponsesFileSearchResult>>,
    },
    ComputerCall {
        id: String,
        status: String,
    },
    ImageGenerationCall {
        id: String,
        result: String,
    },
    CodeInterpreterCall {
        id: String,
        code: Option<String>,
        container_id: String,
        #[serde(default)]
        outputs: Option<Vec<ResponsesCodeInterpreterOutput>>,
    },
    LocalShellCall {
        id: String,
        call_id: String,
        action: ResponsesLocalShellAction,
    },
    McpCall {
        id: String,
        status: String,
        arguments: String,
        name: String,
        server_label: String,
        #[serde(default)]
        output: Option<String>,
        #[serde(default)]
        error: Option<ResponsesMcpError>,
    },
    McpListTools {
        id: String,
        server_label: String,
        tools: Vec<ResponsesMcpTool>,
        #[serde(default)]
        error: Option<ResponsesMcpError>,
    },
    McpApprovalRequest {
        id: String,
        server_label: String,
        name: String,
        arguments: String,
        approval_request_id: String,
    },
}

/// Content part in the output
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesOutputContentPart {
    OutputText {
        text: String,
        #[serde(default)]
        logprobs: Option<Vec<ResponsesLogprob>>,
        #[serde(default)]
        annotations: Vec<ResponsesAnnotation>,
    },
}

#[derive(Debug, Deserialize)]
pub struct ResponsesLogprob {
    pub token: String,
    pub logprob: f64,
    pub top_logprobs: Vec<ResponsesTopLogprob>,
}

#[derive(Debug, Deserialize)]
pub struct ResponsesTopLogprob {
    pub token: String,
    pub logprob: f64,
}

/// Reasoning summary
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub struct ResponsesReasoningSummary {
    pub text: String,
}

/// Annotation (citation) in the output
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesAnnotation {
    UrlCitation {
        url: String,
        title: String,
        #[serde(default)]
        start_index: Option<usize>,
        #[serde(default)]
        end_index: Option<usize>,
    },
    FileCitation {
        file_id: String,
        #[serde(default)]
        filename: Option<String>,
        #[serde(default)]
        index: Option<usize>,
        #[serde(default)]
        start_index: Option<usize>,
        #[serde(default)]
        end_index: Option<usize>,
        #[serde(default)]
        quote: Option<String>,
    },
    ContainerFileCitation {
        container_id: String,
        file_id: String,
        #[serde(default)]
        filename: Option<String>,
        #[serde(default)]
        start_index: Option<usize>,
        #[serde(default)]
        end_index: Option<usize>,
        #[serde(default)]
        index: Option<usize>,
    },
    FilePath {
        file_id: String,
        #[serde(default)]
        index: Option<usize>,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesWebSearchAction {
    Search {
        #[serde(default)]
        query: Option<String>,
        #[serde(default)]
        sources: Option<Vec<ResponsesWebSearchSource>>,
    },
    OpenPage {
        url: String,
    },
    Find {
        url: String,
        pattern: String,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesWebSearchSource {
    Url { url: String },
    Api { name: String },
}

#[derive(Debug, Deserialize)]
pub struct ResponsesFileSearchResult {
    pub attributes: HashMap<String, Value>,
    pub file_id: String,
    pub filename: String,
    pub score: f64,
    pub text: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesCodeInterpreterOutput {
    Logs { logs: String },
    Image { url: String },
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ResponsesMcpError {
    String(String),
    Object {
        #[serde(default)]
        r#type: Option<String>,
        #[serde(default)]
        code: Option<Value>, // number or string
        #[serde(default)]
        message: Option<String>,
    },
}

#[derive(Debug, Deserialize)]
pub struct ResponsesMcpTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub input_schema: Value,
    #[serde(default)]
    pub annotations: Option<HashMap<String, Value>>,
}

/// Error in the response
#[derive(Debug, Deserialize)]
pub struct ResponsesError {
    pub message: String,
    pub r#type: String,
    pub code: String,
    #[serde(default)]
    pub param: Option<String>,
}

/// Usage information
#[derive(Debug, Deserialize)]
pub struct ResponsesUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,

    #[serde(default)]
    pub input_tokens_details: Option<ResponsesInputTokensDetails>,

    #[serde(default)]
    pub output_tokens_details: Option<ResponsesOutputTokensDetails>,
}

#[derive(Debug, Deserialize)]
pub struct ResponsesInputTokensDetails {
    #[serde(default)]
    pub cached_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct ResponsesOutputTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: Option<u32>,
}

/// Incomplete details
#[derive(Debug, Deserialize)]
pub struct ResponsesIncompleteDetails {
    pub reason: String,
}

/// Streaming chunk from the Responses API
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesChunk {
    #[serde(rename = "response.created")]
    ResponseCreated { response: ResponseCreatedData },

    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        item_id: String,
        delta: String,
        #[serde(default)]
        logprobs: Option<Vec<ResponsesLogprob>>,
    },

    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        output_index: usize,
        item: OutputItemData,
    },

    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        output_index: usize,
        item: OutputItemData,
    },

    #[serde(rename = "response.completed")]
    ResponseCompleted { response: ResponseCompletedData },

    #[serde(rename = "response.incomplete")]
    ResponseIncomplete { response: ResponseCompletedData },

    #[serde(rename = "error")]
    Error {
        sequence_number: u32,
        error: ResponsesError,
    },

    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        item_id: String,
        output_index: usize,
        delta: String,
    },

    #[serde(rename = "response.image_generation_call.partial_image")]
    ImageGenerationCallPartialImage {
        item_id: String,
        output_index: usize,
        partial_image_b64: String,
    },

    #[serde(rename = "response.code_interpreter_call_code.delta")]
    CodeInterpreterCallCodeDelta {
        item_id: String,
        output_index: usize,
        delta: String,
    },

    #[serde(rename = "response.code_interpreter_call_code.done")]
    CodeInterpreterCallCodeDone {
        item_id: String,
        output_index: usize,
        code: String,
    },

    #[serde(rename = "response.output_text.annotation.added")]
    OutputTextAnnotationAdded { annotation: ResponsesAnnotation },

    #[serde(rename = "response.reasoning_summary_part.added")]
    ReasoningSummaryPartAdded {
        item_id: String,
        summary_index: usize,
    },

    #[serde(rename = "response.reasoning_summary_text.delta")]
    ReasoningSummaryTextDelta {
        item_id: String,
        summary_index: usize,
        delta: String,
    },

    #[serde(rename = "response.reasoning_summary_part.done")]
    ReasoningSummaryPartDone {
        item_id: String,
        summary_index: usize,
    },
}

#[derive(Debug, Deserialize)]
pub struct ResponseCreatedData {
    pub id: String,
    pub created_at: i64,
    pub model: String,
    #[serde(default)]
    pub service_tier: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputItemData {
    Message {
        id: String,
    },
    Reasoning {
        id: String,
        #[serde(default)]
        encrypted_content: Option<String>,
    },
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        #[serde(default)]
        arguments: Option<String>,
        #[serde(default)]
        status: Option<String>,
    },
    WebSearchCall {
        id: String,
        #[serde(default)]
        status: Option<String>,
        #[serde(default)]
        action: Option<ResponsesWebSearchAction>,
    },
    ComputerCall {
        id: String,
        #[serde(default)]
        status: Option<String>,
    },
    FileSearchCall {
        id: String,
        #[serde(default)]
        queries: Option<Vec<String>>,
        #[serde(default)]
        results: Option<Vec<ResponsesFileSearchResult>>,
    },
    ImageGenerationCall {
        id: String,
        #[serde(default)]
        result: Option<String>,
    },
    CodeInterpreterCall {
        id: String,
        container_id: String,
        #[serde(default)]
        code: Option<String>,
        #[serde(default)]
        outputs: Option<Vec<ResponsesCodeInterpreterOutput>>,
        #[serde(default)]
        status: Option<String>,
    },
    McpCall {
        id: String,
        #[serde(default)]
        status: Option<String>,
    },
    McpListTools {
        id: String,
    },
    McpApprovalRequest {
        id: String,
    },
    LocalShellCall {
        id: String,
        call_id: String,
        action: ResponsesLocalShellAction,
    },
}

#[derive(Debug, Deserialize)]
pub struct ResponseCompletedData {
    pub usage: ResponsesUsage,
    #[serde(default)]
    pub incomplete_details: Option<ResponsesIncompleteDetails>,
    #[serde(default)]
    pub service_tier: Option<String>,
}
