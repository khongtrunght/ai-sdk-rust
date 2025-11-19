use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;

#[derive(Debug, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<OpenAIToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<OpenAIResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,

    // OpenAI-specific options
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction: Option<JsonValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
}

/// Options for streaming responses
#[derive(Debug, Serialize, Clone)]
pub struct StreamOptions {
    pub include_usage: bool,
}

/// Response format for OpenAI API
#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAIResponseFormat {
    /// Text response (default)
    Text,
    /// JSON object response (unvalidated)
    JsonObject,
    /// JSON schema response (structured)
    JsonSchema { json_schema: OpenAIJsonSchema },
}

/// JSON schema for structured outputs
#[derive(Debug, Serialize, Clone)]
pub struct OpenAIJsonSchema {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub schema: JsonValue,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// Content for a chat message - can be string or array of content parts
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum ChatMessageContent {
    /// Simple text content
    Text(String),
    /// Array of content parts (for multi-modal messages)
    Parts(Vec<crate::multimodal::OpenAIContentPart>),
}

/// URL citation annotation from OpenAI API
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct UrlCitationAnnotation {
    #[serde(rename = "type")]
    pub annotation_type: String, // "url_citation"
    pub start_index: u32,
    pub end_index: u32,
    pub url: String,
    pub title: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<ChatMessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub annotations: Option<Vec<UrlCitationAnnotation>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionResponse {
    #[allow(dead_code)]
    pub id: String,
    #[allow(dead_code)]
    pub object: String,
    #[allow(dead_code)]
    pub created: u64,
    #[allow(dead_code)]
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Option<UsageInfo>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Choice {
    #[allow(dead_code)]
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
    pub logprobs: Option<serde_json::Value>,
}

/// Detailed completion token information
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct CompletionTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: Option<u32>,
    #[serde(default)]
    pub accepted_prediction_tokens: Option<u32>,
    #[serde(default)]
    pub rejected_prediction_tokens: Option<u32>,
}

/// Detailed prompt token information
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: Option<u32>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct UsageInfo {
    pub prompt_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u32>,
    pub total_tokens: u32,
    #[serde(default)]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

// Streaming response types
#[derive(Debug, Deserialize)]
pub struct ChatCompletionChunk {
    #[allow(dead_code)]
    pub id: String,
    #[allow(dead_code)]
    pub object: String,
    #[allow(dead_code)]
    pub created: u64,
    #[allow(dead_code)]
    pub model: String,
    pub choices: Vec<StreamChoice>,
    pub usage: Option<UsageInfo>,
}

#[derive(Debug, Deserialize)]
pub struct StreamChoice {
    #[allow(dead_code)]
    pub index: u32,
    pub delta: StreamDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct StreamDelta {
    pub content: Option<String>,
    #[allow(dead_code)]
    pub role: Option<String>,
    pub tool_calls: Option<Vec<OpenAIToolCallDelta>>,
    #[serde(default)]
    pub annotations: Option<Vec<UrlCitationAnnotation>>,
}

// Tool-related types

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAITool {
    pub r#type: String,
    pub function: OpenAIFunction,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAIFunction {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: JsonValue,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum OpenAIToolChoice {
    String(String), // "auto", "none", "required"
    Specific {
        r#type: String,
        function: OpenAIFunctionName,
    },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAIFunctionName {
    pub name: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAIToolCall {
    pub id: String,
    pub r#type: String,
    pub function: OpenAIFunctionCall,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAIFunctionCall {
    pub name: String,
    pub arguments: String, // JSON string
}

#[derive(Debug, Deserialize)]
pub struct OpenAIToolCallDelta {
    pub index: u32,
    pub id: Option<String>,
    #[allow(dead_code)]
    pub r#type: Option<String>,
    pub function: OpenAIFunctionCallDelta,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIFunctionCallDelta {
    pub name: Option<String>,
    pub arguments: Option<String>,
}
