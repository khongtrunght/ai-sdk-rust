use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<ChatMessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Deserialize)]
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

#[derive(Debug, Deserialize)]
pub struct Choice {
    #[allow(dead_code)]
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct UsageInfo {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
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
