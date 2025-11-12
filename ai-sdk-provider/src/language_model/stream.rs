use super::*;
use crate::SharedProviderMetadata;
use serde::{Deserialize, Serialize};

/// A part of a streaming language model response
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum StreamPart {
    // Text
    /// Start of a text content part
    TextStart {
        /// Unique identifier for this text part
        id: String,
        /// Provider-specific metadata
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedProviderMetadata>,
    },
    /// Incremental text content
    TextDelta {
        /// Identifier of the text part
        id: String,
        /// Incremental text content
        delta: String,
        /// Provider-specific metadata
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedProviderMetadata>,
    },
    /// End of a text content part
    TextEnd {
        /// Identifier of the text part
        id: String,
        /// Provider-specific metadata
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedProviderMetadata>,
    },

    // Reasoning
    /// Start of a reasoning content part
    ReasoningStart {
        /// Unique identifier for this reasoning part
        id: String,
        /// Provider-specific metadata
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedProviderMetadata>,
    },
    /// Incremental reasoning content
    ReasoningDelta {
        /// Identifier of the reasoning part
        id: String,
        /// Incremental reasoning content
        delta: String,
        /// Provider-specific metadata
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedProviderMetadata>,
    },
    /// End of a reasoning content part
    ReasoningEnd {
        /// Identifier of the reasoning part
        id: String,
        /// Provider-specific metadata
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedProviderMetadata>,
    },

    // Tool input streaming
    /// Start of a tool input part
    #[serde(rename_all = "camelCase")]
    ToolInputStart {
        /// Unique identifier for this tool input
        id: String,
        /// Name of the tool being called
        tool_name: String,
        /// Provider-specific metadata
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedProviderMetadata>,
        /// Whether the tool was executed by the provider
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_executed: Option<bool>,
        /// Whether the tool call is dynamic
        #[serde(skip_serializing_if = "Option::is_none")]
        dynamic: Option<bool>,
        /// Title of the tool call
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
    },
    /// Incremental tool input content
    ToolInputDelta {
        /// Identifier of the tool input part
        id: String,
        /// Incremental tool input content
        delta: String,
        /// Provider-specific metadata
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedProviderMetadata>,
    },
    /// End of a tool input part
    ToolInputEnd {
        /// Identifier of the tool input part
        id: String,
        /// Provider-specific metadata
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedProviderMetadata>,
    },

    // Complete tool call and result
    /// Complete tool call part
    ToolCall(ToolCallPart),
    /// Complete tool result part
    ToolResult(ToolResultPart),

    // Files and sources
    /// File content part
    File(FilePart),
    /// Source citation part
    Source(SourcePart),

    // Stream lifecycle
    /// Start of the stream
    StreamStart {
        /// Warnings about the call
        warnings: Vec<CallWarning>,
    },
    /// Response metadata from the provider
    ResponseMetadata {
        /// Response metadata
        #[serde(flatten)]
        metadata: ResponseMetadata,
    },
    /// Stream finished
    #[serde(rename_all = "camelCase")]
    Finish {
        /// Token usage information
        usage: Usage,
        /// Reason why generation finished
        finish_reason: FinishReason,
        /// Provider-specific metadata
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedProviderMetadata>,
    },

    // Raw and error
    /// Raw chunk from the provider
    #[serde(rename_all = "camelCase")]
    Raw {
        /// Raw value from the provider
        raw_value: serde_json::Value,
    },
    /// Error during streaming
    Error {
        /// Error message
        error: String, // Simplified for now
    },
}

/// Warning about a language model call
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CallWarning {
    /// Warning message
    // Simplified - expand based on language-model-v3-call-warning.ts
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_part_text_delta() {
        let part = StreamPart::TextDelta {
            id: "0".into(),
            delta: "Hello".into(),
            provider_metadata: None,
        };
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(json["type"], "text-delta");
        assert_eq!(json["delta"], "Hello");
    }

    #[test]
    fn test_stream_part_finish() {
        let part = StreamPart::Finish {
            usage: Usage::default(),
            finish_reason: FinishReason::Stop,
            provider_metadata: None,
        };
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(json["type"], "finish");
        assert_eq!(json["finishReason"], "stop");
    }

    #[test]
    fn test_stream_part_tool_input_start() {
        let part = StreamPart::ToolInputStart {
            id: "tool-1".into(),
            tool_name: "search".into(),
            provider_metadata: None,
            provider_executed: None,
            dynamic: Some(true),
            title: None,
        };
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(json["type"], "tool-input-start");
        assert_eq!(json["toolName"], "search");
        assert_eq!(json["dynamic"], true);
    }
}
