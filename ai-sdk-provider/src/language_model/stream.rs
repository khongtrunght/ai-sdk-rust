use super::*;
use crate::SharedProviderMetadata;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum StreamPart {
    // Text
    TextStart {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedProviderMetadata>,
    },
    TextDelta {
        id: String,
        delta: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedProviderMetadata>,
    },
    TextEnd {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedProviderMetadata>,
    },

    // Reasoning
    ReasoningStart {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedProviderMetadata>,
    },
    ReasoningDelta {
        id: String,
        delta: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedProviderMetadata>,
    },
    ReasoningEnd {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedProviderMetadata>,
    },

    // Tool input streaming
    #[serde(rename_all = "camelCase")]
    ToolInputStart {
        id: String,
        tool_name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedProviderMetadata>,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_executed: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        dynamic: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
    },
    ToolInputDelta {
        id: String,
        delta: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedProviderMetadata>,
    },
    ToolInputEnd {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedProviderMetadata>,
    },

    // Complete tool call and result
    ToolCall(ToolCallPart),
    ToolResult(ToolResultPart),

    // Files and sources
    File(FilePart),
    Source(SourcePart),

    // Stream lifecycle
    StreamStart {
        warnings: Vec<CallWarning>,
    },
    ResponseMetadata {
        #[serde(flatten)]
        metadata: ResponseMetadata,
    },
    #[serde(rename_all = "camelCase")]
    Finish {
        usage: Usage,
        finish_reason: FinishReason,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedProviderMetadata>,
    },

    // Raw and error
    #[serde(rename_all = "camelCase")]
    Raw {
        raw_value: serde_json::Value,
    },
    Error {
        error: String, // Simplified for now
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CallWarning {
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
