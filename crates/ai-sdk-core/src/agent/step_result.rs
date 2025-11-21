use ai_sdk_provider::language_model::{
    CallWarning, Content, FinishReason, RequestInfo, ResponseInfo, ToolCallPart, ToolResultPart,
    Usage,
};
use ai_sdk_provider::SharedProviderMetadata;

/// Result from a single agent step
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Content generated in this step
    pub content: Vec<Content>,

    /// Tool calls made in this step
    pub tool_calls: Option<Vec<ToolCallPart>>,

    /// Tool results from this step
    pub tool_results: Option<Vec<ToolResultPart>>,

    /// Generated text (computed from content)
    pub text: String,

    /// Reasoning text if any
    pub reasoning_text: Option<String>,

    /// Finish reason for this step
    pub finish_reason: FinishReason,

    /// Token usage for this step
    pub usage: Usage,

    /// Warnings from provider
    pub warnings: Vec<CallWarning>,

    /// Request metadata
    pub request: Option<RequestInfo>,

    /// Response metadata
    pub response: Option<ResponseInfo>,

    /// Provider-specific metadata
    pub provider_metadata: Option<SharedProviderMetadata>,
}

impl StepResult {
    /// Extract text content from a content array
    pub fn extract_text(content: &[Content]) -> String {
        content
            .iter()
            .filter_map(|c| match c {
                Content::Text(text_part) => Some(text_part.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Extract reasoning content from a content array
    pub fn extract_reasoning(content: &[Content]) -> Option<String> {
        let reasoning_parts: Vec<String> = content
            .iter()
            .filter_map(|c| match c {
                Content::Reasoning(reasoning_part) => Some(reasoning_part.reasoning.clone()),
                _ => None,
            })
            .collect();

        if reasoning_parts.is_empty() {
            None
        } else {
            Some(reasoning_parts.join(""))
        }
    }

    /// Extract tool calls from content array
    pub fn extract_tool_calls(content: &[Content]) -> Option<Vec<ToolCallPart>> {
        let tool_calls: Vec<ToolCallPart> = content
            .iter()
            .filter_map(|c| match c {
                Content::ToolCall(tc) => Some(tc.clone()),
                _ => None,
            })
            .collect();

        if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        }
    }

    /// Extract tool results from content array
    pub fn extract_tool_results(content: &[Content]) -> Option<Vec<ToolResultPart>> {
        let tool_results: Vec<ToolResultPart> = content
            .iter()
            .filter_map(|c| match c {
                Content::ToolResult(tr) => Some(tr.clone()),
                _ => None,
            })
            .collect();

        if tool_results.is_empty() {
            None
        } else {
            Some(tool_results)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_sdk_provider::language_model::TextPart;

    #[test]
    fn test_extract_text() {
        let content = vec![
            Content::Text(TextPart {
                text: "Hello".to_string(),
                provider_metadata: None,
            }),
            Content::Text(TextPart {
                text: " world".to_string(),
                provider_metadata: None,
            }),
        ];

        assert_eq!(StepResult::extract_text(&content), "Hello world");
    }

    #[test]
    fn test_extract_text_empty() {
        let content = vec![];
        assert_eq!(StepResult::extract_text(&content), "");
    }

    #[test]
    fn test_extract_tool_calls() {
        let content = vec![Content::ToolCall(ToolCallPart {
            tool_call_id: "call_1".to_string(),
            tool_name: "weather".to_string(),
            input: "{}".to_string(),
            provider_executed: None,
            dynamic: None,
            provider_metadata: None,
        })];

        let tool_calls = StepResult::extract_tool_calls(&content);
        assert!(tool_calls.is_some());
        assert_eq!(tool_calls.unwrap().len(), 1);
    }

    #[test]
    fn test_extract_tool_calls_none() {
        let content = vec![Content::Text(TextPart {
            text: "Hello".to_string(),
            provider_metadata: None,
        })];

        assert!(StepResult::extract_tool_calls(&content).is_none());
    }
}
