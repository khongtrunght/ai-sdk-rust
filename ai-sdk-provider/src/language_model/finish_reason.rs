use serde::{Deserialize, Serialize};

/// Reason why a language model finished generating a response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FinishReason {
    /// Model generated stop sequence
    Stop,
    /// Model generated maximum number of tokens
    Length,
    /// Content filter violation stopped the model
    ContentFilter,
    /// Model triggered tool calls
    ToolCalls,
    /// Model stopped because of an error
    Error,
    /// Model stopped for other reasons
    Other,
    /// The model has not transmitted a finish reason
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finish_reason_serialization() {
        assert_eq!(
            serde_json::to_string(&FinishReason::ToolCalls).unwrap(),
            r#""tool-calls""#
        );
        assert_eq!(
            serde_json::to_string(&FinishReason::Stop).unwrap(),
            r#""stop""#
        );
        assert_eq!(
            serde_json::to_string(&FinishReason::ContentFilter).unwrap(),
            r#""content-filter""#
        );
    }

    #[test]
    fn test_finish_reason_deserialization() {
        let reason: FinishReason = serde_json::from_str(r#""tool-calls""#).unwrap();
        assert_eq!(reason, FinishReason::ToolCalls);
    }
}
