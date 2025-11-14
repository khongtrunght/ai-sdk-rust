//! Parse partial JSON with automatic repair.
//!
//! This module provides utilities for parsing potentially incomplete JSON strings,
//! automatically attempting repair when direct parsing fails.

use super::fix_json::fix_json;
use serde_json::Value;

/// The state of a JSON parsing attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseState {
    /// Input was undefined/None
    UndefinedInput,
    /// Successfully parsed without repair
    SuccessfulParse,
    /// Successfully parsed after repair
    RepairedParse,
    /// Failed to parse even after repair
    FailedParse,
}

/// Result of attempting to parse partial JSON.
#[derive(Debug)]
pub struct ParseResult {
    /// The parsed value, if successful
    pub value: Option<Value>,
    /// The state of the parse attempt
    pub state: ParseState,
}

/// Attempts to parse a potentially incomplete JSON string.
///
/// This function tries to parse JSON in two stages:
/// 1. Direct parse - if the JSON is complete and valid
/// 2. Repair and parse - if direct parsing fails, attempts to repair and reparse
///
/// # Examples
///
/// ```
/// use ai_sdk_core::util::parse_partial_json;
///
/// // Parse complete JSON
/// let result = parse_partial_json(Some(r#"{"name":"Alice"}"#));
/// assert!(result.value.is_some());
///
/// // Parse incomplete JSON (will be repaired)
/// let result = parse_partial_json(Some(r#"{"name":"Alice""#));
/// assert!(result.value.is_some());
///
/// // Handle undefined input
/// let result = parse_partial_json(None);
/// assert!(result.value.is_none());
/// ```
pub fn parse_partial_json(json_text: Option<&str>) -> ParseResult {
    let Some(text) = json_text else {
        return ParseResult {
            value: None,
            state: ParseState::UndefinedInput,
        };
    };

    // Try direct parse first
    if let Ok(value) = serde_json::from_str(text) {
        return ParseResult {
            value: Some(value),
            state: ParseState::SuccessfulParse,
        };
    }

    // Try with repair
    let repaired = fix_json(text);
    if let Ok(value) = serde_json::from_str(&repaired) {
        return ParseResult {
            value: Some(value),
            state: ParseState::RepairedParse,
        };
    }

    ParseResult {
        value: None,
        state: ParseState::FailedParse,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_undefined_input() {
        let result = parse_partial_json(None);
        assert_eq!(result.state, ParseState::UndefinedInput);
        assert!(result.value.is_none());
    }

    #[test]
    fn test_successful_parse() {
        let result = parse_partial_json(Some(r#"{"name":"Alice","age":30}"#));
        assert_eq!(result.state, ParseState::SuccessfulParse);
        assert!(result.value.is_some());

        if let Some(Value::Object(obj)) = result.value {
            assert_eq!(obj.get("name").and_then(|v| v.as_str()), Some("Alice"));
            assert_eq!(obj.get("age").and_then(|v| v.as_i64()), Some(30));
        } else {
            panic!("Expected object");
        }
    }

    #[test]
    fn test_repaired_parse() {
        let result = parse_partial_json(Some(r#"{"name":"Alice","age":30"#));
        assert_eq!(result.state, ParseState::RepairedParse);
        assert!(result.value.is_some());

        if let Some(Value::Object(obj)) = result.value {
            assert_eq!(obj.get("name").and_then(|v| v.as_str()), Some("Alice"));
            assert_eq!(obj.get("age").and_then(|v| v.as_i64()), Some(30));
        } else {
            panic!("Expected object");
        }
    }

    #[test]
    fn test_repaired_array() {
        let result = parse_partial_json(Some(r#"[1,2,3"#));
        assert_eq!(result.state, ParseState::RepairedParse);
        assert!(result.value.is_some());

        if let Some(Value::Array(arr)) = result.value {
            assert_eq!(arr.len(), 3);
        } else {
            panic!("Expected array");
        }
    }

    #[test]
    fn test_failed_parse() {
        // Completely invalid JSON that can't be repaired
        let result = parse_partial_json(Some("this is not json at all {["));
        assert_eq!(result.state, ParseState::FailedParse);
        assert!(result.value.is_none());
    }

    #[test]
    fn test_partial_literal() {
        let result = parse_partial_json(Some(r#"{"done":tru"#));
        assert_eq!(result.state, ParseState::RepairedParse);
        assert!(result.value.is_some());

        if let Some(Value::Object(obj)) = result.value {
            assert_eq!(obj.get("done").and_then(|v| v.as_bool()), Some(true));
        } else {
            panic!("Expected object");
        }
    }

    #[test]
    fn test_nested_structures() {
        let result = parse_partial_json(Some(r#"{"outer":{"inner":"value""#));
        assert_eq!(result.state, ParseState::RepairedParse);
        assert!(result.value.is_some());
    }
}
