//! JSON repair utility for fixing incomplete or malformed JSON strings.
//!
//! This module provides a state machine-based JSON repair function that can:
//! - Complete incomplete strings
//! - Close unclosed objects and arrays
//! - Complete partial literals (true, false, null)
//!
//! The implementation uses a single-pass state machine with a stack to track
//! nesting levels, making it efficient for streaming JSON repair.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Root,
    InsideString,
    InsideStringEscape,
    InsideLiteral,
    InsideNumber,
    InsideObjectStart,
    InsideObjectKey,
    InsideObjectBeforeValue,
    InsideObjectAfterValue,
    InsideObjectAfterComma,
    InsideArrayStart,
    InsideArrayAfterValue,
    InsideArrayAfterComma,
}

/// Repairs incomplete or malformed JSON strings.
///
/// This function attempts to fix JSON strings that may be incomplete due to:
/// - Streaming JSON generation
/// - Network interruptions
/// - Character limits
///
/// # Examples
///
/// ```
/// use ai_sdk_core::util::fix_json;
///
/// // Complete an incomplete object
/// assert_eq!(fix_json(r#"{"name":"Alice""#), r#"{"name":"Alice"}"#);
///
/// // Close unclosed arrays
/// assert_eq!(fix_json(r#"[1,2,3"#), r#"[1,2,3]"#);
///
/// // Complete partial literals
/// assert_eq!(fix_json(r#"{"done":tru"#), r#"{"done":true}"#);
/// ```
pub fn fix_json(input: &str) -> String {
    if input.is_empty() {
        return String::new();
    }

    let mut stack: Vec<State> = vec![State::Root];
    let mut last_valid_byte_index = 0;
    let mut literal_start: Option<usize> = None;
    let char_indices: Vec<(usize, char)> = input.char_indices().collect();
    let mut i = 0;

    while i < char_indices.len() {
        let (byte_idx, ch) = char_indices[i];
        let current_state = *stack.last().unwrap_or(&State::Root);

        match (current_state, ch) {
            // Root state
            (State::Root, '{') => {
                stack.push(State::InsideObjectStart);
                last_valid_byte_index = byte_idx;
            }
            (State::Root, '[') => {
                stack.push(State::InsideArrayStart);
                last_valid_byte_index = byte_idx;
            }
            (State::Root, '"') => {
                stack.push(State::InsideString);
                last_valid_byte_index = byte_idx;
            }
            (State::Root, c) if c.is_whitespace() => {
                last_valid_byte_index = byte_idx;
            }
            (State::Root, c) if c.is_ascii_digit() || c == '-' => {
                stack.push(State::InsideNumber);
                last_valid_byte_index = byte_idx;
            }
            (State::Root, 't') | (State::Root, 'f') | (State::Root, 'n') => {
                stack.push(State::InsideLiteral);
                literal_start = Some(byte_idx);
                last_valid_byte_index = byte_idx;
            }

            // Inside string
            (State::InsideString, '"') => {
                stack.pop();
                last_valid_byte_index = byte_idx;
            }
            (State::InsideString, '\\') => {
                stack.push(State::InsideStringEscape);
                last_valid_byte_index = byte_idx;
            }
            (State::InsideString, _) => {
                last_valid_byte_index = byte_idx;
            }

            // Inside string escape
            (State::InsideStringEscape, _) => {
                stack.pop();
                last_valid_byte_index = byte_idx;
            }

            // Inside literal
            (State::InsideLiteral, c) if c.is_ascii_alphabetic() => {
                last_valid_byte_index = byte_idx;
            }
            (State::InsideLiteral, _) => {
                stack.pop();
                // Don't increment i, reprocess this character
                continue;
            }

            // Inside number
            (State::InsideNumber, c)
                if c.is_ascii_digit()
                    || c == '.'
                    || c == 'e'
                    || c == 'E'
                    || c == '+'
                    || c == '-' =>
            {
                last_valid_byte_index = byte_idx;
            }
            (State::InsideNumber, _) => {
                stack.pop();
                // Don't increment i, reprocess this character
                continue;
            }

            // Inside object start
            (State::InsideObjectStart, '"') => {
                stack.pop();
                stack.push(State::InsideObjectKey);
                stack.push(State::InsideString);
                last_valid_byte_index = byte_idx;
            }
            (State::InsideObjectStart, '}') => {
                stack.pop();
                last_valid_byte_index = byte_idx;
            }
            (State::InsideObjectStart, c) if c.is_whitespace() => {
                last_valid_byte_index = byte_idx;
            }

            // Inside object key
            (State::InsideObjectKey, ':') => {
                stack.pop();
                stack.push(State::InsideObjectBeforeValue);
                last_valid_byte_index = byte_idx;
            }
            (State::InsideObjectKey, c) if c.is_whitespace() => {
                last_valid_byte_index = byte_idx;
            }

            // Inside object before value
            (State::InsideObjectBeforeValue, '{') => {
                stack.pop();
                stack.push(State::InsideObjectAfterValue);
                stack.push(State::InsideObjectStart);
                last_valid_byte_index = byte_idx;
            }
            (State::InsideObjectBeforeValue, '[') => {
                stack.pop();
                stack.push(State::InsideObjectAfterValue);
                stack.push(State::InsideArrayStart);
                last_valid_byte_index = byte_idx;
            }
            (State::InsideObjectBeforeValue, '"') => {
                stack.pop();
                stack.push(State::InsideObjectAfterValue);
                stack.push(State::InsideString);
                last_valid_byte_index = byte_idx;
            }
            (State::InsideObjectBeforeValue, c) if c.is_ascii_digit() || c == '-' => {
                stack.pop();
                stack.push(State::InsideObjectAfterValue);
                stack.push(State::InsideNumber);
                last_valid_byte_index = byte_idx;
            }
            (State::InsideObjectBeforeValue, 't')
            | (State::InsideObjectBeforeValue, 'f')
            | (State::InsideObjectBeforeValue, 'n') => {
                stack.pop();
                stack.push(State::InsideObjectAfterValue);
                stack.push(State::InsideLiteral);
                literal_start = Some(byte_idx);
                last_valid_byte_index = byte_idx;
            }
            (State::InsideObjectBeforeValue, c) if c.is_whitespace() => {
                last_valid_byte_index = byte_idx;
            }

            // Inside object after value
            (State::InsideObjectAfterValue, ',') => {
                stack.pop();
                stack.push(State::InsideObjectAfterComma);
                last_valid_byte_index = byte_idx;
            }
            (State::InsideObjectAfterValue, '}') => {
                stack.pop();
                last_valid_byte_index = byte_idx;
            }
            (State::InsideObjectAfterValue, c) if c.is_whitespace() => {
                last_valid_byte_index = byte_idx;
            }

            // Inside object after comma
            (State::InsideObjectAfterComma, '"') => {
                stack.pop();
                stack.push(State::InsideObjectKey);
                stack.push(State::InsideString);
                last_valid_byte_index = byte_idx;
            }
            (State::InsideObjectAfterComma, c) if c.is_whitespace() => {
                last_valid_byte_index = byte_idx;
            }

            // Inside array start
            (State::InsideArrayStart, '{') => {
                stack.pop();
                stack.push(State::InsideArrayAfterValue);
                stack.push(State::InsideObjectStart);
                last_valid_byte_index = byte_idx;
            }
            (State::InsideArrayStart, '[') => {
                stack.pop();
                stack.push(State::InsideArrayAfterValue);
                stack.push(State::InsideArrayStart);
                last_valid_byte_index = byte_idx;
            }
            (State::InsideArrayStart, '"') => {
                stack.pop();
                stack.push(State::InsideArrayAfterValue);
                stack.push(State::InsideString);
                last_valid_byte_index = byte_idx;
            }
            (State::InsideArrayStart, c) if c.is_ascii_digit() || c == '-' => {
                stack.pop();
                stack.push(State::InsideArrayAfterValue);
                stack.push(State::InsideNumber);
                last_valid_byte_index = byte_idx;
            }
            (State::InsideArrayStart, 't')
            | (State::InsideArrayStart, 'f')
            | (State::InsideArrayStart, 'n') => {
                stack.pop();
                stack.push(State::InsideArrayAfterValue);
                stack.push(State::InsideLiteral);
                literal_start = Some(byte_idx);
                last_valid_byte_index = byte_idx;
            }
            (State::InsideArrayStart, ']') => {
                stack.pop();
                last_valid_byte_index = byte_idx;
            }
            (State::InsideArrayStart, c) if c.is_whitespace() => {
                last_valid_byte_index = byte_idx;
            }

            // Inside array after value
            (State::InsideArrayAfterValue, ',') => {
                stack.pop();
                stack.push(State::InsideArrayAfterComma);
                last_valid_byte_index = byte_idx;
            }
            (State::InsideArrayAfterValue, ']') => {
                stack.pop();
                last_valid_byte_index = byte_idx;
            }
            (State::InsideArrayAfterValue, c) if c.is_whitespace() => {
                last_valid_byte_index = byte_idx;
            }

            // Inside array after comma
            (State::InsideArrayAfterComma, '{') => {
                stack.pop();
                stack.push(State::InsideArrayAfterValue);
                stack.push(State::InsideObjectStart);
                last_valid_byte_index = byte_idx;
            }
            (State::InsideArrayAfterComma, '[') => {
                stack.pop();
                stack.push(State::InsideArrayAfterValue);
                stack.push(State::InsideArrayStart);
                last_valid_byte_index = byte_idx;
            }
            (State::InsideArrayAfterComma, '"') => {
                stack.pop();
                stack.push(State::InsideArrayAfterValue);
                stack.push(State::InsideString);
                last_valid_byte_index = byte_idx;
            }
            (State::InsideArrayAfterComma, c) if c.is_ascii_digit() || c == '-' => {
                stack.pop();
                stack.push(State::InsideArrayAfterValue);
                stack.push(State::InsideNumber);
                last_valid_byte_index = byte_idx;
            }
            (State::InsideArrayAfterComma, 't')
            | (State::InsideArrayAfterComma, 'f')
            | (State::InsideArrayAfterComma, 'n') => {
                stack.pop();
                stack.push(State::InsideArrayAfterValue);
                stack.push(State::InsideLiteral);
                literal_start = Some(byte_idx);
                last_valid_byte_index = byte_idx;
            }
            (State::InsideArrayAfterComma, c) if c.is_whitespace() => {
                last_valid_byte_index = byte_idx;
            }

            _ => {
                // Unknown state transition, keep last valid index
            }
        }

        i += 1;
    }

    // Build repaired JSON from valid portion
    let mut result = if last_valid_byte_index < input.len() {
        // Get the end byte index (inclusive of the last valid character)
        let end_byte = if last_valid_byte_index < input.len() {
            // Find the next character boundary after last_valid_byte_index
            input
                .char_indices()
                .find(|(idx, _)| *idx > last_valid_byte_index)
                .map(|(idx, _)| idx)
                .unwrap_or(input.len())
        } else {
            input.len()
        };
        input[..end_byte].to_string()
    } else {
        input.to_string()
    };

    // Pop states and repair
    while let Some(state) = stack.pop() {
        match state {
            State::Root => break,
            State::InsideString => {
                result.push('"');
            }
            State::InsideStringEscape => {
                // Invalid escape at end, remove backslash
                result.pop();
            }
            State::InsideLiteral => {
                // Complete partial literal
                if let Some(start_byte) = literal_start {
                    // Find the end byte (the next char boundary after last_valid_byte_index)
                    let end_byte = input
                        .char_indices()
                        .find(|(idx, _)| *idx > last_valid_byte_index)
                        .map(|(idx, _)| idx)
                        .unwrap_or(input.len());
                    let partial = &input[start_byte..end_byte];
                    if "true".starts_with(partial) && partial != "true" {
                        result.push_str(&"true"[partial.len()..]);
                    } else if "false".starts_with(partial) && partial != "false" {
                        result.push_str(&"false"[partial.len()..]);
                    } else if "null".starts_with(partial) && partial != "null" {
                        result.push_str(&"null"[partial.len()..]);
                    }
                }
            }
            State::InsideObjectStart
            | State::InsideObjectKey
            | State::InsideObjectBeforeValue
            | State::InsideObjectAfterValue
            | State::InsideObjectAfterComma => {
                result.push('}');
            }
            State::InsideArrayStart
            | State::InsideArrayAfterValue
            | State::InsideArrayAfterComma => {
                result.push(']');
            }
            State::InsideNumber => {
                // Number is complete as-is
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complete_string() {
        assert_eq!(fix_json(r#"{"name":"Alice""#), r#"{"name":"Alice"}"#);
    }

    #[test]
    fn test_close_object() {
        assert_eq!(fix_json(r#"{"a":1"#), r#"{"a":1}"#);
        assert_eq!(fix_json(r#"{"a":1,"b":2"#), r#"{"a":1,"b":2}"#);
    }

    #[test]
    fn test_close_array() {
        assert_eq!(fix_json(r#"[1,2,3"#), r#"[1,2,3]"#);
        assert_eq!(fix_json(r#"["a","b""#), r#"["a","b"]"#);
    }

    #[test]
    fn test_complete_literal() {
        assert_eq!(fix_json(r#"{"done":tru"#), r#"{"done":true}"#);
        assert_eq!(fix_json(r#"{"done":fals"#), r#"{"done":false}"#);
        assert_eq!(fix_json(r#"{"value":nul"#), r#"{"value":null}"#);
    }

    #[test]
    fn test_nested_structures() {
        assert_eq!(
            fix_json(r#"{"outer":{"inner":"value""#),
            r#"{"outer":{"inner":"value"}}"#
        );
        assert_eq!(
            fix_json(r#"{"array":[1,2,{"nested":"val""#),
            r#"{"array":[1,2,{"nested":"val"}]}"#
        );
    }

    #[test]
    fn test_already_valid() {
        let valid = r#"{"name":"Alice","age":30}"#;
        assert_eq!(fix_json(valid), valid);
    }

    #[test]
    fn test_empty_string() {
        assert_eq!(fix_json(""), "");
    }

    #[test]
    fn test_escaped_characters() {
        assert_eq!(
            fix_json(r#"{"text":"hello\"world""#),
            r#"{"text":"hello\"world"}"#
        );
    }

    #[test]
    fn test_numbers() {
        assert_eq!(fix_json(r#"{"count":42"#), r#"{"count":42}"#);
        assert_eq!(fix_json(r#"{"value":3.14"#), r#"{"value":3.14}"#);
        assert_eq!(fix_json(r#"{"value":-123"#), r#"{"value":-123}"#);
    }
}
