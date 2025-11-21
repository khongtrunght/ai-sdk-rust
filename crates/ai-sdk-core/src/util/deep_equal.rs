//! Deep equality checking for JSON values.
//!
//! This module provides utilities for comparing JSON values deeply,
//! handling nested structures and different numeric representations.

use serde_json::Value;

/// Checks if two JSON values are deeply equal.
///
/// This function recursively compares all nested structures, handling:
/// - Null values
/// - Booleans
/// - Numbers (compared as f64 for consistency)
/// - Strings
/// - Arrays (order-sensitive)
/// - Objects (order-insensitive)
///
/// # Examples
///
/// ```
/// use ai_sdk_core::util::is_deep_equal;
/// use serde_json::json;
///
/// // Equal objects (order doesn't matter)
/// assert!(is_deep_equal(
///     &json!({"a": 1, "b": 2}),
///     &json!({"b": 2, "a": 1})
/// ));
///
/// // Equal arrays (order matters)
/// assert!(is_deep_equal(
///     &json!([1, 2, 3]),
///     &json!([1, 2, 3])
/// ));
///
/// // Not equal arrays (different order)
/// assert!(!is_deep_equal(
///     &json!([1, 2, 3]),
///     &json!([3, 2, 1])
/// ));
///
/// // Nested structures
/// assert!(is_deep_equal(
///     &json!({"nested": {"value": [1, 2, 3]}}),
///     &json!({"nested": {"value": [1, 2, 3]}})
/// ));
/// ```
pub fn is_deep_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        // Null equality
        (Value::Null, Value::Null) => true,

        // Boolean equality
        (Value::Bool(a), Value::Bool(b)) => a == b,

        // Number equality - compare as f64 for consistency
        (Value::Number(a), Value::Number(b)) => {
            // Try to get as f64 for both
            match (a.as_f64(), b.as_f64()) {
                (Some(a_f), Some(b_f)) => {
                    // Handle NaN case
                    if a_f.is_nan() && b_f.is_nan() {
                        return true;
                    }
                    a_f == b_f
                }
                _ => {
                    // Fall back to exact comparison
                    // Try as i64 first (most common case)
                    if let (Some(a_i), Some(b_i)) = (a.as_i64(), b.as_i64()) {
                        a_i == b_i
                    } else if let (Some(a_u), Some(b_u)) = (a.as_u64(), b.as_u64()) {
                        a_u == b_u
                    } else {
                        false
                    }
                }
            }
        }

        // String equality
        (Value::String(a), Value::String(b)) => a == b,

        // Array equality - recursive, order matters
        (Value::Array(a), Value::Array(b)) => {
            if a.len() != b.len() {
                return false;
            }
            a.iter().zip(b.iter()).all(|(x, y)| is_deep_equal(x, y))
        }

        // Object equality - recursive, order doesn't matter
        (Value::Object(a), Value::Object(b)) => {
            if a.len() != b.len() {
                return false;
            }
            a.iter().all(|(key, val_a)| {
                b.get(key)
                    .map(|val_b| is_deep_equal(val_a, val_b))
                    .unwrap_or(false)
            })
        }

        // Different types
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_null_equality() {
        assert!(is_deep_equal(&json!(null), &json!(null)));
    }

    #[test]
    fn test_boolean_equality() {
        assert!(is_deep_equal(&json!(true), &json!(true)));
        assert!(is_deep_equal(&json!(false), &json!(false)));
        assert!(!is_deep_equal(&json!(true), &json!(false)));
    }

    #[test]
    fn test_number_equality() {
        assert!(is_deep_equal(&json!(42), &json!(42)));
        assert!(is_deep_equal(&json!(3.15), &json!(3.15)));
        assert!(is_deep_equal(&json!(-123), &json!(-123)));
        assert!(!is_deep_equal(&json!(1), &json!(2)));
    }

    #[test]
    fn test_string_equality() {
        assert!(is_deep_equal(&json!("hello"), &json!("hello")));
        assert!(!is_deep_equal(&json!("hello"), &json!("world")));
    }

    #[test]
    fn test_array_equality() {
        assert!(is_deep_equal(&json!([1, 2, 3]), &json!([1, 2, 3])));
        assert!(!is_deep_equal(&json!([1, 2, 3]), &json!([3, 2, 1])));
        assert!(!is_deep_equal(&json!([1, 2]), &json!([1, 2, 3])));
    }

    #[test]
    fn test_object_equality() {
        assert!(is_deep_equal(
            &json!({"a": 1, "b": 2}),
            &json!({"a": 1, "b": 2})
        ));
        assert!(is_deep_equal(
            &json!({"a": 1, "b": 2}),
            &json!({"b": 2, "a": 1})
        ));
        assert!(!is_deep_equal(
            &json!({"a": 1, "b": 2}),
            &json!({"a": 1, "b": 3})
        ));
        assert!(!is_deep_equal(&json!({"a": 1}), &json!({"a": 1, "b": 2})));
    }

    #[test]
    fn test_nested_structures() {
        assert!(is_deep_equal(
            &json!({"nested": {"value": [1, 2, 3]}}),
            &json!({"nested": {"value": [1, 2, 3]}})
        ));
        assert!(!is_deep_equal(
            &json!({"nested": {"value": [1, 2, 3]}}),
            &json!({"nested": {"value": [1, 2, 4]}})
        ));
    }

    #[test]
    fn test_different_types() {
        assert!(!is_deep_equal(&json!(1), &json!("1")));
        assert!(!is_deep_equal(&json!(true), &json!(1)));
        assert!(!is_deep_equal(&json!(null), &json!(0)));
        assert!(!is_deep_equal(&json!([]), &json!({})));
    }

    #[test]
    fn test_complex_nested() {
        let a = json!({
            "users": [
                {"name": "Alice", "age": 30, "active": true},
                {"name": "Bob", "age": 25, "active": false}
            ],
            "meta": {
                "version": "1.0",
                "count": 2
            }
        });

        let b = json!({
            "meta": {
                "count": 2,
                "version": "1.0"
            },
            "users": [
                {"age": 30, "name": "Alice", "active": true},
                {"active": false, "age": 25, "name": "Bob"}
            ]
        });

        assert!(is_deep_equal(&a, &b));
    }

    #[test]
    fn test_empty_structures() {
        assert!(is_deep_equal(&json!([]), &json!([])));
        assert!(is_deep_equal(&json!({}), &json!({})));
        assert!(!is_deep_equal(&json!([]), &json!({})));
    }
}
