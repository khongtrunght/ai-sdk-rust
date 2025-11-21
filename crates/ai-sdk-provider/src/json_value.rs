use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A JSON value can be a string, number, boolean, object, array, or null.
/// JSON values can be serialized and deserialized by serde_json.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum JsonValue {
    /// JSON null value
    Null,
    /// JSON boolean value
    Bool(bool),
    /// JSON number value
    Number(serde_json::Number),
    /// JSON string value
    String(String),
    /// JSON array value
    Array(Vec<JsonValue>),
    /// JSON object value
    Object(JsonObject),
}

/// JSON object type
pub type JsonObject = HashMap<String, JsonValue>;

/// JSON array type
pub type JsonArray = Vec<JsonValue>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_value_roundtrip() {
        let json = r#"{"key": "value", "number": 42, "array": [1, 2, 3]}"#;
        let value: JsonValue = serde_json::from_str(json).unwrap();
        let serialized = serde_json::to_string(&value).unwrap();
        let deserialized: JsonValue = serde_json::from_str(&serialized).unwrap();
        assert_eq!(value, deserialized);
    }
}
