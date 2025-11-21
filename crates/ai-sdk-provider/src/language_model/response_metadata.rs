use serde::{Deserialize, Serialize};

/// Response metadata from the provider.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResponseMetadata {
    /// Provider's response ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// Response timestamp
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>, // ISO 8601 string

    /// Actual model used (may differ from requested)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_metadata_serialization() {
        let metadata = ResponseMetadata {
            id: Some("resp-123".into()),
            timestamp: Some("2024-01-01T00:00:00Z".into()),
            model_id: Some("gpt-4".into()),
        };
        let json = serde_json::to_value(&metadata).unwrap();
        assert_eq!(json["id"], "resp-123");
        assert_eq!(json["timestamp"], "2024-01-01T00:00:00Z");
        assert_eq!(json["modelId"], "gpt-4");
    }

    #[test]
    fn test_response_metadata_partial() {
        let metadata = ResponseMetadata {
            id: Some("resp-123".into()),
            timestamp: None,
            model_id: None,
        };
        let json = serde_json::to_value(&metadata).unwrap();
        assert_eq!(json["id"], "resp-123");
        assert!(json.get("timestamp").is_none());
        assert!(json.get("modelId").is_none());
    }
}
