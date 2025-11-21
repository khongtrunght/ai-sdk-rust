use serde::{Deserialize, Serialize};

/// Data content can be binary, base64, or a URL
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DataContent {
    /// Binary data
    Binary(Vec<u8>),
    /// Base64-encoded string
    Base64(String),
    /// URL reference
    Url(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_content_url() {
        let content = DataContent::Url("https://example.com/image.png".into());
        let json = serde_json::to_string(&content).unwrap();
        assert_eq!(json, r#""https://example.com/image.png""#);
    }

    #[test]
    fn test_data_content_base64() {
        let content = DataContent::Base64("SGVsbG8gV29ybGQ=".into());
        let json = serde_json::to_string(&content).unwrap();
        assert_eq!(json, r#""SGVsbG8gV29ybGQ=""#);
    }
}
