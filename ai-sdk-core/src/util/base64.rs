/// Base64 encoding and decoding utilities
/// Encodes binary data to base64 string
///
/// # Arguments
/// * `data` - The binary data to encode
///
/// # Returns
/// Base64-encoded string
///
/// # Example
/// ```
/// use ai_sdk_core::util::encode_base64;
///
/// let data = b"Hello, World!";
/// let encoded = encode_base64(data);
/// assert_eq!(encoded, "SGVsbG8sIFdvcmxkIQ==");
/// ```
pub fn encode_base64(data: &[u8]) -> String {
    use base64::{engine::general_purpose::STANDARD, Engine};
    STANDARD.encode(data)
}

/// Decodes a base64 string to binary data
///
/// Supports both standard and URL-safe base64 encoding.
///
/// # Arguments
/// * `s` - The base64 string to decode
///
/// # Returns
/// Decoded binary data or error if the string is not valid base64
///
/// # Example
/// ```
/// use ai_sdk_core::util::decode_base64;
///
/// let encoded = "SGVsbG8sIFdvcmxkIQ==";
/// let decoded = decode_base64(encoded).unwrap();
/// assert_eq!(decoded, b"Hello, World!");
/// ```
pub fn decode_base64(s: &str) -> Result<Vec<u8>, base64::DecodeError> {
    use base64::{engine::general_purpose::STANDARD, Engine};

    // Handle URL-safe base64 by converting to standard base64
    let normalized = s.replace('-', "+").replace('_', "/");
    STANDARD.decode(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode() {
        let data = b"Hello, World!";
        let encoded = encode_base64(data);
        assert_eq!(encoded, "SGVsbG8sIFdvcmxkIQ==");
    }

    #[test]
    fn test_decode() {
        let encoded = "SGVsbG8sIFdvcmxkIQ==";
        let decoded = decode_base64(encoded).unwrap();
        assert_eq!(decoded, b"Hello, World!");
    }

    #[test]
    fn test_roundtrip() {
        let original = b"The quick brown fox jumps over the lazy dog";
        let encoded = encode_base64(original);
        let decoded = decode_base64(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_decode_url_safe() {
        // URL-safe base64 uses - and _ instead of + and /
        let url_safe = "SGVsbG8sIFdvcmxkIQ==".replace('+', "-").replace('/', "_");
        let decoded = decode_base64(&url_safe).unwrap();
        assert_eq!(decoded, b"Hello, World!");
    }

    #[test]
    fn test_decode_invalid() {
        let invalid = "not valid base64!!!";
        assert!(decode_base64(invalid).is_err());
    }

    #[test]
    fn test_encode_empty() {
        let data = b"";
        let encoded = encode_base64(data);
        assert_eq!(encoded, "");
    }

    #[test]
    fn test_decode_empty() {
        let decoded = decode_base64("").unwrap();
        assert_eq!(decoded, b"");
    }
}
