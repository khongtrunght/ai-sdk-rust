//! # AI SDK Provider Utils
//!
//! Utility functions and helpers for AI SDK providers.
//!
//! This crate provides common utilities that can be shared across
//! different AI provider implementations.

#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]

use std::collections::HashMap;

/// Merges a base set of headers with optional overrides.
///
/// # Arguments
/// * base - The default headers (usually from config). This map is consumed to avoid extra allocation.
/// * overrides - The optional headers from the user request. These take precedence over base.
pub fn merge_headers(
    mut base: HashMap<String, String>,
    overrides: Option<&HashMap<String, String>>,
) -> HashMap<String, String> {
    if let Some(headers) = overrides {
        // Extend the base map. If keys exist in both, the 'overrides' value overwrites the 'base' value.
        base.extend(headers.iter().map(|(k, v)| (k.clone(), v.clone())));
    }
    base
}

/// Merges headers and converts them to a reqwest HeaderMap.
///
/// This is a convenience function that combines `merge_headers` with conversion
/// to `reqwest::header::HeaderMap`, which can be used directly with `.headers()`.
///
/// # Arguments
/// * base - The default headers (usually from config). This map is consumed to avoid extra allocation.
/// * overrides - The optional headers from the user request. These take precedence over base.
///
/// # Example
/// ```rust,ignore
/// let config_headers = get_config_headers();
/// let custom_headers = Some(user_headers);
///
/// let response = client
///     .post(url)
///     .headers(merge_headers_reqwest(config_headers, custom_headers.as_ref()))
///     .json(&body)
///     .send()
///     .await?;
/// ```
#[cfg(feature = "reqwest")]
pub fn merge_headers_reqwest(
    base: HashMap<String, String>,
    overrides: Option<&HashMap<String, String>>,
) -> reqwest::header::HeaderMap {
    let merged = merge_headers(base, overrides);
    let mut header_map = reqwest::header::HeaderMap::new();

    for (key, value) in merged {
        if let (Ok(name), Ok(val)) = (
            reqwest::header::HeaderName::from_bytes(key.as_bytes()),
            reqwest::header::HeaderValue::from_str(&value),
        ) {
            header_map.insert(name, val);
        }
    }

    header_map
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
