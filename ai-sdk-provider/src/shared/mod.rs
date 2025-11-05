use crate::json_value::JsonObject;
use std::collections::HashMap;

/// HTTP headers as key-value pairs
pub type SharedHeaders = HashMap<String, String>;

/// Additional provider-specific options (input).
/// Keyed by provider name, values are provider-specific configuration.
///
/// Example:
/// ```json
/// {
///   "anthropic": {
///     "cacheControl": { "type": "ephemeral" }
///   }
/// }
/// ```
pub type SharedProviderOptions = HashMap<String, JsonObject>;

/// Additional provider-specific metadata (output).
/// Keyed by provider name, values are provider-specific data.
pub type SharedProviderMetadata = HashMap<String, JsonObject>;
