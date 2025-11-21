use crate::json_value::JsonObject;
use serde::{Deserialize, Serialize};
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

/// Warning from the model that certain features are e.g. unsupported or that compatibility
/// functionality is used (which might lead to suboptimal results).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum SharedWarning {
    /// A configuration setting is not supported by the model
    #[serde(rename = "unsupported-setting")]
    UnsupportedSetting {
        /// The name of the unsupported setting
        setting: String,
        /// Optional details about why the setting is not supported
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },
    /// A compatibility feature is used that might lead to suboptimal results
    #[serde(rename = "compatibility")]
    Compatibility {
        /// The feature that is using compatibility mode
        feature: String,
        /// Optional details about the compatibility issue
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },
    /// Other warning
    #[serde(rename = "other")]
    Other {
        /// Warning message
        message: String,
    },
}
