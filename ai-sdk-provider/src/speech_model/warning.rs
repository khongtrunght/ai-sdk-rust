use serde::{Deserialize, Serialize};

/// Warning from the model provider for this call. The call will proceed, but e.g.
/// some settings might not be supported, which can lead to suboptimal results.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum CallWarning {
    /// A setting is not supported by the model
    #[serde(rename = "unsupported-setting")]
    UnsupportedSetting {
        /// The name of the unsupported setting
        setting: String,
        /// Optional details about why the setting is not supported
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },
    /// Other warning type
    #[serde(rename = "other")]
    Other {
        /// Warning message
        message: String,
    },
}
