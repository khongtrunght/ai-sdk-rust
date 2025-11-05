use crate::{JsonObject, SharedHeaders};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::CallWarning;

/// Image data - can be either base64 encoded string or binary data
#[derive(Debug, Clone)]
pub enum ImageData {
    /// Base64 encoded image string
    Base64(String),
    /// Binary image data
    Binary(Vec<u8>),
}

/// Provider-specific metadata for images
///
/// The outer record is keyed by the provider name, and contains
/// provider-specific metadata including an `images` array with
/// per-image metadata.
///
/// Example:
/// ```json
/// {
///   "openai": {
///     "images": [{"revisedPrompt": "A detailed prompt..."}]
///   }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageProviderMetadata {
    #[serde(flatten)]
    pub metadata: HashMap<String, JsonObject>,
}

/// Response from image generation
#[derive(Debug, Clone)]
pub struct ImageGenerateResponse {
    /// Generated images as base64 encoded strings or binary data.
    /// The images are returned without any unnecessary conversion.
    /// If the API returns base64 encoded strings, the images are returned
    /// as base64 encoded strings. If the API returns binary data, the images
    /// are returned as binary data.
    pub images: Vec<ImageData>,

    /// Warnings for the call, e.g. unsupported settings
    pub warnings: Vec<CallWarning>,

    /// Additional provider-specific metadata. They are passed through
    /// from the provider to the AI SDK and enable provider-specific
    /// results that can be fully encapsulated in the provider.
    pub provider_metadata: Option<ImageProviderMetadata>,

    /// Response information for telemetry and debugging purposes
    pub response: ResponseInfo,
}

/// Response metadata for debugging
#[derive(Debug, Clone)]
pub struct ResponseInfo {
    /// Timestamp for the start of the generated response
    pub timestamp: std::time::SystemTime,

    /// The ID of the response model that was used to generate the response
    pub model_id: String,

    /// Response headers
    pub headers: Option<SharedHeaders>,
}
