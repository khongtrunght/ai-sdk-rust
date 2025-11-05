use crate::shared::{SharedHeaders, SharedProviderMetadata};
use std::time::SystemTime;

use super::CallWarning;

/// Audio data that can be either base64-encoded or binary.
#[derive(Debug, Clone)]
pub enum AudioData {
    /// Base64-encoded audio string
    Base64(String),
    /// Raw binary audio data
    Binary(Vec<u8>),
}

/// Optional request information for telemetry and debugging purposes.
#[derive(Debug, Clone)]
pub struct RequestInfo {
    /// Request body (available only for providers that use HTTP requests).
    pub body: Option<String>,
}

/// Response information for telemetry and debugging purposes.
#[derive(Debug, Clone)]
pub struct ResponseInfo {
    /// Timestamp for the start of the generated response.
    pub timestamp: SystemTime,

    /// The ID of the response model that was used to generate the response.
    pub model_id: String,

    /// Response headers.
    pub headers: Option<SharedHeaders>,

    /// Response body.
    pub body: Option<Vec<u8>>,
}

/// Response from a speech generation call.
#[derive(Debug, Clone)]
pub struct SpeechGenerateResponse {
    /// Generated audio as a string (base64) or binary data (Uint8Array).
    /// The audio should be returned without any unnecessary conversion.
    /// If the API returns base64 encoded strings, the audio should be returned
    /// as base64 encoded strings. If the API returns binary data, the audio
    /// should be returned as binary data.
    pub audio: AudioData,

    /// Warnings for the call, e.g. unsupported settings.
    pub warnings: Vec<CallWarning>,

    /// Optional request information for telemetry and debugging purposes.
    pub request: Option<RequestInfo>,

    /// Response information for telemetry and debugging purposes.
    pub response: ResponseInfo,

    /// Additional provider-specific metadata. They are passed through
    /// from the provider to the AI SDK and enable provider-specific
    /// results that can be fully encapsulated in the provider.
    pub provider_metadata: Option<SharedProviderMetadata>,
}
