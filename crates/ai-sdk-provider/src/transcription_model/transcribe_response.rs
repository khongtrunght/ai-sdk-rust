use crate::json_value::JsonObject;
use crate::shared::SharedHeaders;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;

use super::warning::CallWarning;

/// A segment of transcribed text with timing information.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    /// The text content of this segment.
    pub text: String,

    /// The start time of this segment in seconds.
    #[serde(rename = "startSecond")]
    pub start_second: f64,

    /// The end time of this segment in seconds.
    #[serde(rename = "endSecond")]
    pub end_second: f64,
}

/// Request information for telemetry and debugging purposes.
#[derive(Debug, Clone)]
pub struct RequestInfo {
    /// Raw request HTTP body that was sent to the provider API as a string.
    /// JSON should be stringified. Non-HTTP(s) providers should not set this.
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
    pub body: Option<serde_json::Value>,
}

/// Response from a transcription model.
#[derive(Debug, Clone)]
pub struct TranscriptionResponse {
    /// The complete transcribed text from the audio.
    pub text: String,

    /// Array of transcript segments with timing information.
    /// Each segment represents a portion of the transcribed text with start and end times.
    pub segments: Vec<TranscriptionSegment>,

    /// The detected language of the audio content, as an ISO-639-1 code (e.g., 'en' for English).
    /// May be None if the language couldn't be detected.
    pub language: Option<String>,

    /// The total duration of the audio file in seconds.
    /// May be None if the duration couldn't be determined.
    pub duration_in_seconds: Option<f64>,

    /// Warnings for the call, e.g. unsupported settings.
    pub warnings: Vec<CallWarning>,

    /// Optional request information for telemetry and debugging purposes.
    pub request: Option<RequestInfo>,

    /// Response information for telemetry and debugging purposes.
    pub response: ResponseInfo,

    /// Additional provider-specific metadata.
    /// They are passed through from the provider to the AI SDK and enable
    /// provider-specific results that can be fully encapsulated in the provider.
    pub provider_metadata: Option<HashMap<String, JsonObject>>,
}
