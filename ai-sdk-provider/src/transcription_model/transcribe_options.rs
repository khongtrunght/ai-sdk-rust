use crate::shared::{SharedHeaders, SharedProviderOptions};

/// Audio input format for transcription.
#[derive(Debug, Clone)]
pub enum AudioInput {
    /// Raw audio bytes.
    Binary(Vec<u8>),
    /// Base64-encoded audio data.
    Base64(String),
}

/// Options for generating a transcription.
#[derive(Debug, Clone)]
pub struct TranscriptionOptions {
    /// Audio data to transcribe.
    /// Accepts either raw bytes or base64-encoded audio.
    pub audio: AudioInput,

    /// The IANA media type of the audio data.
    ///
    /// Examples: "audio/mpeg", "audio/wav", "audio/mp4"
    ///
    /// See: <https://www.iana.org/assignments/media-types/media-types.xhtml>
    pub media_type: String,

    /// Additional provider-specific options that are passed through to the provider.
    ///
    /// The outer record is keyed by the provider name, and the inner
    /// record is keyed by the provider-specific metadata key.
    ///
    /// Example:
    /// ```json
    /// {
    ///   "openai": {
    ///     "timestampGranularities": ["word"]
    ///   }
    /// }
    /// ```
    pub provider_options: Option<SharedProviderOptions>,

    /// Abort signal for cancelling the operation.
    /// Currently not implemented - reserved for future use.
    pub abort_signal: Option<()>,

    /// Additional HTTP headers to be sent with the request.
    /// Only applicable for HTTP-based providers.
    pub headers: Option<SharedHeaders>,
}

impl Default for TranscriptionOptions {
    fn default() -> Self {
        Self {
            audio: AudioInput::Binary(Vec::new()),
            media_type: String::new(),
            provider_options: None,
            abort_signal: None,
            headers: None,
        }
    }
}
