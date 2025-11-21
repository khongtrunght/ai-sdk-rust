use crate::shared::{SharedHeaders, SharedProviderOptions};

/// Options for speech generation.
#[derive(Debug, Clone, Default)]
pub struct SpeechGenerateOptions {
    /// Text to convert to speech.
    pub text: String,

    /// The voice to use for speech synthesis.
    /// This is provider-specific and may be a voice ID, name, or other identifier.
    pub voice: Option<String>,

    /// The desired output format for the audio e.g. "mp3", "wav", etc.
    pub output_format: Option<String>,

    /// Instructions for the speech generation e.g. "Speak in a slow and steady tone".
    pub instructions: Option<String>,

    /// The speed of the speech generation.
    pub speed: Option<f32>,

    /// The language for speech generation. This should be an ISO 639-1 language code
    /// (e.g. "en", "es", "fr") or "auto" for automatic language detection.
    /// Provider support varies.
    pub language: Option<String>,

    /// Additional provider-specific options that are passed through to the provider
    /// as body parameters.
    pub provider_options: Option<SharedProviderOptions>,

    /// Abort signal for cancelling the operation.
    pub abort_signal: Option<tokio::sync::watch::Receiver<bool>>,

    /// Additional HTTP headers to be sent with the request.
    /// Only applicable for HTTP-based providers.
    pub headers: Option<SharedHeaders>,
}
