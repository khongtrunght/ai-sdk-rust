use super::transcribe_options::TranscriptionOptions;
use super::transcribe_response::TranscriptionResponse;
use async_trait::async_trait;

/// Transcription model specification version 3.
///
/// The transcription model interface allows converting audio to text
/// with optional timing information for segments.
#[async_trait]
pub trait TranscriptionModel: Send + Sync {
    /// The specification version that this model implements.
    fn specification_version(&self) -> &str {
        "v3"
    }

    /// Name of the provider for logging purposes.
    fn provider(&self) -> &str;

    /// Provider-specific model ID for logging purposes.
    fn model_id(&self) -> &str;

    /// Generates a transcript from audio.
    async fn do_generate(
        &self,
        options: TranscriptionOptions,
    ) -> Result<TranscriptionResponse, Box<dyn std::error::Error + Send + Sync>>;
}
