use super::{SpeechGenerateOptions, SpeechGenerateResponse};
use async_trait::async_trait;

/// Speech model specification version 3.
///
/// A speech model generates speech audio from text input.
#[async_trait]
pub trait SpeechModel: Send + Sync {
    /// The speech model specification version.
    /// This will allow us to evolve the speech model interface and retain backwards compatibility.
    fn specification_version(&self) -> &str {
        "v3"
    }

    /// Name of the provider for logging purposes.
    fn provider(&self) -> &str;

    /// Provider-specific model ID for logging purposes.
    fn model_id(&self) -> &str;

    /// Generates speech audio from text.
    async fn do_generate(
        &self,
        options: SpeechGenerateOptions,
    ) -> Result<SpeechGenerateResponse, Box<dyn std::error::Error + Send + Sync>>;
}
