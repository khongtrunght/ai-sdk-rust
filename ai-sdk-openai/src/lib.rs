mod api_types;
mod chat;
mod embedding;
mod error;
mod image;
mod speech;
mod transcription;

pub use chat::OpenAIChatModel;
pub use embedding::OpenAIEmbeddingModel;
pub use error::OpenAIError;
pub use image::OpenAIImageModel;
pub use speech::OpenAISpeechModel;
pub use transcription::OpenAITranscriptionModel;

// Factory functions
pub fn openai(model_id: impl Into<String>, api_key: impl Into<String>) -> OpenAIChatModel {
    OpenAIChatModel::new(model_id, api_key)
}

pub fn openai_embedding(
    model_id: impl Into<String>,
    api_key: impl Into<String>,
) -> OpenAIEmbeddingModel {
    OpenAIEmbeddingModel::new(model_id, api_key)
}

pub fn openai_image(
    model_id: impl Into<String>,
    api_key: impl Into<String>,
) -> OpenAIImageModel {
    OpenAIImageModel::new(model_id, api_key)
}

pub fn openai_speech(
    model_id: impl Into<String>,
    api_key: impl Into<String>,
) -> OpenAISpeechModel {
    OpenAISpeechModel::new(model_id, api_key)
}

pub fn openai_transcription(
    model_id: impl Into<String>,
    api_key: impl Into<String>,
) -> OpenAITranscriptionModel {
    OpenAITranscriptionModel::new(model_id, api_key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_sdk_provider::LanguageModel;

    #[test]
    fn test_model_creation() {
        let model = openai("gpt-4", "test-key");
        assert_eq!(model.provider(), "openai");
        assert_eq!(model.model_id(), "gpt-4");
    }
}
