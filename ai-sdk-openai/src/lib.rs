mod api_types;
mod chat;
mod error;

pub use chat::OpenAIChatModel;
pub use error::OpenAIError;

// Factory function
pub fn openai(model_id: impl Into<String>, api_key: impl Into<String>) -> OpenAIChatModel {
    OpenAIChatModel::new(model_id, api_key)
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
