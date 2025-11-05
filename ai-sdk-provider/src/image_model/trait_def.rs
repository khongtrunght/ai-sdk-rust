use super::*;
use async_trait::async_trait;

/// Image generation model specification version 3.
///
/// The image model must specify which image model interface
/// version it implements. This will allow us to evolve the image
/// model interface and retain backwards compatibility.
#[async_trait]
pub trait ImageModel: Send + Sync {
    /// Specification version (always "v3")
    fn specification_version(&self) -> &str {
        "v3"
    }

    /// Name of the provider for logging purposes (e.g., "openai")
    fn provider(&self) -> &str;

    /// Provider-specific model ID for logging purposes (e.g., "dall-e-3")
    fn model_id(&self) -> &str;

    /// Limit of how many images can be generated in a single API call.
    ///
    /// Returns None for models that do not have a limit.
    async fn max_images_per_call(&self) -> Option<usize>;

    /// Generates an array of images.
    ///
    /// Naming: "do" prefix to prevent accidental direct usage of the method by the user.
    async fn do_generate(
        &self,
        options: ImageGenerateOptions,
    ) -> Result<ImageGenerateResponse, Box<dyn std::error::Error + Send + Sync>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyImageModel;

    #[async_trait]
    impl ImageModel for DummyImageModel {
        fn provider(&self) -> &str {
            "test"
        }

        fn model_id(&self) -> &str {
            "dummy"
        }

        async fn max_images_per_call(&self) -> Option<usize> {
            Some(1)
        }

        async fn do_generate(
            &self,
            _options: ImageGenerateOptions,
        ) -> Result<ImageGenerateResponse, Box<dyn std::error::Error + Send + Sync>> {
            Ok(ImageGenerateResponse {
                images: vec![ImageData::Base64("test".to_string())],
                warnings: vec![],
                provider_metadata: None,
                response: ResponseInfo {
                    timestamp: std::time::SystemTime::now(),
                    model_id: "dummy".to_string(),
                    headers: None,
                },
            })
        }
    }

    #[tokio::test]
    async fn test_image_model_trait() {
        let model = DummyImageModel;
        assert_eq!(model.provider(), "test");
        assert_eq!(model.model_id(), "dummy");
        assert_eq!(model.specification_version(), "v3");
        assert_eq!(model.max_images_per_call().await, Some(1));
    }
}
