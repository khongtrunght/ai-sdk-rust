use ai_sdk_openai::OpenAIImageModel;
use ai_sdk_provider::{ImageData, ImageGenerateOptions, ImageModel};

#[tokio::test]
#[ignore] // Requires API key
async fn test_openai_image_generation() {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");

    let model = OpenAIImageModel::new("dall-e-3", api_key);

    let options = ImageGenerateOptions {
        prompt: "A cute cat wearing sunglasses".into(),
        n: 1,
        size: Some("1024x1024".into()),
        aspect_ratio: None,
        seed: None,
        provider_options: None,
        headers: None,
    };

    let response = model.do_generate(options).await.unwrap();

    assert_eq!(response.images.len(), 1);
    match &response.images[0] {
        ImageData::Base64(data) => assert!(!data.is_empty()),
        ImageData::Binary(_) => panic!("Expected base64 data"),
    }
}

#[tokio::test]
async fn test_max_images_per_call() {
    let model = OpenAIImageModel::new("dall-e-3", "test-key");
    assert_eq!(model.max_images_per_call().await, Some(1));

    let model = OpenAIImageModel::new("dall-e-2", "test-key");
    assert_eq!(model.max_images_per_call().await, Some(10));

    let model = OpenAIImageModel::new("gpt-image-1", "test-key");
    assert_eq!(model.max_images_per_call().await, Some(10));
}

#[tokio::test]
async fn test_model_properties() {
    let model = OpenAIImageModel::new("dall-e-3", "test-key");
    assert_eq!(model.provider(), "openai");
    assert_eq!(model.model_id(), "dall-e-3");
    assert_eq!(model.specification_version(), "v3");
}
