mod common;

use ai_sdk_openai::OpenAIImageModel;
use ai_sdk_provider::{ImageData, ImageGenerateOptions, ImageModel};
use common::{load_json_fixture, TestServer};

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

// Fixture-based tests (run without API key)

#[tokio::test]
async fn test_openai_image_generation_with_fixture() {
    // Setup mock server
    let test_server = TestServer::new().await;

    // Load fixture
    let fixture = load_json_fixture("image-dalle3-1");

    // Configure mock to return fixture
    test_server
        .mock_json_response("/v1/images/generations", fixture)
        .await;

    // Create model pointing to mock server
    let model = OpenAIImageModel::new("dall-e-3", "test-key")
        .with_base_url(format!("{}/v1", test_server.base_url));

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

    // Verify response structure
    assert_eq!(response.images.len(), 1);

    // Verify we got base64 data (not binary)
    match &response.images[0] {
        ImageData::Base64(data) => {
            assert!(!data.is_empty());
            // Verify it's valid base64
            assert_eq!(
                data,
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            );
        }
        ImageData::Binary(_) => panic!("Expected base64 data"),
    }
}
