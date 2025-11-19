mod common;

use ai_sdk_openai::OpenAIImageModel;
use ai_sdk_provider::{ImageData, ImageGenerateOptions, ImageModel};
use common::TestServer;
use serde_json::json;

/// Helper to create image generation response JSON (like TypeScript's prepareJsonResponse)
fn create_image_response(images: &[(&str, Option<&str>)]) -> serde_json::Value {
    json!({
        "created": 1733837122,
        "data": images.iter().map(|(b64, revised_prompt)| {
            let mut obj = json!({
                "b64_json": b64
            });
            if let Some(prompt) = revised_prompt {
                obj["revised_prompt"] = json!(prompt);
            }
            obj
        }).collect::<Vec<_>>()
    })
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

#[tokio::test]
async fn test_openai_image_extract_images() {
    // Setup mock server
    let test_server = TestServer::new().await;

    // Create response with mock images (like TypeScript)
    let response = create_image_response(&[
        (
            "base64-image-1",
            Some("A charming visual illustration of a baby sea otter swimming joyously."),
        ),
        ("base64-image-2", None),
    ]);

    test_server
        .mock_json_response("/v1/images/generations", response)
        .await;

    let model = OpenAIImageModel::new("dall-e-3", "test-key")
        .with_base_url(format!("{}/v1", test_server.base_url));

    let options = ImageGenerateOptions {
        prompt: "A cute baby sea otter".into(),
        n: 1,
        size: None,
        aspect_ratio: None,
        seed: None,
        provider_options: None,
        headers: None,
    };

    let result = model.do_generate(options).await.unwrap();

    // Verify we got both images
    assert_eq!(result.images.len(), 2);

    match &result.images[0] {
        ImageData::Base64(data) => assert_eq!(data, "base64-image-1"),
        ImageData::Binary(_) => panic!("Expected base64 data"),
    }

    match &result.images[1] {
        ImageData::Base64(data) => assert_eq!(data, "base64-image-2"),
        ImageData::Binary(_) => panic!("Expected base64 data"),
    }
}

#[tokio::test]
async fn test_openai_image_pass_model_and_settings() {
    // Setup mock server
    let test_server = TestServer::new().await;

    let response = create_image_response(&[("base64-image-1", None)]);

    test_server
        .mock_json_response("/v1/images/generations", response)
        .await;

    let model = OpenAIImageModel::new("dall-e-3", "test-key")
        .with_base_url(format!("{}/v1", test_server.base_url));

    let options = ImageGenerateOptions {
        prompt: "A cute baby sea otter".into(),
        n: 1,
        size: Some("1024x1024".into()),
        aspect_ratio: None,
        seed: None,
        provider_options: None,
        headers: None,
    };

    model.do_generate(options).await.unwrap();

    // Verify request body
    let request_body = test_server
        .last_request_body()
        .await
        .expect("Request body should exist");

    let expected = json!({
        "model": "dall-e-3",
        "prompt": "A cute baby sea otter",
        "n": 1,
        "size": "1024x1024",
        "response_format": "b64_json"
    });

    assert_eq!(request_body, expected);
}
