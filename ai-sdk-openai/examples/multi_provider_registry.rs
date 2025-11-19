//! Example of using the provider registry with multiple providers.
//!
//! This example demonstrates:
//! - Creating a provider registry
//! - Registering multiple providers
//! - Accessing models via composite IDs
//! - Using custom providers

use ai_sdk_core::registry::{create_provider_registry, CustomProviderBuilder, ProviderRegistry};
use ai_sdk_openai::OpenAIProvider;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create OpenAI provider
    // Note: In real use, get API key from environment variable
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "test-key".to_string());

    let openai = Arc::new(OpenAIProvider::new(api_key));

    // Create a custom provider that combines specific models
    // This could override specific models or add new ones
    let custom = CustomProviderBuilder::new()
        .fallback_provider(openai.clone())
        .build();

    // Create registry with multiple providers
    let registry = create_provider_registry(Some(":"))
        .with_provider("openai", openai)
        .with_provider("custom", custom)
        .build();

    // List available providers
    println!("Available providers: {:?}", registry.list_providers());

    // Access models via composite IDs (provider:model)
    println!("\nAccessing models via registry:");

    // Language model from OpenAI provider
    match registry.language_model("openai:gpt-4") {
        Ok(model) => {
            println!("✓ Successfully retrieved openai:gpt-4");
            println!("  Provider: {}", model.provider());
            println!("  Model ID: {}", model.model_id());
        }
        Err(e) => println!("✗ Error: {}", e),
    }

    // Embedding model from OpenAI provider
    match registry.text_embedding_model("openai:text-embedding-3-small") {
        Ok(model) => {
            println!("✓ Successfully retrieved openai:text-embedding-3-small");
            println!("  Provider: {}", model.provider());
            println!("  Model ID: {}", model.model_id());
        }
        Err(e) => println!("✗ Error: {}", e),
    }

    // Image model from OpenAI provider
    match registry.image_model("openai:dall-e-3") {
        Ok(model) => {
            println!("✓ Successfully retrieved openai:dall-e-3");
            println!("  Provider: {}", model.provider());
            println!("  Model ID: {}", model.model_id());
        }
        Err(e) => println!("✗ Error: {}", e),
    }

    // Model from custom provider (falls back to OpenAI)
    match registry.language_model("custom:gpt-3.5-turbo") {
        Ok(model) => {
            println!("✓ Successfully retrieved custom:gpt-3.5-turbo (via fallback)");
            println!("  Provider: {}", model.provider());
            println!("  Model ID: {}", model.model_id());
        }
        Err(e) => println!("✗ Error: {}", e),
    }

    // Error cases
    println!("\nError handling:");

    // Invalid provider
    match registry.language_model("nonexistent:gpt-4") {
        Ok(_) => println!("Unexpected success"),
        Err(e) => println!("✓ Expected error for invalid provider: {}", e),
    }

    // Invalid model
    match registry.language_model("openai:invalid-model") {
        Ok(_) => println!("Unexpected success"),
        Err(e) => println!("✓ Expected error for invalid model: {}", e),
    }

    // Invalid format (missing separator)
    match registry.language_model("gpt-4") {
        Ok(_) => println!("Unexpected success"),
        Err(e) => println!("✓ Expected error for invalid format: {}", e),
    }

    println!("\n✅ Registry example completed successfully!");

    Ok(())
}
