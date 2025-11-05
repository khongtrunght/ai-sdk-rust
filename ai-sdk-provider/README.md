# AI SDK Provider Specification (v3)

This crate defines the provider interface specification that all AI model providers must implement.

## Overview

The AI SDK Provider crate provides trait-based interfaces for different types of AI models:

- **Language Models** - Text generation and chat completion
- **Embedding Models** - Text embedding generation
- **Image Models** - Image generation
- **Speech Models** - Text-to-speech synthesis
- **Transcription Models** - Speech-to-text transcription
- **Reranking Models** - Document reranking by relevance

All model interfaces follow the v3 specification pattern with consistent error handling, metadata support, and provider-specific options.

## Model Types

### Language Models

The `LanguageModel` trait provides an interface for text generation and chat models.

```rust
use ai_sdk_provider::{LanguageModel, CallOptions};

// Use with any provider implementation
async fn generate_text<M: LanguageModel>(model: &M, prompt: &str) {
    let options = CallOptions { /* ... */ };
    let response = model.do_generate(options).await?;
    println!("Generated: {}", response.text);
}
```

### Embedding Models

The `EmbeddingModel` trait provides an interface for generating text embeddings.

### Image Models

The `ImageModel` trait provides an interface for image generation models like DALL-E.

### Speech Models

The `SpeechModel` trait provides an interface for text-to-speech synthesis.

### Transcription Models

The `TranscriptionModel` trait provides an interface for speech-to-text transcription.

### Reranking Models

The `RerankingModel` trait provides an interface for document reranking.
Reranking models take a query and a list of documents and reorder them by
relevance.

Currently, no providers are implemented in the `ai-sdk-provider` crate.
Provider implementations should be added in separate crates.

#### Example

See `examples/reranking_trait.rs` for a complete example of implementing
the `RerankingModel` trait.

```rust
use ai_sdk_provider::{RerankingModel, RerankOptions, Documents};

async fn rerank_documents<M: RerankingModel>(
    model: &M,
    query: &str,
    docs: Vec<String>
) -> Result<Vec<(usize, f64)>, Box<dyn std::error::Error>> {
    let options = RerankOptions {
        documents: Documents::Text { values: docs },
        query: query.to_string(),
        top_n: Some(5),
        abort_signal: None,
        provider_options: None,
        headers: None,
    };

    let response = model.do_rerank(options).await?;

    Ok(response.ranking.into_iter()
        .map(|r| (r.index, r.relevance_score))
        .collect())
}
```

## Provider Implementations

Provider implementations are maintained in separate crates:

- `ai-sdk-openai` - OpenAI provider (Language, Embedding, Image, Speech, Transcription)
- `ai-sdk-anthropic` - Anthropic provider (Language)
- `ai-sdk-cohere` - Cohere provider (Language, Embedding, Reranking)
- And more...

## Shared Types

The crate also provides shared types used across all model interfaces:

- `SharedHeaders` - HTTP headers
- `SharedProviderOptions` - Provider-specific input options
- `SharedProviderMetadata` - Provider-specific output metadata
- `SharedWarning` - Warnings about unsupported features or compatibility issues
- `JsonValue`, `JsonObject`, `JsonArray` - JSON value types

## Contributing

When implementing a new provider:

1. Implement the appropriate model trait(s)
2. Handle provider-specific options through `SharedProviderOptions`
3. Return provider-specific metadata through `SharedProviderMetadata`
4. Emit warnings for unsupported features using `SharedWarning`
5. Follow async/await patterns for all I/O operations

## License

See the main AI SDK repository for license information.
