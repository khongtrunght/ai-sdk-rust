# AI SDK OpenAI - Rust Implementation

[![Crates.io](https://img.shields.io/crates/v/ai-sdk-openai)](https://crates.io/crates/ai-sdk-openai)
[![Documentation](https://docs.rs/ai-sdk-openai/badge.svg)](https://docs.rs/ai-sdk-openai)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/khongtrunght/ai-sdk-rust/blob/main/LICENSE-MIT)

OpenAI provider implementation for the AI SDK in Rust. This crate provides a unified interface for interacting with OpenAI's models including GPT, DALL-E, Whisper, and Embeddings.

## Features

- 🤖 **Chat Completion** - GPT-4, GPT-3.5 and other chat models
- 🔢 **Embeddings** - text-embedding-3-small, text-embedding-3-large
- 🖼️ **Image Generation** - DALL-E 2 and DALL-E 3
- 🗣️ **Speech Synthesis** - Text-to-speech with OpenAI TTS
- 👂 **Transcription** - Speech-to-text with Whisper
- 🔄 **Streaming** - Stream responses for real-time applications
- 🛠️ **Tool Calling** - Function calling support for chat models

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
ai-sdk-openai = "0.1"
ai-sdk-provider = "0.1"
```

## Quick Start

### Chat Completion

```rust
use ai_sdk_openai::OpenAIChatModel;
use ai_sdk_provider::{LanguageModel, CallOptions, Message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY")?;
    let model = OpenAIChatModel::new("gpt-4", api_key);

    let options = CallOptions {
        messages: vec![
            Message::user("What is Rust?"),
        ],
        ..Default::default()
    };

    let response = model.do_generate(options).await?;
    println!("Response: {}", response.text);

    Ok(())
}
```

### Embeddings

```rust
use ai_sdk_openai::OpenAIEmbeddingModel;
use ai_sdk_provider::EmbeddingModel;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY")?;
    let model = OpenAIEmbeddingModel::new("text-embedding-3-small", api_key);

    let embeddings = model.do_embed(vec!["Hello, world!".to_string()]).await?;
    println!("Embedding dimension: {}", embeddings[0].len());

    Ok(())
}
```

### Image Generation

```rust
use ai_sdk_openai::OpenAIImageModel;
use ai_sdk_provider::{ImageModel, ImageOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY")?;
    let model = OpenAIImageModel::new("dall-e-3", api_key);

    let options = ImageOptions {
        prompt: "A beautiful sunset over mountains".to_string(),
        ..Default::default()
    };

    let result = model.do_generate(options).await?;
    println!("Generated image URL: {}", result.url);

    Ok(())
}
```

## Supported Models

### Chat Models
- `gpt-4` - Most capable GPT-4 model
- `gpt-4-turbo` - GPT-4 Turbo with improved performance
- `gpt-3.5-turbo` - Fast and efficient model

### Embedding Models
- `text-embedding-3-small` - Smaller, faster embedding model
- `text-embedding-3-large` - Larger, more accurate embedding model

### Image Models
- `dall-e-2` - DALL-E 2 image generation
- `dall-e-3` - DALL-E 3 with improved quality

### Speech Models
- `tts-1` - Standard text-to-speech
- `tts-1-hd` - High-definition text-to-speech
- `whisper-1` - Speech-to-text transcription

## Documentation

- [API Documentation](https://docs.rs/ai-sdk-openai)
- [Examples](https://github.com/khongtrunght/ai-sdk-rust/tree/main/ai-sdk-openai/examples)
- [AI SDK Provider Traits](https://docs.rs/ai-sdk-provider)

## Repository

This crate is part of the [AI SDK for Rust](https://github.com/khongtrunght/ai-sdk-rust) workspace.

## Contributing

We welcome contributions! Please see the [Contributing Guide](https://github.com/khongtrunght/ai-sdk-rust/blob/main/CONTRIBUTING.md) for guidelines.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/khongtrunght/ai-sdk-rust/blob/main/LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](https://github.com/khongtrunght/ai-sdk-rust/blob/main/LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
