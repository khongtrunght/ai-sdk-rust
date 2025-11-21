# AI SDK for Rust

[![Crates.io](https://img.shields.io/crates/v/ai-sdk-provider)](https://crates.io/crates/ai-sdk-provider)
[![Documentation](https://docs.rs/ai-sdk-provider/badge.svg)](https://docs.rs/ai-sdk-provider)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)
[![CI](https://github.com/khongtrunght/ai-sdk-rust/workflows/CI/badge.svg)](https://github.com/khongtrunght/ai-sdk-rust/actions)
[![Security Audit](https://github.com/khongtrunght/ai-sdk-rust/workflows/Security%20Audit/badge.svg)](https://github.com/khongtrunght/ai-sdk-rust/actions)
[![Rust Version](https://img.shields.io/badge/rust-1.70%2B-blue.svg)](https://www.rust-lang.org)

Rust implementation of the AI SDK, providing a unified interface for interacting with various AI model providers.

## Overview

This workspace contains:

- **`ai-sdk-provider`** - Core provider specification and trait definitions
- **`ai-sdk-openai`** - OpenAI implementation (GPT, DALL-E, Whisper, Embeddings)

## Features

- 🤖 **Language Models** - Chat completion and text generation
- 🔢 **Embeddings** - Text embedding generation
- 🖼️ **Image Generation** - DALL-E integration
- 🗣️ **Speech Synthesis** - Text-to-speech
- 👂 **Transcription** - Speech-to-text with Whisper
- 📊 **Reranking** - Document reranking interface

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
ai-sdk-provider = "0.1"
ai-sdk-openai = "0.1"
```

### Example: Chat Completion

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

## Documentation

- [API Documentation](https://docs.rs/ai-sdk-provider)
- [Examples](./crates/ai-sdk-openai/examples/)
- [Contributing Guide](./CONTRIBUTING.md)

## Model Support

### Language Models
- OpenAI GPT (GPT-4, GPT-3.5)

### Embedding Models
- OpenAI text-embedding-3-small
- OpenAI text-embedding-3-large

### Image Models
- OpenAI DALL-E 2
- OpenAI DALL-E 3

### Speech & Transcription
- OpenAI TTS (text-to-speech)
- OpenAI Whisper (transcription)

## Development

### Prerequisites

- Rust 1.70 or later
- An OpenAI API key for testing

### Building

```bash
cargo build --workspace
```

### Running Tests

```bash
# Unit tests (no API key required)
cargo test --workspace

# Integration tests (requires OPENAI_API_KEY)
cargo test --workspace --features integration-tests
```

### Code Quality

```bash
cargo fmt --check        # Check formatting
cargo clippy --workspace # Run linter
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Security

For security vulnerabilities, please see [SECURITY.md](./SECURITY.md).

## Acknowledgments

This project is inspired by the [Vercel AI SDK](https://github.com/vercel/ai) and aims to bring similar functionality to the Rust ecosystem.

