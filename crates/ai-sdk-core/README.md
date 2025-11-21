# AI SDK Core - High-Level APIs for Rust

[![Crates.io](https://img.shields.io/crates/v/ai-sdk-core)](https://crates.io/crates/ai-sdk-core)
[![Documentation](https://docs.rs/ai-sdk-core/badge.svg)](https://docs.rs/ai-sdk-core)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/khongtrunght/ai-sdk-rust/blob/main/LICENSE-MIT)

High-level, ergonomic APIs for working with AI models in Rust. This crate provides simple, composable functions for text generation, streaming, embeddings, and tool calling.

## Features

- 🤖 **Text Generation** - Simple `generate_text()` API with builder pattern
- 🌊 **Streaming** - Real-time streaming with `stream_text()`
- 🔧 **Tool Calling** - Automatic tool execution loops with retry logic
- 🔢 **Embeddings** - Generate embeddings with `embed()` and `embed_many()`
- ⚡ **Async First** - Built on tokio for high-performance async operations
- 🔄 **Retry Logic** - Built-in exponential backoff for resilient operations
- 📊 **Stop Conditions** - Control generation with max tokens, tool calls, etc.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
ai-sdk-core = "0.1"
ai-sdk-openai = "0.1"  # Or any other provider
ai-sdk-provider = "0.1"
```

## Quick Start

### Text Generation

```rust
use ai_sdk_core::generate_text;
use ai_sdk_openai::openai;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY")?;

    let result = generate_text()
        .model(openai("gpt-4").api_key(api_key))
        .prompt("Explain quantum computing in simple terms")
        .execute()
        .await?;

    println!("Response: {}", result.text());
    println!("Usage: {:?}", result.usage());

    Ok(())
}
```

### Streaming Text

```rust
use ai_sdk_core::stream_text;
use ai_sdk_openai::openai;
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY")?;

    let mut stream = stream_text()
        .model(openai("gpt-4").api_key(api_key))
        .prompt("Write a haiku about Rust")
        .execute()
        .await?;

    while let Some(chunk) = stream.next().await {
        match chunk? {
            ai_sdk_core::StreamPart::TextDelta(text) => {
                print!("{}", text);
            }
            ai_sdk_core::StreamPart::FinishReason(reason) => {
                println!("\n\nFinished: {:?}", reason);
            }
            _ => {}
        }
    }

    Ok(())
}
```

### Tool Calling

```rust
use ai_sdk_core::{generate_text, Tool, ToolContext, ToolError};
use ai_sdk_openai::openai;
use async_trait::async_trait;
use std::sync::Arc;

struct WeatherTool;

#[async_trait]
impl Tool for WeatherTool {
    fn name(&self) -> &str {
        "get_weather"
    }

    fn description(&self) -> &str {
        "Get the current weather for a location"
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                }
            },
            "required": ["location"]
        })
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Result<serde_json::Value, ToolError> {
        let location = input["location"].as_str().unwrap_or("Unknown");
        Ok(serde_json::json!({
            "location": location,
            "temperature": 72,
            "condition": "Sunny"
        }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY")?;

    let result = generate_text()
        .model(openai("gpt-4").api_key(api_key))
        .prompt("What's the weather in San Francisco?")
        .tools(vec![Arc::new(WeatherTool)])
        .max_tool_calls(5)
        .execute()
        .await?;

    println!("Response: {}", result.text());
    println!("Tool calls made: {}", result.tool_calls().len());

    Ok(())
}
```

### Embeddings

```rust
use ai_sdk_core::embed;
use ai_sdk_openai::openai_embedding;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY")?;

    let result = embed()
        .model(openai_embedding("text-embedding-3-small").api_key(api_key))
        .value("Hello, world!")
        .execute()
        .await?;

    println!("Embedding dimension: {}", result.embedding().len());

    Ok(())
}
```

### Batch Embeddings

```rust
use ai_sdk_core::embed_many;
use ai_sdk_openai::openai_embedding;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY")?;

    let texts = vec![
        "First document",
        "Second document",
        "Third document",
    ];

    let result = embed_many()
        .model(openai_embedding("text-embedding-3-small").api_key(api_key))
        .values(texts)
        .execute()
        .await?;

    for (i, embedding) in result.embeddings().iter().enumerate() {
        println!("Document {}: {} dimensions", i, embedding.len());
    }

    Ok(())
}
```

## Advanced Features

### Stop Conditions

Control when text generation should stop:

```rust
use ai_sdk_core::{generate_text, StopCondition};
use ai_sdk_openai::openai;

let result = generate_text()
    .model(openai("gpt-4").api_key(api_key))
    .prompt("Count from 1 to 100")
    .stop_condition(StopCondition::MaxTokens(50))
    .execute()
    .await?;
```

### Retry Configuration

Configure automatic retries with exponential backoff:

```rust
use ai_sdk_core::{generate_text, RetryConfig};
use ai_sdk_openai::openai;
use std::time::Duration;

let result = generate_text()
    .model(openai("gpt-4").api_key(api_key))
    .prompt("Hello")
    .retry(RetryConfig {
        max_retries: 3,
        initial_delay: Duration::from_secs(1),
        max_delay: Duration::from_secs(10),
    })
    .execute()
    .await?;
```

## Architecture

This crate is part of the [AI SDK for Rust](https://github.com/khongtrunght/ai-sdk-rust) workspace:

- **ai-sdk-provider** - Core traits and types for providers
- **ai-sdk-core** - High-level APIs (this crate)
- **ai-sdk-openai** - OpenAI implementation

## Documentation

- [API Documentation](https://docs.rs/ai-sdk-core)
- [AI SDK Provider Traits](https://docs.rs/ai-sdk-provider)
- [Examples](https://github.com/khongtrunght/ai-sdk-rust/tree/main/ai-sdk-core/examples)

## Repository

This crate is part of the [AI SDK for Rust](https://github.com/khongtrunght/ai-sdk-rust) workspace.

## Contributing

We welcome contributions! Please see the [Contributing Guide](https://github.com/khongtrunght/ai-sdk-rust/blob/main/CONTRIBUTING.md) for guidelines.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/khongtrunght/ai-sdk-rust/blob/main/LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](https://github.com/khongtrunght/ai-sdk-rust/blob/main/LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
