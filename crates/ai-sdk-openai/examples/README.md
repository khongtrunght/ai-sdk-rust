# Examples

This directory contains examples demonstrating how to use the OpenAI provider.

## Prerequisites

All examples require an OpenAI API key. Set it as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Running Examples

```bash
# Chat completion
cargo run --example openai_basic

# Text embeddings
cargo run --example embedding_basic

# Image generation (DALL-E)
cargo run --example image_basic

# Text-to-speech
cargo run --example speech_basic

# Speech-to-text transcription
cargo run --example transcription_basic
```

## Examples Overview

- **openai_basic.rs** - Chat completion with GPT models
- **embedding_basic.rs** - Generate text embeddings
- **image_basic.rs** - Generate images with DALL-E
- **speech_basic.rs** - Convert text to speech
- **transcription_basic.rs** - Transcribe audio to text with Whisper

## API Key Security

Never commit your API key to version control. Use environment variables or a
`.env` file (which should be in `.gitignore`).
