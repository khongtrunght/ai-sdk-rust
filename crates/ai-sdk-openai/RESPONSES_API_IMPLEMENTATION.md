# OpenAI Responses API Implementation

## Overview

This document describes the basic skeleton implementation of the OpenAI Responses API for the Rust AI SDK. The implementation follows Option A: a foundational structure that can be expanded incrementally.

## Structure

The Responses API implementation is organized in the `src/responses/` directory:

```
src/responses/
├── mod.rs          # Module exports and model ID constants
├── model.rs        # OpenAIResponsesLanguageModel implementation
├── options.rs      # Provider-specific options (OpenAIResponsesProviderOptions)
└── api_types.rs    # API request/response types
```

## Core Components

### 1. Model (`model.rs`)

**OpenAIResponsesLanguageModel** - Main model implementation

- Implements the `LanguageModel` trait from `ai-sdk-provider`
- Constructor: `new(model_id, config: OpenAIConfig)`
- Methods implemented:
  - ✅ `do_generate` - Non-streaming text generation
  - ⚠️ `do_stream` - Streaming (skeleton only, returns "not implemented" error)

**Key Features:**
- Converts standard AI SDK messages to Responses API format
- Handles reasoning models (o1, o3, o4-mini) with special logic
- Supports provider-specific options via `OpenAIResponsesProviderOptions`
- Validates and warns about unsupported options
- Parses annotations (citations) as source parts
- Uses shared `OpenAIConfig` pattern consistent with other models

### 2. Options (`options.rs`)

**OpenAIResponsesProviderOptions** - Provider-specific configuration

Supported options:
- `conversation` - Conversation ID for continuing conversations
- `include` - Additional response data to include
- `instructions` - System instructions
- `logprobs` - Log probabilities (boolean or number 1-20)
- `max_tool_calls` - Maximum tool calls allowed
- `metadata` - Request metadata
- `parallel_tool_calls` - Enable parallel tool calling
- `previous_response_id` - Continue from previous response
- `prompt_cache_key` - Manual prompt caching control
- `prompt_cache_retention` - Cache retention policy ('in_memory' or '24h')
- `reasoning_effort` - Reasoning effort for reasoning models
- `reasoning_summary` - Reasoning summary format
- `safety_identifier` - User monitoring identifier
- `service_tier` - Service tier ('auto', 'flex', 'priority', 'default')
- `store` - Whether to store the response
- `strict_json_schema` - Strict JSON schema validation
- `text_verbosity` - Text verbosity level
- `truncation` - Truncation strategy
- `user` - End-user identifier

### 3. API Types (`api_types.rs`)

Complete type definitions for the Responses API:

**Request Types:**
- `ResponsesRequest` - Main request structure
- `ResponsesInputItem` - Input messages or item references
- `ResponsesMessage` - Message with role and content
- `ResponsesContent` - Text or multimodal content
- `ResponsesContentPart` - Content parts (input/output text)

**Response Types:**
- `ResponsesResponse` - Main response structure
- `ResponsesOutputItem` - Output items (messages, function calls, reasoning)
- `ResponsesOutputContentPart` - Output content parts
- `ResponsesAnnotation` - Citations (URL or file)
- `ResponsesError` - Error information
- `ResponsesUsage` - Token usage with detailed breakdown
- `ResponsesIncompleteDetails` - Information about incomplete responses

**Streaming Types:**
- `ResponsesChunk` - Streaming event types
- `OutputItemData` - Streaming output items
- `ResponseCreatedData` - Response creation event
- `ResponseCompletedData` - Response completion event

### 4. Model Constants (`mod.rs`)

Exported constants for all OpenAI Responses API models:

**Reasoning Models:**
- `O1`, `O1_2024_12_17`
- `O3`, `O3_2025_04_16`
- `O3_MINI`, `O3_MINI_2025_01_31`
- `O4_MINI`, `O4_MINI_2025_04_16`

**GPT-5 Series:**
- `GPT_5`, `GPT_5_2025_08_07`
- `GPT_5_MINI`, `GPT_5_MINI_2025_08_07`
- `GPT_5_NANO`, `GPT_5_NANO_2025_08_07`
- `GPT_5_CHAT_LATEST`
- `GPT_5_1`, `GPT_5_1_CHAT_LATEST`

**GPT-4.1 Series:**
- `GPT_4_1`, `GPT_4_1_2025_04_14`
- `GPT_4_1_MINI`, `GPT_4_1_MINI_2025_04_14`
- `GPT_4_1_NANO`, `GPT_4_1_NANO_2025_04_14`

**GPT-4O Series:**
- `GPT_4O`, `GPT_4O_2024_05_13`, `GPT_4O_2024_08_06`, `GPT_4O_2024_11_20`
- `GPT_4O_MINI`, `GPT_4O_MINI_2024_07_18`

**GPT-4 Series:**
- `GPT_4_TURBO`, `GPT_4_TURBO_2024_04_09`
- `GPT_4`, `GPT_4_0613`
- `GPT_4_5_PREVIEW`, `GPT_4_5_PREVIEW_2025_02_27`

**GPT-3.5 Series:**
- `GPT_3_5_TURBO`, `GPT_3_5_TURBO_0125`, `GPT_3_5_TURBO_1106`

**ChatGPT:**
- `CHATGPT_4O_LATEST`

## Usage Example

```rust
use ai_sdk_openai::responses::{OpenAIResponsesLanguageModel, GPT_4O};
use ai_sdk_openai::{OpenAIConfig, OpenAIUrlOptions};
use ai_sdk_provider::language_model::{CallOptions, LanguageModel, Message, UserContentPart};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let api_key = std::env::var("OPENAI_API_KEY")?;

    // Create configuration
    let config = OpenAIConfig {
        provider: "openai".to_string(),
        url: Arc::new(|opts: OpenAIUrlOptions| {
            format!("https://api.openai.com/v1{}", opts.path)
        }),
        headers: Arc::new(move || {
            let mut headers = std::collections::HashMap::new();
            headers.insert("Authorization".to_string(), format!("Bearer {}", api_key));
            headers
        }),
        generate_id: None,
        file_id_prefixes: Some(vec!["file-".to_string()]),
    };

    // Create model
    let model = OpenAIResponsesLanguageModel::new(GPT_4O, config);

    // Generate response
    let options = CallOptions {
        prompt: vec![Message::User {
            content: vec![UserContentPart::Text {
                text: "Hello!".to_string(),
            }],
        }],
        ..Default::default()
    };

    let response = model.do_generate(options).await?;
    println!("Response: {:?}", response.content);

    Ok(())
}
```

## Implementation Status

### ✅ Completed (Basic Skeleton)

1. **Core Structure**
   - Module organization following established patterns
   - Model ID constants for all Responses API models
   - Basic type definitions for requests and responses

2. **Non-Streaming Generation**
   - `do_generate` method fully implemented
   - Message format conversion
   - Provider options parsing
   - Reasoning model detection and handling
   - Error handling and validation
   - Warning generation for unsupported options

3. **API Types**
   - Complete request/response type definitions
   - Serialization/deserialization support
   - Streaming chunk types (for future use)

4. **Configuration**
   - Uses shared `OpenAIConfig` pattern
   - Header merging via `merge_headers_reqwest` utility
   - Consistent with other model implementations

5. **Documentation**
   - Model constants documented
   - Example code provided
   - Compiles without errors

### ⚠️ TODO (Future Enhancements)

1. **Streaming Support**
   - Implement `do_stream` method
   - Parse and emit streaming chunks
   - Handle delta updates for text/reasoning
   - Tool call streaming support

2. **Multimodal Support**
   - Image content in messages
   - File attachments
   - Audio content (for GPT-4O with audio)
   - File ID detection and handling

3. **Tool Calling**
   - Tool definition conversion
   - Tool call parsing and execution
   - Tool result handling
   - Parallel tool call support

4. **Advanced Features**
   - Conversation continuation
   - Item references for multi-turn conversations
   - Reasoning token tracking
   - Cached token tracking
   - Citation extraction and formatting

5. **Testing**
   - Unit tests for type conversions
   - Integration tests with OpenAI API
   - Mock tests for error scenarios
   - Streaming tests

6. **Optimization**
   - Response streaming performance
   - Memory usage optimization
   - Better error messages
   - Retry logic for transient errors

## Design Decisions

1. **Shared Configuration**: Uses the same `OpenAIConfig` struct as other models (chat, transcription, etc.) for consistency.

2. **Warning System**: Uses the simple `CallWarning { message }` struct from the provider crate for unsupported options.

3. **Reasoning Model Detection**: Leverages existing `is_reasoning_model` utility to handle model-specific behavior.

4. **Error Handling**: Returns `Box<dyn std::error::Error + Send + Sync>` for flexibility in error types.

5. **Incremental Implementation**: Provides a working skeleton that can be enhanced incrementally without breaking changes.

## Integration

The Responses API is integrated into the OpenAI provider (`src/provider.rs`) and is available through:

```rust
pub mod responses;  // In lib.rs
```

The model is used in the provider's `get_language_model` method when appropriate.

## Next Steps

To expand this implementation, consider:

1. **Streaming First**: Implement `do_stream` as it's a core feature
2. **Tool Support**: Add tool calling for interactive workflows
3. **Multimodal**: Add image/file support for GPT-4O
4. **Testing**: Add comprehensive tests
5. **Documentation**: Add more examples and use cases

## References

- TypeScript Implementation: `/Users/khongtrunght/work/intel_internet/workspace/21_4/ai/packages/openai/src/responses`
- OpenAI API Docs: https://platform.openai.com/docs/api-reference/responses

