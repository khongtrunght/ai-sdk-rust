# Chat Model Test Fixtures

This directory contains fixture files for testing the OpenAI Chat Model without making real API calls.

## Fixture Format

### JSON Fixtures (*.json)
Complete API response objects for non-streaming tests. Each file contains a full OpenAI Chat Completion API response.

Example usage:
```rust
let fixture = load_json_fixture("chat-completion-simple-1");
test_server.mock_json_response("/v1/chat/completions", fixture).await;
```

### Chunks Fixtures (*-chunks.txt)
Streaming response events, one JSON object per line. Used for testing streaming endpoints.

Format: Each line is a Server-Sent Event JSON object (without "data:" prefix).

Example usage:
```rust
let chunks = load_chunks_fixture("chat-completion-simple-1");
test_server.mock_streaming_response("/v1/chat/completions", chunks).await;
```

## Fixture List

### Basic Chat Completion
- `chat-completion-simple-1.json` - Simple text response
- `chat-completion-simple-1-chunks.txt` - Streaming version

### Tool Calling
- `chat-tool-calling-1.json` - Single tool call (get_weather)
- `chat-tool-calling-1-chunks.txt` - Streaming version
- `chat-tool-calling-calculate-1.json` - Calculator tool
- `chat-tool-calling-calculate-1-chunks.txt` - Streaming version
- `chat-multiple-tools-1.json` - Multiple tool calls (weather + time)
- `chat-no-tool-call-1.json` - No tool call despite tools available

### Azure/Router
- `azure-model-router-1-chunks.txt` - Azure model router with content filtering (from TypeScript)

## Adding New Fixtures

To add a new fixture:

1. Capture a real API response (sanitize any sensitive data)
2. Save JSON response to `<name>.json`
3. For streaming: save events to `<name>-chunks.txt` (one JSON per line, no "data:" prefix)
4. Update this README with fixture description
5. Follow naming convention: `chat-<category>-<scenario>-<variant>.json`

## Naming Convention

- Prefix: Always `chat-` for chat model fixtures
- Category: `completion`, `tool-calling`, `settings`, `streaming`, `error`, etc.
- Scenario: Descriptive name (e.g., `temperature`, `multiple-tools`, `json-schema`)
- Variant: Numeric suffix for multiple variants (e.g., `-1`, `-2`)
- Examples:
  - `chat-settings-temperature-1.json`
  - `chat-streaming-tool-delta-1-chunks.txt`
  - `chat-error-rate-limit-1.json`

## Fixture Sources

- **TypeScript:** Copied from `packages/openai/src/chat/__fixtures__/`
- **Created:** New fixtures created for Rust-specific test scenarios
- **Captured:** Real API responses (sanitized) for comprehensive coverage

## Test Coverage Summary

The fixtures support 67 tests across 8 test modules:
- `basic_test.rs` - 9 tests (basic generation, usage, metadata)
- `settings_test.rs` - 8 tests (settings propagation)
- `response_format_test.rs` - 9 tests (JSON schema, structured outputs)
- `tool_calling_test.rs` - 4 tests (function calling)
- `model_specific_test.rs` - 4 tests (o1/o3/o4 models, model detection)
- `extension_settings_test.rs` - 8 tests (OpenAI extensions)
- `advanced_features_test.rs` - 5 tests (annotations, tokens)
- `streaming_test.rs` - 19 tests (streaming scenarios)

Status: 60 tests passing, 7 tests ignored (pending feature implementation)
