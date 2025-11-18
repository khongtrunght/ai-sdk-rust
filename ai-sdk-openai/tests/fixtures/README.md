# Test Fixtures

This directory contains fixture files for testing the OpenAI provider without making real API calls.

## Fixture Format

### JSON Fixtures (*.json)
Complete API response objects for non-streaming tests. Each file contains a full OpenAI Responses API response.

Example usage:
```rust
let fixture = load_json_fixture("openai-web-search-tool-1");
test_server.mock_json_response("/v1/responses", fixture).await;
```

### Chunks Fixtures (*-chunks.txt)
Streaming response events, one JSON object per line. Used for testing streaming endpoints.

Format: Each line is a Server-Sent Event JSON object (without "data:" prefix).

Example usage:
```rust
let chunks = load_chunks_fixture("openai-web-search-tool-1");
test_server.mock_streaming_response("/v1/responses", chunks).await;
```

## Fixture List

### Web Search Tool
- `openai-web-search-tool-1.json` - Web search with multiple results
- `openai-web-search-tool-1-chunks.txt` - Streaming version

### Code Interpreter Tool
- `openai-code-interpreter-tool-1.json` - Python code execution
- `openai-code-interpreter-tool-1-chunks.txt` - Streaming version

### File Search Tool
- `openai-file-search-tool-1.json` - File search operation
- `openai-file-search-tool-1-chunks.txt` - Streaming version
- `openai-file-search-tool-2.json` - Alternative file search scenario
- `openai-file-search-tool-2-chunks.txt` - Streaming version

### Image Generation Tool
- `openai-image-generation-tool-1.json` - DALL-E image generation
- `openai-image-generation-tool-1-chunks.txt` - Streaming version

### Local Shell Tool
- `openai-local-shell-tool-1.json` - Shell command execution
- `openai-local-shell-tool-1-chunks.txt` - Streaming version

### MCP Tool
- `openai-mcp-tool-1.json` - Model Context Protocol tool usage
- `openai-mcp-tool-1-chunks.txt` - Streaming version

## Fixture Sources

These fixtures are ported from the TypeScript AI SDK at:
`packages/openai/src/responses/__fixtures__/`

## Adding New Fixtures

To add a new fixture:

1. Capture a real API response (sanitize any sensitive data)
2. Save JSON response to `<name>.json`
3. For streaming: save events to `<name>-chunks.txt` (one JSON per line, no "data:" prefix)
4. Update this README with fixture description
5. Verify with `cargo test -- --ignored test_load_json_fixture`
