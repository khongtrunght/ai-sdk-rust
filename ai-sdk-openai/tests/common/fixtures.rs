use std::fs;
use std::path::Path;

/// Loads a JSON fixture from the chat fixtures directory
///
/// # Arguments
/// * `filename` - The fixture name without extension (e.g., "chat-completion-simple-1")
#[allow(dead_code)]
pub fn load_json_fixture(filename: &str) -> serde_json::Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/chat/fixtures")
        .join(format!("{}.json", filename));

    let content = fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Failed to read fixture: {}", path.display()));

    serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("Failed to parse JSON fixture at {}: {}", path.display(), e))
}

/// Loads a chunks fixture (streaming responses) from the chat fixtures directory
///
/// # Arguments
/// * `filename` - The fixture name without extension (e.g., "chat-completion-simple-1")
#[allow(dead_code)]
pub fn load_chunks_fixture(filename: &str) -> Vec<String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/chat/fixtures")
        .join(format!("{}-chunks.txt", filename));

    let content = fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Failed to read chunks fixture: {}", path.display()));

    // Each line is a JSON event - add SSE formatting
    let mut chunks: Vec<String> = content
        .lines()
        .filter(|line| !line.trim().is_empty()) // Skip empty lines
        .map(|line| format!("data: {}\n\n", line))
        .collect();

    // Add SSE terminator
    chunks.push("data: [DONE]\n\n".to_string());

    chunks
}
