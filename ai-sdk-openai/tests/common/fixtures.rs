use std::fs;
use std::path::{Path, PathBuf};

/// Returns the path to the fixtures directory
fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

/// Loads a JSON fixture file and parses it into a serde_json::Value
///
/// # Arguments
/// * `filename` - The fixture name without extension (e.g., "openai-web-search-tool-1")
///
/// # Panics
/// Panics if the file doesn't exist or contains invalid JSON
pub fn load_json_fixture(filename: &str) -> serde_json::Value {
    let path = fixtures_dir().join(format!("{}.json", filename));
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Failed to read fixture: {}", path.display()));

    serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("Failed to parse fixture JSON at {}: {}", path.display(), e))
}

/// Loads a streaming chunks fixture and formats it as SSE events
///
/// # Arguments
/// * `filename` - The fixture name without extension (e.g., "openai-web-search-tool-1")
///
/// # Returns
/// Vector of SSE-formatted strings (includes "data: " prefix and terminator)
///
/// # Panics
/// Panics if the file doesn't exist
#[allow(dead_code)]
pub fn load_chunks_fixture(filename: &str) -> Vec<String> {
    let path = fixtures_dir().join(format!("{}-chunks.txt", filename));
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Failed to read fixture: {}", path.display()));

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
