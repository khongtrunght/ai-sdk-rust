/// URL download utilities
use reqwest::Client;
use thiserror::Error;

const USER_AGENT: &str = "ai-sdk-rust/0.1.0";

/// Errors that can occur during file download
#[derive(Debug, Error)]
pub enum DownloadError {
    /// HTTP request failed
    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),

    /// HTTP response returned non-success status code
    #[error("HTTP status {0}")]
    HttpStatus(u16),

    /// Invalid URL provided
    #[error("Invalid URL: {0}")]
    InvalidUrl(String),
}

/// Result of a successful file download
#[derive(Debug, Clone)]
pub struct DownloadedFile {
    /// The binary file data
    pub data: Vec<u8>,

    /// The media type from Content-Type header, if present
    pub media_type: Option<String>,
}

/// Downloads a file from a URL
///
/// # Arguments
/// * `url` - The URL to download from
///
/// # Returns
/// Downloaded file data and metadata, or error if download fails
///
/// # Example
/// ```no_run
/// use ai_sdk_core::util::download;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let file = download("https://example.com/image.jpg").await?;
///     println!("Downloaded {} bytes", file.data.len());
///     println!("Media type: {:?}", file.media_type);
///     Ok(())
/// }
/// ```
pub async fn download(url: &str) -> Result<DownloadedFile, DownloadError> {
    // Validate URL
    if url.is_empty() {
        return Err(DownloadError::InvalidUrl("Empty URL".to_string()));
    }

    // Create HTTP client
    let client = Client::builder()
        .user_agent(USER_AGENT)
        .build()
        .map_err(DownloadError::HttpError)?;

    // Make request
    let response = client.get(url).send().await?;

    // Check status
    let status = response.status();
    if !status.is_success() {
        return Err(DownloadError::HttpStatus(status.as_u16()));
    }

    // Extract media type from Content-Type header
    let media_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(|s| {
            // Extract just the media type, ignoring parameters like charset
            s.split(';').next().unwrap_or(s).trim().to_string()
        });

    // Download body
    let data = response.bytes().await?.to_vec();

    Ok(DownloadedFile { data, media_type })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_url() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(download(""));
        assert!(matches!(result, Err(DownloadError::InvalidUrl(_))));
    }

    // Note: More comprehensive tests would require mocking HTTP responses
    // or using a test HTTP server. For now, we just test basic error cases.

    #[test]
    fn test_user_agent_constant() {
        assert!(USER_AGENT.contains("ai-sdk-rust"));
    }
}
