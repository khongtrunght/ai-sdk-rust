use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Test server wrapper around wiremock::MockServer
///
/// Provides convenient methods for mocking OpenAI API endpoints
pub struct TestServer {
    pub server: MockServer,
    pub base_url: String,
}

impl TestServer {
    /// Creates a new test server with random port
    pub async fn new() -> Self {
        let server = MockServer::start().await;
        let base_url = server.uri();

        Self { server, base_url }
    }

    /// Mocks a JSON response for a POST endpoint
    ///
    /// # Arguments
    /// * `endpoint` - The API endpoint path (e.g., "/v1/chat/completions")
    /// * `response_body` - The JSON response body
    pub async fn mock_json_response(&self, endpoint: &str, response_body: serde_json::Value) {
        Mock::given(method("POST"))
            .and(path(endpoint))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(response_body)
                    .insert_header("content-type", "application/json"),
            )
            .mount(&self.server)
            .await;
    }

    /// Mocks a streaming SSE response for a POST endpoint
    ///
    /// # Arguments
    /// * `endpoint` - The API endpoint path (e.g., "/v1/chat/completions")
    /// * `chunks` - SSE-formatted chunks (from load_chunks_fixture)
    #[allow(dead_code)]
    pub async fn mock_streaming_response(&self, endpoint: &str, chunks: Vec<String>) {
        let body = chunks.join("");

        Mock::given(method("POST"))
            .and(path(endpoint))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(body)
                    .insert_header("content-type", "text/event-stream")
                    .insert_header("cache-control", "no-cache")
                    .insert_header("connection", "keep-alive"),
            )
            .mount(&self.server)
            .await;
    }

    /// Mocks an error response
    ///
    /// # Arguments
    /// * `endpoint` - The API endpoint path
    /// * `status_code` - HTTP status code (e.g., 400, 500)
    /// * `error_body` - Optional error response body
    #[allow(dead_code)]
    pub async fn mock_error_response(
        &self,
        endpoint: &str,
        status_code: u16,
        error_body: Option<serde_json::Value>,
    ) {
        let mut response = ResponseTemplate::new(status_code);

        if let Some(body) = error_body {
            response = response.set_body_json(body);
        }

        Mock::given(method("POST"))
            .and(path(endpoint))
            .respond_with(response)
            .mount(&self.server)
            .await;
    }

    /// Gets the last received request's body as JSON
    ///
    /// # Returns
    /// The last request body as a serde_json::Value, or None if no requests were received
    #[allow(dead_code)]
    pub async fn last_request_body(&self) -> Option<serde_json::Value> {
        let requests = self.server.received_requests().await?;
        let last_request = requests.last()?;
        serde_json::from_slice(&last_request.body).ok()
    }
}
