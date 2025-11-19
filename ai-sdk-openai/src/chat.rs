use ai_sdk_provider::language_model::{
    AssistantContentPart, Message, ResponseInfo, StreamError, TextPart, ToolCallPart,
    UserContentPart,
};
use ai_sdk_provider::*;
use async_stream::stream;
use async_trait::async_trait;
use futures::stream::StreamExt;
use log::warn;
use reqwest::Client;
use std::collections::HashMap;

use crate::model_detection::{
    is_o1_model, is_reasoning_model, is_search_preview_model, supports_flex_processing,
};

/// OpenAI implementation of chat model.
pub struct OpenAIChatModel {
    model_id: String,
    api_key: String,
    client: Client,
    base_url: String,
}

impl OpenAIChatModel {
    /// Creates a new chat model with the specified model ID and API key.
    pub fn new(model_id: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            api_key: api_key.into(),
            client: Client::new(),
            base_url: "https://api.openai.com/v1".into(),
        }
    }

    /// Configures a custom base URL for the API endpoint.
    ///
    /// This is primarily useful for testing with mock servers.
    ///
    /// # Arguments
    /// * `base_url` - Custom base URL (e.g., "http://localhost:8080")
    ///
    /// # Example
    /// ```rust
    /// # use ai_sdk_openai::OpenAIChatModel;
    /// let model = OpenAIChatModel::new("gpt-4", "api-key")
    ///     .with_base_url("http://localhost:8080");
    /// ```
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    fn convert_prompt_to_messages(&self, prompt: &[Message]) -> Vec<crate::api_types::ChatMessage> {
        // Convert our prompt format to OpenAI's message format
        let mut openai_messages = Vec::new();

        for msg in prompt {
            match msg {
                Message::System { content } => {
                    // For o1 models, convert system messages to developer messages
                    let role = if is_o1_model(&self.model_id) {
                        "developer"
                    } else {
                        "system"
                    };
                    openai_messages.push(crate::api_types::ChatMessage {
                        role: role.into(),
                        content: Some(crate::api_types::ChatMessageContent::Text(content.clone())),
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
                Message::User { content } => {
                    // Check if we have any file content (multi-modal)
                    let has_files = content
                        .iter()
                        .any(|part| matches!(part, UserContentPart::File { .. }));

                    if has_files {
                        // Multi-modal message: convert to array of content parts
                        let mut openai_content = Vec::new();

                        for part in content {
                            match part {
                                UserContentPart::Text { text } => {
                                    openai_content.push(
                                        crate::multimodal::OpenAIContentPart::Text {
                                            text: text.clone(),
                                        },
                                    );
                                }
                                UserContentPart::File { data, media_type } => {
                                    // Determine file type and convert accordingly
                                    if media_type.starts_with("image/") {
                                        match crate::multimodal::convert_image_part(
                                            data, media_type,
                                        ) {
                                            Ok(part) => openai_content.push(part),
                                            Err(e) => {
                                                // Log error but continue
                                                eprintln!(
                                                    "Warning: Failed to convert image: {}",
                                                    e
                                                );
                                            }
                                        }
                                    } else if media_type.starts_with("audio/") {
                                        match crate::multimodal::convert_audio_part(
                                            data, media_type,
                                        ) {
                                            Ok(part) => openai_content.push(part),
                                            Err(e) => {
                                                eprintln!(
                                                    "Warning: Failed to convert audio: {}",
                                                    e
                                                );
                                            }
                                        }
                                    } else {
                                        eprintln!(
                                            "Warning: Unsupported media type: {}",
                                            media_type
                                        );
                                    }
                                }
                            }
                        }

                        openai_messages.push(crate::api_types::ChatMessage {
                            role: "user".into(),
                            content: Some(crate::api_types::ChatMessageContent::Parts(
                                openai_content,
                            )),
                            tool_calls: None,
                            tool_call_id: None,
                        });
                    } else {
                        // Text-only message: join all text parts
                        let text_content = content
                            .iter()
                            .filter_map(|part| match part {
                                UserContentPart::Text { text } => Some(text.clone()),
                                UserContentPart::File { .. } => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n");

                        openai_messages.push(crate::api_types::ChatMessage {
                            role: "user".into(),
                            content: Some(crate::api_types::ChatMessageContent::Text(text_content)),
                            tool_calls: None,
                            tool_call_id: None,
                        });
                    }
                }
                Message::Assistant { content } => {
                    let mut text_content = String::new();
                    let mut tool_calls = Vec::new();

                    for part in content {
                        match part {
                            AssistantContentPart::Text(text_part) => {
                                text_content.push_str(&text_part.text);
                            }
                            AssistantContentPart::ToolCall(tool_call) => {
                                tool_calls.push(crate::api_types::OpenAIToolCall {
                                    id: tool_call.tool_call_id.clone(),
                                    r#type: "function".to_string(),
                                    function: crate::api_types::OpenAIFunctionCall {
                                        name: tool_call.tool_name.clone(),
                                        arguments: tool_call.input.clone(),
                                    },
                                });
                            }
                            // Skip other content types for now
                            _ => {}
                        }
                    }

                    openai_messages.push(crate::api_types::ChatMessage {
                        role: "assistant".into(),
                        content: if text_content.is_empty() {
                            None
                        } else {
                            Some(crate::api_types::ChatMessageContent::Text(text_content))
                        },
                        tool_calls: if tool_calls.is_empty() {
                            None
                        } else {
                            Some(tool_calls)
                        },
                        tool_call_id: None,
                    });
                }
                Message::Tool { content } => {
                    // Convert tool results to OpenAI tool messages
                    for tool_result in content {
                        openai_messages.push(crate::api_types::ChatMessage {
                            role: "tool".into(),
                            content: Some(crate::api_types::ChatMessageContent::Text(
                                serde_json::to_string(&tool_result.output).unwrap_or_default(),
                            )),
                            tool_calls: None,
                            tool_call_id: Some(tool_result.tool_call_id.clone()),
                        });
                    }
                }
            }
        }

        openai_messages
    }

    fn convert_tools(&self, tools: &[language_model::Tool]) -> Vec<crate::api_types::OpenAITool> {
        tools
            .iter()
            .filter_map(|tool| match tool {
                language_model::Tool::Function(function_tool) => {
                    Some(crate::api_types::OpenAITool {
                        r#type: "function".to_string(),
                        function: crate::api_types::OpenAIFunction {
                            name: function_tool.name.clone(),
                            description: function_tool.description.clone(),
                            parameters: function_tool.input_schema.clone(),
                        },
                    })
                }
                // Skip provider-defined tools for now
                language_model::Tool::ProviderDefined(_) => None,
            })
            .collect()
    }

    fn convert_tool_choice(
        &self,
        tool_choice: &language_model::ToolChoice,
    ) -> crate::api_types::OpenAIToolChoice {
        match tool_choice {
            language_model::ToolChoice::Auto => {
                crate::api_types::OpenAIToolChoice::String("auto".to_string())
            }
            language_model::ToolChoice::None => {
                crate::api_types::OpenAIToolChoice::String("none".to_string())
            }
            language_model::ToolChoice::Required => {
                crate::api_types::OpenAIToolChoice::String("required".to_string())
            }
            language_model::ToolChoice::Tool { tool_name } => {
                crate::api_types::OpenAIToolChoice::Specific {
                    r#type: "function".to_string(),
                    function: crate::api_types::OpenAIFunctionName {
                        name: tool_name.clone(),
                    },
                }
            }
        }
    }

    fn convert_response_format(
        &self,
        response_format: &language_model::ResponseFormat,
    ) -> crate::api_types::OpenAIResponseFormat {
        match response_format {
            language_model::ResponseFormat::Text => crate::api_types::OpenAIResponseFormat::Text,
            language_model::ResponseFormat::Json {
                schema,
                name,
                description,
            } => {
                if let Some(schema) = schema {
                    // Structured output with JSON schema
                    crate::api_types::OpenAIResponseFormat::JsonSchema {
                        json_schema: crate::api_types::OpenAIJsonSchema {
                            name: name.clone().unwrap_or_else(|| "response".to_string()),
                            description: description.clone(),
                            schema: schema.clone(),
                            strict: Some(true), // Enable strict mode for better validation
                        },
                    }
                } else {
                    // Unvalidated JSON mode
                    crate::api_types::OpenAIResponseFormat::JsonObject
                }
            }
        }
    }

    fn map_finish_reason(&self, reason: Option<&str>) -> FinishReason {
        match reason {
            Some("stop") => FinishReason::Stop,
            Some("length") => FinishReason::Length,
            Some("content_filter") => FinishReason::ContentFilter,
            Some("tool_calls") => FinishReason::ToolCalls,
            _ => FinishReason::Unknown,
        }
    }

    /// Extract OpenAI-specific options from provider_options
    fn extract_openai_options(
        &self,
        provider_options: &Option<std::collections::HashMap<String, json_value::JsonObject>>,
    ) -> OpenAIOptions {
        use json_value::JsonValue;

        let mut opts = OpenAIOptions::default();

        if let Some(provider_opts) = provider_options {
            if let Some(openai_opts) = provider_opts.get("openai") {
                // logitBias: HashMap<String, f64>
                if let Some(JsonValue::Object(logit_bias)) = openai_opts.get("logitBias") {
                    let mut bias_map = HashMap::new();
                    for (k, v) in logit_bias {
                        if let JsonValue::Number(n) = v {
                            if let Some(f) = n.as_f64() {
                                bias_map.insert(k.clone(), f);
                            }
                        }
                    }
                    if !bias_map.is_empty() {
                        opts.logit_bias = Some(bias_map);
                    }
                }

                // logprobs: bool
                if let Some(JsonValue::Bool(b)) = openai_opts.get("logprobs") {
                    opts.logprobs = Some(*b);
                }

                // parallelToolCalls: bool
                if let Some(JsonValue::Bool(b)) = openai_opts.get("parallelToolCalls") {
                    opts.parallel_tool_calls = Some(*b);
                }

                // user: String
                if let Some(JsonValue::String(s)) = openai_opts.get("user") {
                    opts.user = Some(s.clone());
                }

                // reasoningEffort: String
                if let Some(JsonValue::String(s)) = openai_opts.get("reasoningEffort") {
                    opts.reasoning_effort = Some(s.clone());
                }

                // maxCompletionTokens: u32
                if let Some(JsonValue::Number(n)) = openai_opts.get("maxCompletionTokens") {
                    if let Some(u) = n.as_u64() {
                        opts.max_completion_tokens = Some(u as u32);
                    }
                }

                // store: bool
                if let Some(JsonValue::Bool(b)) = openai_opts.get("store") {
                    opts.store = Some(*b);
                }

                // metadata: HashMap<String, String>
                if let Some(JsonValue::Object(metadata)) = openai_opts.get("metadata") {
                    let mut meta_map = HashMap::new();
                    for (k, v) in metadata {
                        if let JsonValue::String(s) = v {
                            meta_map.insert(k.clone(), s.clone());
                        }
                    }
                    if !meta_map.is_empty() {
                        opts.metadata = Some(meta_map);
                    }
                }

                // prediction: JsonValue
                if let Some(v) = openai_opts.get("prediction") {
                    // Convert JsonValue to serde_json::Value
                    if let Ok(json_str) = serde_json::to_string(v) {
                        if let Ok(json_val) = serde_json::from_str(&json_str) {
                            opts.prediction = Some(json_val);
                        }
                    }
                }

                // serviceTier: String
                if let Some(JsonValue::String(s)) = openai_opts.get("serviceTier") {
                    opts.service_tier = Some(s.clone());
                }

                // textVerbosity: String
                if let Some(JsonValue::String(s)) = openai_opts.get("textVerbosity") {
                    opts.verbosity = Some(s.clone());
                }

                // promptCacheKey: String
                if let Some(JsonValue::String(s)) = openai_opts.get("promptCacheKey") {
                    opts.prompt_cache_key = Some(s.clone());
                }

                // safetyIdentifier: String
                if let Some(JsonValue::String(s)) = openai_opts.get("safetyIdentifier") {
                    opts.safety_identifier = Some(s.clone());
                }
            }
        }

        opts
    }
}

/// Struct to hold extracted OpenAI options
#[derive(Default)]
struct OpenAIOptions {
    logit_bias: Option<HashMap<String, f64>>,
    logprobs: Option<bool>,
    parallel_tool_calls: Option<bool>,
    user: Option<String>,
    reasoning_effort: Option<String>,
    max_completion_tokens: Option<u32>,
    store: Option<bool>,
    metadata: Option<HashMap<String, String>>,
    prediction: Option<serde_json::Value>,
    service_tier: Option<String>,
    verbosity: Option<String>,
    prompt_cache_key: Option<String>,
    safety_identifier: Option<String>,
}

#[async_trait]
impl LanguageModel for OpenAIChatModel {
    fn provider(&self) -> &str {
        "openai"
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn supported_urls(&self) -> HashMap<String, Vec<String>> {
        let mut urls = HashMap::new();
        urls.insert("image/*".into(), vec![r"^https?://.*$".into()]);
        urls
    }

    async fn do_generate(
        &self,
        options: CallOptions,
    ) -> Result<GenerateResponse, Box<dyn std::error::Error + Send + Sync>> {
        // Extract OpenAI-specific options
        let openai_opts = self.extract_openai_options(&options.provider_options);

        // Handle temperature for search preview models
        let temperature = if is_search_preview_model(&self.model_id) {
            if options.temperature.is_some() {
                warn!(
                    "Temperature is not supported for search preview model '{}'. Removing temperature setting.",
                    self.model_id
                );
            }
            None
        } else {
            options.temperature
        };

        // Handle max_tokens vs max_completion_tokens for reasoning models
        let (max_tokens, max_completion_tokens) = if is_reasoning_model(&self.model_id) {
            // For reasoning models, use max_completion_tokens instead of max_tokens
            let mct = openai_opts
                .max_completion_tokens
                .or(options.max_output_tokens);
            (None, mct)
        } else {
            (options.max_output_tokens, openai_opts.max_completion_tokens)
        };

        // Validate service_tier for flex processing
        let service_tier = match &openai_opts.service_tier {
            Some(tier) if tier == "flex" && !supports_flex_processing(&self.model_id) => {
                warn!(
                    "Flex processing is not supported for model '{}'. Supported models: o3, o4-mini, gpt-5. Service tier will be ignored.",
                    self.model_id
                );
                None
            }
            tier => tier.clone(),
        };

        let request = crate::api_types::ChatCompletionRequest {
            model: self.model_id.clone(),
            messages: self.convert_prompt_to_messages(&options.prompt),
            temperature,
            max_tokens,
            stream: Some(false),
            tools: options.tools.as_ref().map(|t| self.convert_tools(t)),
            tool_choice: options
                .tool_choice
                .as_ref()
                .map(|tc| self.convert_tool_choice(tc)),
            response_format: options
                .response_format
                .as_ref()
                .map(|rf| self.convert_response_format(rf)),
            stream_options: None, // Not needed for non-streaming

            // OpenAI-specific options
            logit_bias: openai_opts.logit_bias,
            logprobs: openai_opts.logprobs,
            top_logprobs: None, // TODO: Extract from logprobs if it's a number
            user: openai_opts.user,
            parallel_tool_calls: openai_opts.parallel_tool_calls,
            reasoning_effort: openai_opts.reasoning_effort,
            max_completion_tokens,
            store: openai_opts.store,
            metadata: openai_opts.metadata,
            prediction: openai_opts.prediction,
            service_tier,
            verbosity: openai_opts.verbosity,
            prompt_cache_key: openai_opts.prompt_cache_key,
            safety_identifier: openai_opts.safety_identifier,
        };

        // Build request with custom headers
        let mut request_builder = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request);

        // Add custom headers if provided
        if let Some(headers) = &options.headers {
            for (key, value) in headers {
                request_builder = request_builder.header(key, value);
            }
        }

        let response = request_builder.send().await?;

        if !response.status().is_success() {
            return Err(Box::new(crate::error::OpenAIError::ApiError {
                message: format!("API returned status {}", response.status()),
                status_code: Some(response.status().as_u16()),
            }));
        }

        // Capture headers before consuming response
        let headers: HashMap<String, String> = response
            .headers()
            .iter()
            .map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or("").to_string()))
            .collect();

        let api_response: crate::api_types::ChatCompletionResponse = response.json().await?;

        let choice = &api_response.choices[0];

        // Build content from the response
        let mut content = Vec::new();

        // Add text content if present
        if let Some(message_content) = &choice.message.content {
            // Extract text from ChatMessageContent
            let text = match message_content {
                crate::api_types::ChatMessageContent::Text(s) => s.clone(),
                crate::api_types::ChatMessageContent::Parts(parts) => {
                    // Join text parts if we somehow get a multi-part response
                    parts
                        .iter()
                        .filter_map(|part| match part {
                            crate::multimodal::OpenAIContentPart::Text { text } => {
                                Some(text.clone())
                            }
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                }
            };

            if !text.is_empty() {
                content.push(Content::Text(TextPart {
                    text,
                    provider_metadata: None,
                }));
            }
        }

        // Add tool calls if present
        if let Some(tool_calls) = &choice.message.tool_calls {
            for tool_call in tool_calls {
                content.push(Content::ToolCall(ToolCallPart {
                    tool_call_id: tool_call.id.clone(),
                    tool_name: tool_call.function.name.clone(),
                    input: tool_call.function.arguments.clone(),
                    provider_executed: None,
                    dynamic: None,
                    provider_metadata: None,
                }));
            }
        }

        let usage = api_response
            .usage
            .as_ref()
            .map(|u| Usage {
                input_tokens: Some(u.prompt_tokens),
                output_tokens: u.completion_tokens,
                total_tokens: Some(u.total_tokens),
                reasoning_tokens: None,
                cached_input_tokens: None,
            })
            .unwrap_or_default();

        // Set finish reason based on whether tool calls were made
        let finish_reason = if choice.message.tool_calls.is_some() {
            FinishReason::ToolCalls
        } else {
            self.map_finish_reason(choice.finish_reason.as_deref())
        };

        // Build provider metadata with logprobs if present
        let provider_metadata = if let Some(logprobs) = &choice.logprobs {
            if let Some(content_logprobs) = logprobs.get("content") {
                let mut metadata = HashMap::new();
                let mut openai_metadata = HashMap::new();

                // Convert from serde_json::Value to JsonValue
                let json_value: ai_sdk_provider::json_value::JsonValue =
                    serde_json::from_value(content_logprobs.clone())
                        .unwrap_or(ai_sdk_provider::json_value::JsonValue::Null);

                openai_metadata.insert("logprobs".to_string(), json_value);
                metadata.insert("openai".to_string(), openai_metadata);
                Some(metadata)
            } else {
                None
            }
        } else {
            None
        };

        // Build response info
        let response_info = Some(ResponseInfo {
            headers: Some(headers),
            body: Some(serde_json::to_value(&api_response).unwrap_or(serde_json::json!({}))),
            id: Some(api_response.id.clone()),
            timestamp: Some({
                // Convert Unix timestamp to ISO 8601
                let secs = api_response.created as i64;
                let hours = secs / 3600;
                let minutes = (secs % 3600) / 60;
                let seconds = secs % 60;
                format!("1970-01-01T{:02}:{:02}:{:02}Z", hours, minutes, seconds)
            }),
            model_id: Some(api_response.model.clone()),
        });

        Ok(GenerateResponse {
            content,
            finish_reason,
            usage,
            provider_metadata,
            request: None,
            response: response_info,
            warnings: vec![],
        })
    }

    async fn do_stream(
        &self,
        options: CallOptions,
    ) -> Result<StreamResponse, Box<dyn std::error::Error + Send + Sync + 'static>> {
        // Extract OpenAI-specific options
        let openai_opts = self.extract_openai_options(&options.provider_options);

        // Handle temperature for search preview models
        let temperature = if is_search_preview_model(&self.model_id) {
            if options.temperature.is_some() {
                warn!(
                    "Temperature is not supported for search preview model '{}'. Removing temperature setting.",
                    self.model_id
                );
            }
            None
        } else {
            options.temperature
        };

        // Handle max_tokens vs max_completion_tokens for reasoning models
        let (max_tokens, max_completion_tokens) = if is_reasoning_model(&self.model_id) {
            // For reasoning models, use max_completion_tokens instead of max_tokens
            let mct = openai_opts
                .max_completion_tokens
                .or(options.max_output_tokens);
            (None, mct)
        } else {
            (options.max_output_tokens, openai_opts.max_completion_tokens)
        };

        // Validate service_tier for flex processing
        let service_tier = match &openai_opts.service_tier {
            Some(tier) if tier == "flex" && !supports_flex_processing(&self.model_id) => {
                warn!(
                    "Flex processing is not supported for model '{}'. Supported models: o3, o4-mini, gpt-5. Service tier will be ignored.",
                    self.model_id
                );
                None
            }
            tier => tier.clone(),
        };

        let request = crate::api_types::ChatCompletionRequest {
            model: self.model_id.clone(),
            messages: self.convert_prompt_to_messages(&options.prompt),
            temperature,
            max_tokens,
            stream: Some(true),
            tools: options.tools.as_ref().map(|t| self.convert_tools(t)),
            tool_choice: options
                .tool_choice
                .as_ref()
                .map(|tc| self.convert_tool_choice(tc)),
            response_format: options
                .response_format
                .as_ref()
                .map(|rf| self.convert_response_format(rf)),
            stream_options: Some(crate::api_types::StreamOptions {
                include_usage: true,
            }),

            // OpenAI-specific options
            logit_bias: openai_opts.logit_bias,
            logprobs: openai_opts.logprobs,
            top_logprobs: None,
            user: openai_opts.user,
            parallel_tool_calls: openai_opts.parallel_tool_calls,
            reasoning_effort: openai_opts.reasoning_effort,
            max_completion_tokens,
            store: openai_opts.store,
            metadata: openai_opts.metadata,
            prediction: openai_opts.prediction,
            service_tier,
            verbosity: openai_opts.verbosity,
            prompt_cache_key: openai_opts.prompt_cache_key,
            safety_identifier: openai_opts.safety_identifier,
        };

        // Build request with custom headers
        let mut request_builder = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request);

        // Add custom headers if provided
        if let Some(headers) = &options.headers {
            for (key, value) in headers {
                request_builder = request_builder.header(key, value);
            }
        }

        let response = request_builder.send().await?;

        let status = response.status();
        if !status.is_success() {
            return Err(Box::new(crate::error::OpenAIError::ApiError {
                message: format!("API returned status {}", status),
                status_code: Some(status.as_u16()),
            }));
        }

        let stream_impl = stream! {
            let mut byte_stream = response.bytes_stream();
            let mut buffer = String::new();
            let mut tool_calls: Vec<crate::api_types::OpenAIToolCall> = Vec::new();
            let mut accumulated_usage: Option<Usage> = None;
            let mut last_finish_reason: Option<FinishReason> = None;

            while let Some(chunk_result) = byte_stream.next().await {
                match chunk_result {
                    Ok(bytes) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));

                        // Process SSE lines
                        while let Some(line_end) = buffer.find('\n') {
                            let line = buffer[..line_end].trim().to_string();
                            buffer.drain(..line_end + 1);

                            if let Some(data) = line.strip_prefix("data: ") {
                                if data == "[DONE]" {
                                    break;
                                }

                                // Parse JSON chunk and convert to StreamPart
                                if let Ok(chunk) = serde_json::from_str::<crate::api_types::ChatCompletionChunk>(data) {
                                    // Capture usage if present
                                    if let Some(usage_info) = &chunk.usage {
                                        accumulated_usage = Some(Usage {
                                            input_tokens: Some(usage_info.prompt_tokens),
                                            output_tokens: usage_info.completion_tokens,
                                            total_tokens: Some(usage_info.total_tokens),
                                            reasoning_tokens: None,
                                            cached_input_tokens: None,
                                        });
                                    }

                                    // Convert OpenAI chunk to our StreamPart
                                    if let Some(choice) = chunk.choices.first() {
                                        // Handle text content
                                        if let Some(content) = &choice.delta.content {
                                            yield Ok(StreamPart::TextDelta {
                                                id: "0".into(),
                                                delta: content.clone(),
                                                provider_metadata: None,
                                            });
                                        }

                                        // Handle tool call deltas
                                        if let Some(tool_call_deltas) = &choice.delta.tool_calls {
                                            for tool_call_delta in tool_call_deltas {
                                                let index = tool_call_delta.index as usize;

                                                // Initialize new tool call if needed
                                                if tool_calls.len() <= index {
                                                    let tool_id = tool_call_delta.id.clone().unwrap_or_default();
                                                    let tool_name = tool_call_delta.function.name.clone().unwrap_or_default();

                                                    tool_calls.push(crate::api_types::OpenAIToolCall {
                                                        id: tool_id.clone(),
                                                        r#type: "function".to_string(),
                                                        function: crate::api_types::OpenAIFunctionCall {
                                                            name: tool_name.clone(),
                                                            arguments: String::new(),
                                                        },
                                                    });

                                                    // Emit ToolInputStart
                                                    yield Ok(StreamPart::ToolInputStart {
                                                        id: tool_id,
                                                        tool_name,
                                                        provider_metadata: None,
                                                        provider_executed: None,
                                                        dynamic: None,
                                                        title: None,
                                                    });
                                                }

                                                // Accumulate arguments
                                                if let Some(args_delta) = &tool_call_delta.function.arguments {
                                                    tool_calls[index].function.arguments.push_str(args_delta);

                                                    // Emit ToolInputDelta
                                                    yield Ok(StreamPart::ToolInputDelta {
                                                        id: tool_calls[index].id.clone(),
                                                        delta: args_delta.clone(),
                                                        provider_metadata: None,
                                                    });
                                                }
                                            }
                                        }

                                        // Handle finish reason
                                        if let Some(finish_reason) = &choice.finish_reason {
                                            if !finish_reason.is_empty() && finish_reason != "null" {
                                                // Emit ToolInputEnd and ToolCall for each complete tool
                                                for tool_call in &tool_calls {
                                                    yield Ok(StreamPart::ToolInputEnd {
                                                        id: tool_call.id.clone(),
                                                        provider_metadata: None,
                                                    });

                                                    yield Ok(StreamPart::ToolCall(ToolCallPart {
                                                        tool_call_id: tool_call.id.clone(),
                                                        tool_name: tool_call.function.name.clone(),
                                                        input: tool_call.function.arguments.clone(),
                                                        provider_executed: None,
                                                        dynamic: None,
                                                        provider_metadata: None,
                                                    }));
                                                }

                                                let mapped_reason = match finish_reason.as_str() {
                                                    "stop" => FinishReason::Stop,
                                                    "length" => FinishReason::Length,
                                                    "content_filter" => FinishReason::ContentFilter,
                                                    "tool_calls" => FinishReason::ToolCalls,
                                                    _ => FinishReason::Unknown,
                                                };

                                                // Store the finish reason but don't emit Finish yet
                                                // OpenAI may send usage in a subsequent chunk
                                                last_finish_reason = Some(mapped_reason);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        yield Err(StreamError::Other(format!("Stream error: {}", e)));
                        break;
                    }
                }
            }

            // Emit Finish event with accumulated usage
            // OpenAI sends usage in a separate chunk after finish_reason when stream_options.include_usage is true
            if let Some(finish_reason) = last_finish_reason {
                let usage_to_send = accumulated_usage.unwrap_or_default();
                yield Ok(StreamPart::Finish {
                    usage: usage_to_send,
                    finish_reason,
                    provider_metadata: None,
                });
            }
        };

        Ok(StreamResponse {
            stream: Box::pin(stream_impl),
            request: None,
            response: None,
        })
    }
}
