use ai_sdk_provider::language_model::{
    AssistantContentPart, CallOptions, CallWarning, Content, FinishReason, GenerateResponse,
    LanguageModel, Message, ResponseInfo, SourcePart, SourceType, StreamResponse, TextPart,
    ToolCallPart, Usage, UserContentPart,
};
use ai_sdk_provider_utils::merge_headers_reqwest;
use async_trait::async_trait;
use log::warn;
use reqwest::Client;
use std::collections::HashMap;

use crate::model_detection::is_reasoning_model;
use crate::openai_config::{OpenAIConfig, OpenAIUrlOptions};

use super::api_types::*;
use super::options::{LogprobsOption, OpenAIResponsesProviderOptions};

/// OpenAI Responses API implementation
pub struct OpenAIResponsesLanguageModel {
    model_id: String,
    config: OpenAIConfig,
    client: Client,
}

impl OpenAIResponsesLanguageModel {
    /// Creates a new Responses API model with the specified model ID and configuration.
    ///
    /// # Arguments
    /// * `model_id` - The model ID (e.g., "gpt-4o", "o3-mini")
    /// * `config` - OpenAI configuration
    ///
    /// # Example
    /// ```rust,ignore
    /// use ai_sdk_openai::{OpenAIConfig, responses::{OpenAIResponsesLanguageModel, GPT_4O}};
    ///
    /// let api_key = std::env::var("OPENAI_API_KEY").unwrap();
    /// let config = OpenAIConfig::new(
    ///     "openai",
    ///     |opts| format!("https://api.openai.com/v1{}", opts.path),
    ///     move || {
    ///         let mut headers = std::collections::HashMap::new();
    ///         headers.insert("Authorization".to_string(), format!("Bearer {}", api_key));
    ///         headers
    ///     }
    /// );
    ///
    /// let model = OpenAIResponsesLanguageModel::new(GPT_4O, config);
    /// ```
    pub fn new(model_id: impl Into<String>, config: impl Into<OpenAIConfig>) -> Self {
        Self {
            model_id: model_id.into(),
            config: config.into(),
            client: Client::new(),
        }
    }

    fn convert_prompt_to_input(&self, prompt: &[Message]) -> Vec<ResponsesInputItem> {
        let mut input = Vec::new();

        for msg in prompt {
            match msg {
                Message::System { content } => {
                    // For reasoning models, use "developer" role instead of "system"
                    let role = if is_reasoning_model(&self.model_id) {
                        "developer"
                    } else {
                        "system"
                    };

                    input.push(ResponsesInputItem::Message(ResponsesMessage {
                        role: role.to_string(),
                        content: Some(ResponsesContent::Text(content.clone())),
                        id: None,
                    }));
                }
                Message::User { content } => {
                    // For now, just extract text content
                    // TODO: Add multimodal support
                    let text_content = content
                        .iter()
                        .filter_map(|part| match part {
                            UserContentPart::Text { text } => Some(text.clone()),
                            UserContentPart::File { .. } => None, // TODO: Support files
                        })
                        .collect::<Vec<_>>()
                        .join("\n");

                    input.push(ResponsesInputItem::Message(ResponsesMessage {
                        role: "user".to_string(),
                        content: Some(ResponsesContent::Text(text_content)),
                        id: None,
                    }));
                }
                Message::Assistant { content } => {
                    let mut parts = Vec::new();

                    for part in content {
                        // TODO: Add tool call support
                        if let AssistantContentPart::Text(text_part) = part {
                            parts.push(ResponsesContentPart::OutputText {
                                text: text_part.text.clone(),
                            });
                        }
                    }

                    if !parts.is_empty() {
                        input.push(ResponsesInputItem::Message(ResponsesMessage {
                            role: "assistant".to_string(),
                            content: Some(ResponsesContent::Parts(parts)),
                            id: None,
                        }));
                    }
                }
                Message::Tool { .. } => {
                    // TODO: Add tool result support
                    warn!("Tool messages not yet supported in Responses API");
                }
            }
        }

        input
    }

    fn map_finish_reason(&self, incomplete_reason: Option<&str>) -> FinishReason {
        match incomplete_reason {
            None => FinishReason::Stop,
            Some("max_output_tokens") => FinishReason::Length,
            Some("content_filter") => FinishReason::ContentFilter,
            _ => FinishReason::Unknown,
        }
    }
}

#[async_trait]
impl LanguageModel for OpenAIResponsesLanguageModel {
    fn provider(&self) -> &str {
        &self.config.provider
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn supported_urls(&self) -> HashMap<String, Vec<String>> {
        let mut urls = HashMap::new();
        urls.insert("image/*".into(), vec![r"^https?://.*$".into()]);
        urls.insert("application/pdf".into(), vec![r"^https?://.*$".into()]);
        urls
    }

    async fn do_generate(
        &self,
        options: CallOptions,
    ) -> Result<GenerateResponse, Box<dyn std::error::Error + Send + Sync>> {
        let mut warnings = Vec::new();

        // Parse provider-specific options
        let responses_options =
            OpenAIResponsesProviderOptions::from(options.provider_options.clone());

        // Validate unsupported options
        if options.top_k.is_some() {
            warnings.push(CallWarning {
                message: "topK is not supported by the Responses API".into(),
            });
        }

        if options.seed.is_some() {
            warnings.push(CallWarning {
                message: "seed is not supported by the Responses API".into(),
            });
        }

        if options.presence_penalty.is_some() {
            warnings.push(CallWarning {
                message: "presencePenalty is not supported by the Responses API".into(),
            });
        }

        if options.frequency_penalty.is_some() {
            warnings.push(CallWarning {
                message: "frequencyPenalty is not supported by the Responses API".into(),
            });
        }

        if options
            .stop_sequences
            .as_ref()
            .is_some_and(|s| !s.is_empty())
        {
            warnings.push(CallWarning {
                message: "stopSequences is not supported by the Responses API".into(),
            });
        }

        // Handle temperature for reasoning models
        let temperature = if is_reasoning_model(&self.model_id) {
            if options.temperature.is_some() {
                warnings.push(CallWarning {
                    message: "temperature is not supported for reasoning models".into(),
                });
            }
            None
        } else {
            options.temperature
        };

        // Handle top_p for reasoning models
        let top_p = if is_reasoning_model(&self.model_id) {
            if options.top_p.is_some() {
                warnings.push(CallWarning {
                    message: "topP is not supported for reasoning models".into(),
                });
            }
            None
        } else {
            options.top_p
        };

        // Build request
        let request = ResponsesRequest {
            model: self.model_id.clone(),
            input: self.convert_prompt_to_input(&options.prompt),
            temperature,
            top_p,
            max_output_tokens: options.max_output_tokens,
            stream: Some(false),

            // Provider options
            conversation: responses_options.conversation,
            include: responses_options.include,
            instructions: responses_options.instructions,
            max_tool_calls: responses_options.max_tool_calls,
            metadata: responses_options.metadata,
            parallel_tool_calls: responses_options.parallel_tool_calls,
            previous_response_id: responses_options.previous_response_id,
            prompt_cache_key: responses_options.prompt_cache_key,
            prompt_cache_retention: responses_options.prompt_cache_retention,
            safety_identifier: responses_options.safety_identifier,
            service_tier: responses_options.service_tier,
            store: responses_options.store,
            top_logprobs: responses_options.logprobs.map(|lp| match lp {
                LogprobsOption::Enabled => super::TOP_LOGPROBS_MAX,
                LogprobsOption::TopN(n) => n,
            }),
            truncation: responses_options.truncation,
            user: responses_options.user,
        };

        // Build request URL
        let url = (self.config.url)(OpenAIUrlOptions {
            model_id: self.model_id.clone(),
            path: "/responses".to_string(),
        });

        // Send request
        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .headers(merge_headers_reqwest(
                (self.config.headers)(),
                options.headers.as_ref(),
            ))
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        let response_headers: HashMap<String, String> = response
            .headers()
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
            .collect();

        if !status.is_success() {
            let error_text = response.text().await?;
            return Err(format!("API error {}: {}", status, error_text).into());
        }

        let api_response: ResponsesResponse = response.json().await?;

        // Handle errors in response
        if let Some(error) = api_response.error {
            return Err(format!("Responses API error: {}", error.message).into());
        }

        // Build content from response
        let mut content = Vec::new();

        if let Some(output_items) = api_response.output {
            for item in output_items {
                match item {
                    ResponsesOutputItem::Message { content: parts, .. } => {
                        for part in parts {
                            match part {
                                ResponsesOutputContentPart::OutputText {
                                    text,
                                    annotations,
                                    ..
                                } => {
                                    if !text.is_empty() {
                                        content.push(Content::Text(TextPart {
                                            text,
                                            provider_metadata: None,
                                        }));
                                    }

                                    // Add annotations as sources
                                    for annotation in annotations {
                                        match annotation {
                                            ResponsesAnnotation::UrlCitation {
                                                url, title, ..
                                            } => {
                                                let id = self
                                                    .config
                                                    .generate_id
                                                    .as_ref()
                                                    .map(|f| f())
                                                    .unwrap_or_else(|| {
                                                        format!(
                                                            "source-url-{}",
                                                            std::time::SystemTime::now()
                                                                .duration_since(
                                                                    std::time::UNIX_EPOCH
                                                                )
                                                                .unwrap()
                                                                .as_nanos()
                                                        )
                                                    });

                                                content.push(Content::Source(SourcePart {
                                                    id,
                                                    source_type: SourceType::Url,
                                                    url: Some(url),
                                                    title: Some(title),
                                                    provider_metadata: None,
                                                }));
                                            }
                                            ResponsesAnnotation::FileCitation {
                                                file_id,
                                                filename,
                                                ..
                                            } => {
                                                let id = self
                                                    .config
                                                    .generate_id
                                                    .as_ref()
                                                    .map(|f| f())
                                                    .unwrap_or_else(|| {
                                                        format!(
                                                            "source-file-{}",
                                                            std::time::SystemTime::now()
                                                                .duration_since(
                                                                    std::time::UNIX_EPOCH
                                                                )
                                                                .unwrap()
                                                                .as_nanos()
                                                        )
                                                    });

                                                content.push(Content::Source(SourcePart {
                                                    id,
                                                    source_type: SourceType::Document,
                                                    url: None,
                                                    title: filename.or(Some(file_id)),
                                                    provider_metadata: None,
                                                }));
                                            }
                                            _ => {} // Ignore other annotation types for now
                                        }
                                    }
                                }
                            }
                        }
                    }
                    ResponsesOutputItem::FunctionCall {
                        call_id,
                        name,
                        arguments,
                        ..
                    } => {
                        content.push(Content::ToolCall(ToolCallPart {
                            tool_call_id: call_id,
                            tool_name: name,
                            input: arguments,
                            provider_executed: None,
                            dynamic: None,
                            provider_metadata: None,
                        }));
                    }
                    ResponsesOutputItem::Reasoning { summary, .. } => {
                        // Add reasoning as text content
                        for summary_part in summary {
                            if !summary_part.text.is_empty() {
                                content.push(Content::Text(TextPart {
                                    text: summary_part.text,
                                    provider_metadata: None,
                                }));
                            }
                        }
                    }
                    // Ignore other output items for now
                    _ => {}
                }
            }
        }

        // Build usage
        let usage = api_response
            .usage
            .map(|u| Usage {
                input_tokens: Some(u.input_tokens),
                output_tokens: Some(u.output_tokens),
                total_tokens: Some(u.input_tokens + u.output_tokens),
                reasoning_tokens: u.output_tokens_details.and_then(|d| d.reasoning_tokens),
                cached_input_tokens: u.input_tokens_details.and_then(|d| d.cached_tokens),
            })
            .unwrap_or_default();

        // Build provider metadata
        let mut provider_metadata = HashMap::new();
        let mut openai_metadata = HashMap::new();
        openai_metadata.insert(
            "responseId".to_string(),
            ai_sdk_provider::json_value::JsonValue::String(api_response.id.clone()),
        );
        if let Some(service_tier) = api_response.service_tier {
            openai_metadata.insert(
                "serviceTier".to_string(),
                ai_sdk_provider::json_value::JsonValue::String(service_tier),
            );
        }
        provider_metadata.insert("openai".to_string(), openai_metadata);

        // Build response info
        let response_info = Some(ResponseInfo {
            headers: Some(response_headers),
            body: None,
            id: Some(api_response.id),
            timestamp: api_response.created_at.map(|ts| {
                let secs = ts;
                let hours = secs / 3600;
                let minutes = (secs % 3600) / 60;
                let seconds = secs % 60;
                format!("1970-01-01T{:02}:{:02}:{:02}Z", hours, minutes, seconds)
            }),
            model_id: Some(api_response.model),
        });

        Ok(GenerateResponse {
            content,
            finish_reason: self.map_finish_reason(
                api_response
                    .incomplete_details
                    .as_ref()
                    .map(|d| d.reason.as_str()),
            ),
            usage,
            provider_metadata: Some(provider_metadata),
            request: None,
            response: response_info,
            warnings,
        })
    }

    async fn do_stream(
        &self,
        _options: CallOptions,
    ) -> Result<StreamResponse, Box<dyn std::error::Error + Send + Sync + 'static>> {
        // TODO: Implement streaming support
        Err("Streaming not yet implemented for Responses API".into())
    }
}
