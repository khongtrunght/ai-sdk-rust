use ai_sdk_provider::language_model::{Message, StreamError, TextPart, UserContentPart};
use ai_sdk_provider::*;
use async_stream::stream;
use async_trait::async_trait;
use futures::stream::StreamExt;
use reqwest::Client;
use std::collections::HashMap;

pub struct OpenAIChatModel {
    model_id: String,
    api_key: String,
    client: Client,
    base_url: String,
}

impl OpenAIChatModel {
    pub fn new(model_id: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            api_key: api_key.into(),
            client: Client::new(),
            base_url: "https://api.openai.com/v1".into(),
        }
    }

    fn convert_prompt_to_messages(&self, prompt: &[Message]) -> Vec<crate::api_types::ChatMessage> {
        // Convert our prompt format to OpenAI's message format
        prompt
            .iter()
            .map(|msg| {
                match msg {
                    Message::System { content } => crate::api_types::ChatMessage {
                        role: "system".into(),
                        content: content.clone(),
                    },
                    Message::User { content } => crate::api_types::ChatMessage {
                        role: "user".into(),
                        content: content
                            .iter()
                            .filter_map(|part| match part {
                                UserContentPart::Text { text } => Some(text.clone()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n"),
                    },
                    Message::Assistant { content } => crate::api_types::ChatMessage {
                        role: "assistant".into(),
                        content: format!("{:?}", content), // Simplified
                    },
                    Message::Tool { .. } => crate::api_types::ChatMessage {
                        role: "tool".into(),
                        content: "".into(),
                    },
                }
            })
            .collect()
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
        let request = crate::api_types::ChatCompletionRequest {
            model: self.model_id.clone(),
            messages: self.convert_prompt_to_messages(&options.prompt),
            temperature: options.temperature,
            max_tokens: options.max_output_tokens,
            stream: Some(false),
        };

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(Box::new(crate::error::OpenAIError::ApiError {
                message: format!("API returned status {}", response.status()),
                status_code: Some(response.status().as_u16()),
            }));
        }

        let api_response: crate::api_types::ChatCompletionResponse = response.json().await?;

        let choice = &api_response.choices[0];

        let content = vec![Content::Text(TextPart {
            text: choice.message.content.clone(),
            provider_metadata: None,
        })];

        let usage = api_response
            .usage
            .as_ref()
            .map(|u| Usage {
                input_tokens: Some(u.prompt_tokens),
                output_tokens: Some(u.completion_tokens),
                total_tokens: Some(u.total_tokens),
                reasoning_tokens: None,
                cached_input_tokens: None,
            })
            .unwrap_or_default();

        Ok(GenerateResponse {
            content,
            finish_reason: self.map_finish_reason(choice.finish_reason.as_deref()),
            usage,
            provider_metadata: None,
            request: None,
            response: None,
            warnings: vec![],
        })
    }

    async fn do_stream(
        &self,
        options: CallOptions,
    ) -> Result<StreamResponse, Box<dyn std::error::Error + Send + Sync + 'static>> {
        let request = crate::api_types::ChatCompletionRequest {
            model: self.model_id.clone(),
            messages: self.convert_prompt_to_messages(&options.prompt),
            temperature: options.temperature,
            max_tokens: options.max_output_tokens,
            stream: Some(true),
        };

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

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
                                    // Convert OpenAI chunk to our StreamPart
                                    if let Some(choice) = chunk.choices.first() {
                                        if let Some(content) = &choice.delta.content {
                                            yield Ok(StreamPart::TextDelta {
                                                id: "0".into(),
                                                delta: content.clone(),
                                                provider_metadata: None,
                                            });
                                        }

                                        // Handle finish reason
                                        if let Some(finish_reason) = &choice.finish_reason {
                                            if !finish_reason.is_empty() && finish_reason != "null" {
                                                let mapped_reason = match finish_reason.as_str() {
                                                    "stop" => FinishReason::Stop,
                                                    "length" => FinishReason::Length,
                                                    "content_filter" => FinishReason::ContentFilter,
                                                    "tool_calls" => FinishReason::ToolCalls,
                                                    _ => FinishReason::Unknown,
                                                };
                                                yield Ok(StreamPart::Finish {
                                                    usage: Usage::default(),
                                                    finish_reason: mapped_reason,
                                                    provider_metadata: None,
                                                });
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
        };

        Ok(StreamResponse {
            stream: Box::pin(stream_impl),
            request: None,
            response: None,
        })
    }
}
