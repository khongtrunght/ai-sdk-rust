use super::language_model_middleware::{GenerateFn, LanguageModelMiddleware, StreamFn};
use ai_sdk_provider::language_model::{
    CallOptions, Content, LanguageModel, ResponseMetadata, StreamPart, StreamResponse,
};
use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

/// Middleware that simulates streaming from non-streaming calls
///
/// When `do_stream` is called, this middleware calls `do_generate` instead
/// and converts the result into a synthetic stream.
pub struct SimulateStreamingMiddleware;

#[async_trait]
impl LanguageModelMiddleware for SimulateStreamingMiddleware {
    async fn wrap_stream(
        &self,
        do_generate: GenerateFn,
        _do_stream: StreamFn,
        _params: &CallOptions,
        _model: &dyn LanguageModel,
    ) -> Result<StreamResponse, Box<dyn std::error::Error + Send + Sync + 'static>> {
        // Call do_generate instead of do_stream
        let result = do_generate().await?;

        // Save response/request for return value before moving into task
        let request = result.request.clone();
        let response = result.response.clone();

        // Create a channel for the synthetic stream
        let (tx, rx) = mpsc::channel(100);

        // Spawn a task to emit stream parts
        tokio::spawn(async move {
            // Emit stream-start
            let _ = tx
                .send(Ok(StreamPart::StreamStart {
                    warnings: result.warnings.clone(),
                }))
                .await;

            // Emit response metadata if available
            if let Some(response_info) = result.response.clone() {
                let metadata = ResponseMetadata {
                    id: response_info.id,
                    timestamp: response_info.timestamp,
                    model_id: response_info.model_id,
                };
                let _ = tx.send(Ok(StreamPart::ResponseMetadata { metadata })).await;
            }

            // Emit content parts as stream parts
            for (idx, content) in result.content.iter().enumerate() {
                match content {
                    Content::Text(text_part) => {
                        let id = idx.to_string();

                        // Emit text-start
                        let _ = tx
                            .send(Ok(StreamPart::TextStart {
                                id: id.clone(),
                                provider_metadata: text_part.provider_metadata.clone(),
                            }))
                            .await;

                        // Emit text-delta
                        let _ = tx
                            .send(Ok(StreamPart::TextDelta {
                                id: id.clone(),
                                delta: text_part.text.clone(),
                                provider_metadata: text_part.provider_metadata.clone(),
                            }))
                            .await;

                        // Emit text-end
                        let _ = tx
                            .send(Ok(StreamPart::TextEnd {
                                id,
                                provider_metadata: text_part.provider_metadata.clone(),
                            }))
                            .await;
                    }
                    Content::ToolCall(tool_call) => {
                        // Emit tool-call (complete, not streamed)
                        let _ = tx.send(Ok(StreamPart::ToolCall(tool_call.clone()))).await;
                    }
                    Content::ToolResult(tool_result) => {
                        // Emit tool-result
                        let _ = tx
                            .send(Ok(StreamPart::ToolResult(tool_result.clone())))
                            .await;
                    }
                    Content::Reasoning(reasoning_part) => {
                        let id = idx.to_string();

                        // Emit reasoning-start
                        let _ = tx
                            .send(Ok(StreamPart::ReasoningStart {
                                id: id.clone(),
                                provider_metadata: reasoning_part.provider_metadata.clone(),
                            }))
                            .await;

                        // Emit reasoning-delta
                        let _ = tx
                            .send(Ok(StreamPart::ReasoningDelta {
                                id: id.clone(),
                                delta: reasoning_part.reasoning.clone(),
                                provider_metadata: reasoning_part.provider_metadata.clone(),
                            }))
                            .await;

                        // Emit reasoning-end
                        let _ = tx
                            .send(Ok(StreamPart::ReasoningEnd {
                                id,
                                provider_metadata: reasoning_part.provider_metadata.clone(),
                            }))
                            .await;
                    }
                    Content::File(file_part) => {
                        // Emit file part directly
                        let _ = tx.send(Ok(StreamPart::File(file_part.clone()))).await;
                    }
                    Content::Source(source_part) => {
                        // Emit source part directly
                        let _ = tx.send(Ok(StreamPart::Source(source_part.clone()))).await;
                    }
                }
            }

            // Emit finish
            let _ = tx
                .send(Ok(StreamPart::Finish {
                    usage: result.usage,
                    finish_reason: result.finish_reason,
                    provider_metadata: result.provider_metadata,
                }))
                .await;
        });

        Ok(StreamResponse {
            stream: Box::pin(ReceiverStream::new(rx)),
            request,
            response,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_sdk_provider::language_model::{
        FinishReason, GenerateResponse, TextPart, ToolCallPart, Usage,
    };
    use futures::StreamExt;

    #[tokio::test]
    async fn test_simulate_streaming() {
        let middleware = SimulateStreamingMiddleware;

        // Create a mock generate function
        let do_generate: GenerateFn = std::sync::Arc::new(|| {
            Box::pin(async {
                Ok(GenerateResponse {
                    content: vec![Content::Text(TextPart {
                        text: "Hello, world!".to_string(),
                        provider_metadata: None,
                    })],
                    finish_reason: FinishReason::Stop,
                    usage: Usage::default(),
                    provider_metadata: None,
                    request: None,
                    response: None,
                    warnings: vec![],
                })
            })
        });

        // Create a dummy stream function (won't be called)
        let do_stream: StreamFn = std::sync::Arc::new(|| {
            Box::pin(async {
                panic!("do_stream should not be called");
            })
        });

        struct DummyModel;
        #[async_trait]
        impl LanguageModel for DummyModel {
            fn provider(&self) -> &str {
                "test"
            }
            fn model_id(&self) -> &str {
                "dummy"
            }
            async fn do_generate(
                &self,
                _opts: CallOptions,
            ) -> Result<GenerateResponse, Box<dyn std::error::Error + Send + Sync>> {
                unimplemented!()
            }
            async fn do_stream(
                &self,
                _opts: CallOptions,
            ) -> Result<StreamResponse, Box<dyn std::error::Error + Send + Sync + 'static>>
            {
                unimplemented!()
            }
        }

        let model = DummyModel;
        let params = CallOptions::default();

        let result = middleware
            .wrap_stream(do_generate, do_stream, &params, &model)
            .await
            .unwrap();

        // Collect stream parts
        let parts: Vec<_> = result
            .stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        // Verify stream structure
        assert!(matches!(parts[0], StreamPart::StreamStart { .. }));
        assert!(matches!(parts[1], StreamPart::TextStart { .. }));
        assert!(matches!(parts[2], StreamPart::TextDelta { .. }));
        assert!(matches!(parts[3], StreamPart::TextEnd { .. }));
        assert!(matches!(parts[4], StreamPart::Finish { .. }));
    }

    #[tokio::test]
    async fn test_simulate_streaming_with_tool_calls() {
        let middleware = SimulateStreamingMiddleware;

        let do_generate: GenerateFn = std::sync::Arc::new(|| {
            Box::pin(async {
                Ok(GenerateResponse {
                    content: vec![
                        Content::Text(TextPart {
                            text: "Let me check...".to_string(),
                            provider_metadata: None,
                        }),
                        Content::ToolCall(ToolCallPart {
                            tool_call_id: "call_1".to_string(),
                            tool_name: "get_weather".to_string(),
                            input: serde_json::json!({"location": "SF"}).to_string(),
                            provider_metadata: None,
                            provider_executed: None,
                            dynamic: None,
                        }),
                    ],
                    finish_reason: FinishReason::ToolCalls,
                    usage: Usage::default(),
                    provider_metadata: None,
                    request: None,
                    response: None,
                    warnings: vec![],
                })
            })
        });

        let do_stream: StreamFn = std::sync::Arc::new(|| {
            Box::pin(async {
                panic!("do_stream should not be called");
            })
        });

        struct DummyModel;
        #[async_trait]
        impl LanguageModel for DummyModel {
            fn provider(&self) -> &str {
                "test"
            }
            fn model_id(&self) -> &str {
                "dummy"
            }
            async fn do_generate(
                &self,
                _opts: CallOptions,
            ) -> Result<GenerateResponse, Box<dyn std::error::Error + Send + Sync>> {
                unimplemented!()
            }
            async fn do_stream(
                &self,
                _opts: CallOptions,
            ) -> Result<StreamResponse, Box<dyn std::error::Error + Send + Sync + 'static>>
            {
                unimplemented!()
            }
        }

        let model = DummyModel;
        let params = CallOptions::default();

        let result = middleware
            .wrap_stream(do_generate, do_stream, &params, &model)
            .await
            .unwrap();

        let parts: Vec<_> = result
            .stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        // Should have: StreamStart, TextStart, TextDelta, TextEnd, ToolCall, Finish
        assert_eq!(parts.len(), 6);
        assert!(matches!(parts[0], StreamPart::StreamStart { .. }));
        assert!(matches!(parts[4], StreamPart::ToolCall(_)));
        assert!(matches!(parts[5], StreamPart::Finish { .. }));
    }
}
