use super::language_model_middleware::{CallType, LanguageModelMiddleware};
use ai_sdk_provider::language_model::{CallOptions, LanguageModel};
use async_trait::async_trait;

/// Middleware that applies default settings to call options
///
/// Settings from the middleware are used as defaults, but can be overridden
/// by the actual call options.
pub struct DefaultSettingsMiddleware {
    settings: CallOptions,
}

impl DefaultSettingsMiddleware {
    /// Create a new default settings middleware
    pub fn new(settings: CallOptions) -> Self {
        Self { settings }
    }
}

#[async_trait]
impl LanguageModelMiddleware for DefaultSettingsMiddleware {
    async fn transform_params(
        &self,
        _call_type: CallType,
        params: CallOptions,
        _model: &dyn LanguageModel,
    ) -> Result<CallOptions, Box<dyn std::error::Error + Send + Sync>> {
        Ok(merge_call_options(&self.settings, &params))
    }
}

/// Merge two sets of call options
///
/// Values from `overrides` take precedence over values from `base`.
fn merge_call_options(base: &CallOptions, overrides: &CallOptions) -> CallOptions {
    CallOptions {
        prompt: if overrides.prompt.is_empty() {
            base.prompt.clone()
        } else {
            overrides.prompt.clone()
        },
        temperature: overrides.temperature.or(base.temperature),
        max_output_tokens: overrides.max_output_tokens.or(base.max_output_tokens),
        top_p: overrides.top_p.or(base.top_p),
        top_k: overrides.top_k.or(base.top_k),
        frequency_penalty: overrides.frequency_penalty.or(base.frequency_penalty),
        presence_penalty: overrides.presence_penalty.or(base.presence_penalty),
        stop_sequences: overrides
            .stop_sequences
            .clone()
            .or_else(|| base.stop_sequences.clone()),
        seed: overrides.seed.or(base.seed),
        tools: overrides.tools.clone().or_else(|| base.tools.clone()),
        tool_choice: overrides
            .tool_choice
            .clone()
            .or_else(|| base.tool_choice.clone()),
        response_format: overrides
            .response_format
            .clone()
            .or_else(|| base.response_format.clone()),
        headers: merge_headers(&base.headers, &overrides.headers),
        include_raw_chunks: overrides.include_raw_chunks.or(base.include_raw_chunks),
        provider_options: overrides
            .provider_options
            .clone()
            .or_else(|| base.provider_options.clone()),
    }
}

/// Merge headers from base and overrides
///
/// Headers from `overrides` take precedence over headers from `base`.
fn merge_headers(
    base: &Option<std::collections::HashMap<String, String>>,
    overrides: &Option<std::collections::HashMap<String, String>>,
) -> Option<std::collections::HashMap<String, String>> {
    match (base, overrides) {
        (None, None) => None,
        (Some(b), None) => Some(b.clone()),
        (None, Some(o)) => Some(o.clone()),
        (Some(b), Some(o)) => {
            let mut merged = b.clone();
            for (key, value) in o {
                merged.insert(key.clone(), value.clone());
            }
            Some(merged)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_call_options_empty_override() {
        let base = CallOptions {
            temperature: Some(0.7),
            max_output_tokens: Some(100),
            ..Default::default()
        };
        let overrides = CallOptions::default();

        let merged = merge_call_options(&base, &overrides);

        assert_eq!(merged.temperature, Some(0.7));
        assert_eq!(merged.max_output_tokens, Some(100));
    }

    #[test]
    fn test_merge_call_options_with_override() {
        let base = CallOptions {
            temperature: Some(0.7),
            max_output_tokens: Some(100),
            ..Default::default()
        };
        let overrides = CallOptions {
            temperature: Some(0.5),
            ..Default::default()
        };

        let merged = merge_call_options(&base, &overrides);

        assert_eq!(merged.temperature, Some(0.5)); // Overridden
        assert_eq!(merged.max_output_tokens, Some(100)); // From base
    }

    #[test]
    fn test_merge_headers() {
        use std::collections::HashMap;

        let mut base_map = HashMap::new();
        base_map.insert("Content-Type".to_string(), "application/json".to_string());
        base_map.insert("User-Agent".to_string(), "test".to_string());

        let mut override_map = HashMap::new();
        override_map.insert("Content-Type".to_string(), "text/plain".to_string());
        override_map.insert("Authorization".to_string(), "Bearer token".to_string());

        let merged = merge_headers(&Some(base_map), &Some(override_map)).unwrap();

        assert_eq!(merged.len(), 3);
        assert_eq!(merged.get("Content-Type").unwrap(), "text/plain"); // Overridden
        assert_eq!(merged.get("User-Agent").unwrap(), "test"); // From base
        assert_eq!(merged.get("Authorization").unwrap(), "Bearer token"); // From override
    }

    #[tokio::test]
    async fn test_default_settings_middleware() {
        use ai_sdk_provider::language_model::{
            Content, FinishReason, GenerateResponse, TextPart, Usage,
        };

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
                Ok(GenerateResponse {
                    content: vec![Content::Text(TextPart {
                        text: "test".to_string(),
                        provider_metadata: None,
                    })],
                    finish_reason: FinishReason::Stop,
                    usage: Usage::default(),
                    provider_metadata: None,
                    request: None,
                    response: None,
                    warnings: vec![],
                })
            }

            async fn do_stream(
                &self,
                _opts: CallOptions,
            ) -> Result<
                ai_sdk_provider::language_model::StreamResponse,
                Box<dyn std::error::Error + Send + Sync + 'static>,
            > {
                unimplemented!()
            }
        }

        let middleware = DefaultSettingsMiddleware::new(CallOptions {
            temperature: Some(0.7),
            max_output_tokens: Some(100),
            ..Default::default()
        });

        let params = CallOptions {
            temperature: Some(0.5),
            ..Default::default()
        };

        let model = DummyModel;
        let transformed = middleware
            .transform_params(CallType::Generate, params, &model)
            .await
            .unwrap();

        assert_eq!(transformed.temperature, Some(0.5)); // User override
        assert_eq!(transformed.max_output_tokens, Some(100)); // From default
    }
}
