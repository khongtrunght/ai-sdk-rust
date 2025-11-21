use ai_sdk_provider::json_value::JsonValue;
use std::collections::HashMap;

/// OpenAI-specific options for chat completions
#[derive(Default)]
pub struct OpenAIChatOptions {
    /// Modify the likelihood of specified tokens appearing in the completion.
    ///
    /// Accepts a map of tokens (specified by their token ID in the GPT tokenizer)
    /// to an associated bias value from -100 to 100.
    pub logit_bias: Option<HashMap<String, f64>>,

    /// Return the log probabilities of the tokens.
    ///
    /// Setting to true will return the log probabilities of the tokens that were generated.
    pub logprobs: Option<bool>,

    /// Whether to enable parallel function calling during tool use.
    ///
    /// Defaults to true.
    pub parallel_tool_calls: Option<bool>,

    /// A unique identifier representing your end-user.
    ///
    /// This can help OpenAI to monitor and detect abuse.
    pub user: Option<String>,

    /// Reasoning effort for reasoning models.
    ///
    /// Valid values: 'none', 'minimal', 'low', 'medium', 'high'
    /// Defaults to 'medium'.
    pub reasoning_effort: Option<String>,

    /// Maximum number of completion tokens to generate.
    ///
    /// Useful for reasoning models.
    pub max_completion_tokens: Option<u32>,

    /// Whether to enable persistence in responses API.
    pub store: Option<bool>,

    /// Metadata to associate with the request.
    ///
    /// Keys must be max 64 characters, values must be max 512 characters.
    pub metadata: Option<HashMap<String, String>>,

    /// Parameters for prediction mode.
    pub prediction: Option<serde_json::Value>,

    /// Service tier for the request.
    ///
    /// - 'auto': Default service tier. The request will be processed with the service tier configured in the
    ///   Project settings. Unless otherwise configured, the Project will use 'default'.
    /// - 'flex': 50% cheaper processing at the cost of increased latency. Only available for o3 and o4-mini models.
    /// - 'priority': Higher-speed processing with predictably low latency at premium cost. Available for Enterprise customers.
    /// - 'default': The request will be processed with the standard pricing and performance for the selected model.
    ///
    /// Defaults to 'auto'.
    pub service_tier: Option<String>,

    /// Controls the verbosity of the model's responses.
    ///
    /// Valid values: 'low', 'medium', 'high'
    /// Lower values will result in more concise responses, while higher values will result in more verbose responses.
    pub verbosity: Option<String>,

    /// A cache key for prompt caching.
    ///
    /// Allows manual control over prompt caching behavior.
    /// Useful for improving cache hit rates and working around automatic caching issues.
    pub prompt_cache_key: Option<String>,

    /// A stable identifier used to help detect users of your application
    /// that may be violating OpenAI's usage policies.
    ///
    /// The IDs should be a string that uniquely identifies each user. We recommend hashing their
    /// username or email address, in order to avoid sending us any identifying information.
    pub safety_identifier: Option<String>,
}

impl OpenAIChatOptions {
    /// Extract OpenAI-specific options from provider_options
    pub fn from_provider_options(
        provider_options: &Option<HashMap<String, ai_sdk_provider::json_value::JsonObject>>,
    ) -> Self {
        let mut opts = OpenAIChatOptions::default();

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
