use ai_sdk_provider::json_value::JsonValue;
use ai_sdk_provider::SharedProviderOptions;

/// OpenAI-specific options for the Responses API
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct OpenAIResponsesProviderOptions {
    /// Conversation ID for continuing a conversation
    pub conversation: Option<String>,

    /// Include options for additional response data
    pub include: Option<Vec<String>>,

    /// System instructions for the model
    pub instructions: Option<String>,

    /// Return log probabilities of tokens (true or number 1-20)
    pub logprobs: Option<LogprobsOption>,

    /// Maximum number of tool calls allowed
    pub max_tool_calls: Option<u32>,

    /// Metadata for the request
    pub metadata: Option<serde_json::Value>,

    /// Enable parallel tool calls
    pub parallel_tool_calls: Option<bool>,

    /// Previous response ID to continue from
    pub previous_response_id: Option<String>,

    /// Prompt cache key for manual caching control
    pub prompt_cache_key: Option<String>,

    /// Prompt cache retention policy ('in_memory' or '24h')
    pub prompt_cache_retention: Option<String>,

    /// Reasoning effort for reasoning models
    pub reasoning_effort: Option<String>,

    /// Reasoning summary format
    pub reasoning_summary: Option<String>,

    /// Safety identifier for user monitoring
    pub safety_identifier: Option<String>,

    /// Service tier ('auto', 'flex', 'priority', 'default')
    pub service_tier: Option<String>,

    /// Whether to store the response (defaults to true)
    pub store: Option<bool>,

    /// Use strict JSON schema validation
    pub strict_json_schema: Option<bool>,

    /// Text verbosity level ('low', 'medium', 'high')
    pub text_verbosity: Option<String>,

    /// Truncation strategy ('auto', 'disabled')
    pub truncation: Option<String>,

    /// End-user identifier
    pub user: Option<String>,
}

/// Logprobs can be a boolean or a number (1-20)
#[derive(Debug, Clone, Copy)]
pub enum LogprobsOption {
    Enabled,
    TopN(u8), // 1-20
}

impl From<Option<SharedProviderOptions>> for OpenAIResponsesProviderOptions {
    fn from(opts: Option<SharedProviderOptions>) -> Self {
        match opts {
            None => Self::default(),
            Some(opts) => {
                let openai_opts = opts.get("openai");

                Self {
                    conversation: openai_opts
                        .and_then(|o| o.get("conversation"))
                        .and_then(|v| match v {
                            JsonValue::String(s) => Some(s.clone()),
                            _ => None,
                        }),

                    include: openai_opts
                        .and_then(|o| o.get("include"))
                        .and_then(|v| match v {
                            JsonValue::Array(arr) => Some(
                                arr.iter()
                                    .filter_map(|v| match v {
                                        JsonValue::String(s) => Some(s.clone()),
                                        _ => None,
                                    })
                                    .collect(),
                            ),
                            _ => None,
                        }),

                    instructions: openai_opts
                        .and_then(|o| o.get("instructions"))
                        .and_then(|v| match v {
                            JsonValue::String(s) => Some(s.clone()),
                            _ => None,
                        }),

                    logprobs: openai_opts
                        .and_then(|o| o.get("logprobs"))
                        .and_then(|v| match v {
                            JsonValue::Bool(true) => Some(LogprobsOption::Enabled),
                            JsonValue::Number(n) => {
                                let num = n.as_u64()? as u8;
                                if (1..=20).contains(&num) {
                                    Some(LogprobsOption::TopN(num))
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        }),

                    max_tool_calls: openai_opts
                        .and_then(|o| o.get("maxToolCalls"))
                        .and_then(|v| match v {
                            JsonValue::Number(n) => Some(n.as_u64()? as u32),
                            _ => None,
                        }),

                    metadata: openai_opts
                        .and_then(|o| o.get("metadata"))
                        .and_then(|v| serde_json::to_value(v).ok()),

                    parallel_tool_calls: openai_opts
                        .and_then(|o| o.get("parallelToolCalls"))
                        .and_then(|v| match v {
                            JsonValue::Bool(b) => Some(*b),
                            _ => None,
                        }),

                    previous_response_id: openai_opts
                        .and_then(|o| o.get("previousResponseId"))
                        .and_then(|v| match v {
                            JsonValue::String(s) => Some(s.clone()),
                            _ => None,
                        }),

                    prompt_cache_key: openai_opts.and_then(|o| o.get("promptCacheKey")).and_then(
                        |v| match v {
                            JsonValue::String(s) => Some(s.clone()),
                            _ => None,
                        },
                    ),

                    prompt_cache_retention: openai_opts
                        .and_then(|o| o.get("promptCacheRetention"))
                        .and_then(|v| match v {
                            JsonValue::String(s) => Some(s.clone()),
                            _ => None,
                        }),

                    reasoning_effort: openai_opts.and_then(|o| o.get("reasoningEffort")).and_then(
                        |v| match v {
                            JsonValue::String(s) => Some(s.clone()),
                            _ => None,
                        },
                    ),

                    reasoning_summary: openai_opts
                        .and_then(|o| o.get("reasoningSummary"))
                        .and_then(|v| match v {
                            JsonValue::String(s) => Some(s.clone()),
                            _ => None,
                        }),

                    safety_identifier: openai_opts
                        .and_then(|o| o.get("safetyIdentifier"))
                        .and_then(|v| match v {
                            JsonValue::String(s) => Some(s.clone()),
                            _ => None,
                        }),

                    service_tier: openai_opts
                        .and_then(|o| o.get("serviceTier"))
                        .and_then(|v| match v {
                            JsonValue::String(s) => Some(s.clone()),
                            _ => None,
                        }),

                    store: openai_opts
                        .and_then(|o| o.get("store"))
                        .and_then(|v| match v {
                            JsonValue::Bool(b) => Some(*b),
                            _ => None,
                        }),

                    strict_json_schema: openai_opts
                        .and_then(|o| o.get("strictJsonSchema"))
                        .and_then(|v| match v {
                            JsonValue::Bool(b) => Some(*b),
                            _ => None,
                        }),

                    text_verbosity: openai_opts.and_then(|o| o.get("textVerbosity")).and_then(
                        |v| match v {
                            JsonValue::String(s) => Some(s.clone()),
                            _ => None,
                        },
                    ),

                    truncation: openai_opts.and_then(|o| o.get("truncation")).and_then(
                        |v| match v {
                            JsonValue::String(s) => Some(s.clone()),
                            _ => None,
                        },
                    ),

                    user: openai_opts
                        .and_then(|o| o.get("user"))
                        .and_then(|v| match v {
                            JsonValue::String(s) => Some(s.clone()),
                            _ => None,
                        }),
                }
            }
        }
    }
}
