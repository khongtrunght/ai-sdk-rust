use ai_sdk_core::JsonValue;
use ai_sdk_provider::SharedProviderOptions;

pub struct OpenAIEmbeddingProviderOptions {
    pub dimensions: Option<u32>,
    pub user: Option<String>,
}

impl From<Option<SharedProviderOptions>> for OpenAIEmbeddingProviderOptions {
    fn from(opts: Option<SharedProviderOptions>) -> Self {
        match opts {
            None => OpenAIEmbeddingProviderOptions {
                dimensions: None,
                user: None,
            },
            Some(opts) => {
                let openai_opts = opts.get("openai");
                OpenAIEmbeddingProviderOptions {
                    dimensions: openai_opts.and_then(|o| o.get("dimensions")).and_then(
                        |d| match d {
                            JsonValue::Number(n) => n.as_u64().map(|n| n as u32),
                            _ => None,
                        },
                    ),
                    user: openai_opts
                        .and_then(|o| o.get("user"))
                        .and_then(|u| match u {
                            JsonValue::String(s) => Some(s.clone()),
                            _ => None,
                        }),
                }
            }
        }
    }
}
