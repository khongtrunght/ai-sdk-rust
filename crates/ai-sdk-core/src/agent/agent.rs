use crate::error::GenerateTextError;
use crate::tool::Tool;
use crate::GenerateTextResult;
use crate::StreamTextResult;
use ai_sdk_provider::language_model::Message;
use async_trait::async_trait;
use std::sync::Arc;

/// Core agent interface for autonomous behavior
#[async_trait]
pub trait Agent: Send + Sync {
    /// Agent version for backwards compatibility
    fn version(&self) -> &str {
        "agent-v1"
    }

    /// Optional identifier for the agent
    fn id(&self) -> Option<&str>;

    /// Executable tools available to the agent (from ai-sdk-core)
    fn tools(&self) -> &[Arc<dyn Tool>];

    /// Generate response (non-streaming)
    async fn generate(
        &self,
        params: AgentCallParameters,
    ) -> Result<GenerateTextResult, GenerateTextError>;

    /// Generate response (streaming)
    async fn stream(
        &self,
        params: AgentCallParameters,
    ) -> Result<StreamTextResult, GenerateTextError>;
}

/// Parameters for calling an agent
pub struct AgentCallParameters {
    /// Simple text prompt (alternative to messages)
    pub prompt: Option<String>,

    /// Full message array (alternative to prompt)
    pub messages: Option<Vec<Message>>,

    /// Optional abort signal (future enhancement)
    pub abort_signal: Option<()>,
}

impl AgentCallParameters {
    /// Create parameters from a simple prompt
    pub fn from_prompt(prompt: impl Into<String>) -> Self {
        Self {
            prompt: Some(prompt.into()),
            messages: None,
            abort_signal: None,
        }
    }

    /// Create parameters from messages
    pub fn from_messages(messages: Vec<Message>) -> Self {
        Self {
            prompt: None,
            messages: Some(messages),
            abort_signal: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_call_parameters_from_prompt() {
        let params = AgentCallParameters::from_prompt("Hello world");
        assert!(params.prompt.is_some());
        assert!(params.messages.is_none());
        assert_eq!(params.prompt.unwrap(), "Hello world");
    }

    #[test]
    fn test_agent_call_parameters_from_messages() {
        let messages = vec![];
        let params = AgentCallParameters::from_messages(messages);
        assert!(params.prompt.is_none());
        assert!(params.messages.is_some());
    }
}
