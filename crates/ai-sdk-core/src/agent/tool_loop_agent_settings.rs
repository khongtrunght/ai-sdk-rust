use super::step_result::StepResult;
use super::stop_condition::StopCondition;
use super::tool_loop_agent::ToolLoopAgent;
use crate::tool::Tool;
use ai_sdk_provider::language_model::{LanguageModel, Message, ToolChoice, Usage};
use futures::future::BoxFuture;
use std::sync::Arc;

/// Settings for configuring a ToolLoopAgent
pub struct ToolLoopAgentSettings {
    /// Optional agent identifier
    pub id: Option<String>,

    /// System prompt instructions
    pub instructions: Option<String>,

    /// Language model to use
    pub model: Arc<dyn LanguageModel>,

    /// Executable tools available to the agent (from ai-sdk-core)
    pub tools: Vec<Arc<dyn Tool>>,

    /// Tool choice strategy (default: Auto)
    pub tool_choice: ToolChoice,

    /// Stop conditions (default: step_count_is(20))
    pub stop_conditions: Vec<StopCondition>,

    /// Callback after each step
    pub on_step_finish: Option<OnStepFinishCallback>,

    /// Callback after all steps complete
    pub on_finish: Option<OnFinishCallback>,

    /// Hook to modify parameters before call
    pub prepare_call: Option<PrepareCallFn>,

    /// Hook to modify settings per step
    pub prepare_step: Option<PrepareStepFn>,
}

// Callback type definitions

/// Callback invoked after each step completes
pub type OnStepFinishCallback = Arc<dyn Fn(StepResult) -> BoxFuture<'static, ()> + Send + Sync>;

/// Callback invoked after all steps complete
pub type OnFinishCallback = Arc<dyn Fn(FinishContext) -> BoxFuture<'static, ()> + Send + Sync>;

/// Function to prepare the call before starting
pub type PrepareCallFn = Arc<
    dyn Fn(
            PrepareCallContext,
        ) -> BoxFuture<'static, Result<Vec<Message>, crate::error::GenerateTextError>>
        + Send
        + Sync,
>;

/// Function to prepare each step
pub type PrepareStepFn = Arc<
    dyn Fn(PrepareStepContext) -> BoxFuture<'static, Result<(), crate::error::GenerateTextError>>
        + Send
        + Sync,
>;

/// Context provided to on_finish callback
pub struct FinishContext {
    /// All steps executed
    pub steps: Vec<StepResult>,
    /// Total token usage across all steps
    pub total_usage: Usage,
    /// Last step result
    pub last_step: StepResult,
}

/// Context provided to prepare_call hook
pub struct PrepareCallContext<'a> {
    /// Initial messages
    pub messages: Vec<Message>,
    /// Reference to the agent
    pub agent: &'a ToolLoopAgent,
}

/// Context provided to prepare_step hook
pub struct PrepareStepContext<'a> {
    /// Reference to the model
    pub model: &'a dyn LanguageModel,
    /// All previous steps
    pub steps: &'a [StepResult],
    /// Current step number
    pub step_number: usize,
    /// Current messages
    pub messages: &'a [Message],
}

impl ToolLoopAgentSettings {
    /// Create a new builder for ToolLoopAgentSettings
    pub fn builder(model: Arc<dyn LanguageModel>) -> ToolLoopAgentSettingsBuilder {
        ToolLoopAgentSettingsBuilder {
            id: None,
            instructions: None,
            model,
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            stop_conditions: Vec::new(),
            on_step_finish: None,
            on_finish: None,
            prepare_call: None,
            prepare_step: None,
        }
    }
}

/// Builder for ToolLoopAgentSettings
pub struct ToolLoopAgentSettingsBuilder {
    id: Option<String>,
    instructions: Option<String>,
    model: Arc<dyn LanguageModel>,
    tools: Vec<Arc<dyn Tool>>,
    tool_choice: ToolChoice,
    stop_conditions: Vec<StopCondition>,
    on_step_finish: Option<OnStepFinishCallback>,
    on_finish: Option<OnFinishCallback>,
    prepare_call: Option<PrepareCallFn>,
    prepare_step: Option<PrepareStepFn>,
}

impl ToolLoopAgentSettingsBuilder {
    /// Set agent ID
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set system instructions
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Set executable tools available to the agent (from ai-sdk-core)
    pub fn tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools = tools;
        self
    }

    /// Set tool choice strategy
    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = tool_choice;
        self
    }

    /// Set stop conditions
    pub fn stop_conditions(mut self, stop_conditions: Vec<StopCondition>) -> Self {
        self.stop_conditions = stop_conditions;
        self
    }

    /// Set on_step_finish callback
    pub fn on_step_finish(mut self, callback: OnStepFinishCallback) -> Self {
        self.on_step_finish = Some(callback);
        self
    }

    /// Set on_finish callback
    pub fn on_finish(mut self, callback: OnFinishCallback) -> Self {
        self.on_finish = Some(callback);
        self
    }

    /// Set prepare_call hook
    pub fn prepare_call(mut self, hook: PrepareCallFn) -> Self {
        self.prepare_call = Some(hook);
        self
    }

    /// Set prepare_step hook
    pub fn prepare_step(mut self, hook: PrepareStepFn) -> Self {
        self.prepare_step = Some(hook);
        self
    }

    /// Build the settings
    pub fn build(self) -> ToolLoopAgentSettings {
        ToolLoopAgentSettings {
            id: self.id,
            instructions: self.instructions,
            model: self.model,
            tools: self.tools,
            tool_choice: self.tool_choice,
            stop_conditions: self.stop_conditions,
            on_step_finish: self.on_step_finish,
            on_finish: self.on_finish,
            prepare_call: self.prepare_call,
            prepare_step: self.prepare_step,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock model for testing
    struct MockModel;

    #[async_trait::async_trait]
    impl LanguageModel for MockModel {
        fn specification_version(&self) -> &str {
            "v3"
        }

        fn provider(&self) -> &str {
            "mock"
        }

        fn model_id(&self) -> &str {
            "mock-1"
        }

        async fn supported_urls(&self) -> std::collections::HashMap<String, Vec<String>> {
            std::collections::HashMap::new()
        }

        async fn do_generate(
            &self,
            _options: ai_sdk_provider::language_model::CallOptions,
        ) -> Result<
            ai_sdk_provider::language_model::GenerateResponse,
            Box<dyn std::error::Error + Send + Sync>,
        > {
            unimplemented!()
        }

        async fn do_stream(
            &self,
            _options: ai_sdk_provider::language_model::CallOptions,
        ) -> Result<
            ai_sdk_provider::language_model::StreamResponse,
            Box<dyn std::error::Error + Send + Sync + 'static>,
        > {
            unimplemented!()
        }
    }

    #[test]
    fn test_settings_builder() {
        let model: Arc<dyn LanguageModel> = Arc::new(MockModel);
        let settings = ToolLoopAgentSettings::builder(model)
            .id("test-agent")
            .instructions("You are a helpful assistant")
            .tool_choice(ToolChoice::Auto)
            .build();

        assert_eq!(settings.id, Some("test-agent".to_string()));
        assert_eq!(
            settings.instructions,
            Some("You are a helpful assistant".to_string())
        );
        assert_eq!(settings.tool_choice, ToolChoice::Auto);
    }
}
