//! Agent framework for autonomous tool-using agents
//!
//! This module provides the agent abstraction that enables autonomous behavior
//! through tool execution loops with stop conditions and callbacks.

#[allow(clippy::module_inception)]
mod agent;
mod step_result;
mod stop_condition;
mod tool_loop_agent;
mod tool_loop_agent_settings;

pub use agent::{Agent, AgentCallParameters};
pub use step_result::StepResult;
pub use stop_condition::{
    has_tool_call, is_stop_condition_met, step_count_is, StopCondition, StopConditionContext,
};
pub use tool_loop_agent::ToolLoopAgent;
pub use tool_loop_agent_settings::{
    FinishContext, OnFinishCallback, OnStepFinishCallback, PrepareCallContext, PrepareCallFn,
    PrepareStepContext, PrepareStepFn, ToolLoopAgentSettings,
};
