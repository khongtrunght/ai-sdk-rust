use super::step_result::StepResult;
use futures::future::BoxFuture;

/// Function type for stop conditions
pub type StopConditionFn =
    Box<dyn Fn(&StopConditionContext) -> BoxFuture<'static, bool> + Send + Sync>;

/// Context provided to stop conditions for evaluation
#[derive(Clone)]
pub struct StopConditionContext {
    /// All steps executed so far
    pub steps: Vec<StepResult>,
}

/// A condition that determines when to stop the agent loop
pub struct StopCondition {
    condition: StopConditionFn,
}

// Implement Clone manually for StopCondition
impl Clone for StopCondition {
    fn clone(&self) -> Self {
        // We can't actually clone the boxed function, so we panic
        // In practice, stop conditions should be created fresh rather than cloned
        // Or we need to use Arc instead
        panic!("StopCondition cannot be cloned - use Arc<StopCondition> instead")
    }
}

impl StopCondition {
    /// Create a new stop condition from a function
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&StopConditionContext) -> BoxFuture<'static, bool> + Send + Sync + 'static,
    {
        Self {
            condition: Box::new(f),
        }
    }

    /// Evaluate this stop condition
    pub async fn evaluate(&self, context: &StopConditionContext) -> bool {
        (self.condition)(context).await
    }
}

/// Check if any stop condition is met
pub async fn is_stop_condition_met(conditions: &[StopCondition], steps: &[StepResult]) -> bool {
    if conditions.is_empty() {
        return false;
    }

    let context = StopConditionContext {
        steps: steps.to_vec(),
    };

    // Evaluate all conditions in parallel
    let results = futures::future::join_all(conditions.iter().map(|c| c.evaluate(&context))).await;

    // Return true if ANY condition is met
    results.iter().any(|&result| result)
}

/// Stop after a certain number of steps
pub fn step_count_is(count: usize) -> StopCondition {
    StopCondition::new(move |ctx| {
        let steps_len = ctx.steps.len();
        Box::pin(async move { steps_len >= count })
    })
}

/// Stop when a specific tool was called in the last step
pub fn has_tool_call(tool_name: String) -> StopCondition {
    StopCondition::new(move |ctx| {
        let has_call = ctx
            .steps
            .last()
            .and_then(|step| step.tool_calls.as_ref())
            .map(|calls| calls.iter().any(|call| call.tool_name == tool_name))
            .unwrap_or(false);
        Box::pin(async move { has_call })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_sdk_provider::language_model::{Content, FinishReason, TextPart, ToolCallPart, Usage};

    fn create_test_step(tool_name: Option<&str>) -> StepResult {
        let content = if let Some(name) = tool_name {
            vec![Content::ToolCall(ToolCallPart {
                tool_call_id: "call_1".to_string(),
                tool_name: name.to_string(),
                input: "{}".to_string(),
                provider_executed: None,
                dynamic: None,
                provider_metadata: None,
            })]
        } else {
            vec![Content::Text(TextPart {
                text: "Hello".to_string(),
                provider_metadata: None,
            })]
        };

        let tool_calls = StepResult::extract_tool_calls(&content);

        StepResult {
            content,
            tool_calls,
            tool_results: None,
            text: "Hello".to_string(),
            reasoning_text: None,
            finish_reason: FinishReason::Stop,
            usage: Usage {
                input_tokens: Some(10),
                output_tokens: Some(20),
                total_tokens: Some(30),
                reasoning_tokens: None,
                cached_input_tokens: None,
            },
            warnings: vec![],
            request: None,
            response: None,
            provider_metadata: None,
        }
    }

    #[tokio::test]
    async fn test_step_count_is() {
        let condition = step_count_is(3);
        let context = StopConditionContext {
            steps: vec![create_test_step(None), create_test_step(None)],
        };

        assert!(!condition.evaluate(&context).await);

        let context = StopConditionContext {
            steps: vec![
                create_test_step(None),
                create_test_step(None),
                create_test_step(None),
            ],
        };

        assert!(condition.evaluate(&context).await);
    }

    #[tokio::test]
    async fn test_has_tool_call() {
        let condition = has_tool_call("weather".to_string());

        let context = StopConditionContext {
            steps: vec![create_test_step(Some("weather"))],
        };

        assert!(condition.evaluate(&context).await);

        let context = StopConditionContext {
            steps: vec![create_test_step(Some("other_tool"))],
        };

        assert!(!condition.evaluate(&context).await);
    }

    #[tokio::test]
    async fn test_is_stop_condition_met() {
        let conditions = vec![step_count_is(2), has_tool_call("weather".to_string())];

        // No conditions met
        let steps = vec![create_test_step(None)];
        assert!(!is_stop_condition_met(&conditions, &steps).await);

        // Step count condition met
        let steps = vec![create_test_step(None), create_test_step(None)];
        assert!(is_stop_condition_met(&conditions, &steps).await);

        // Tool call condition met
        let steps = vec![create_test_step(Some("weather"))];
        assert!(is_stop_condition_met(&conditions, &steps).await);
    }
}
