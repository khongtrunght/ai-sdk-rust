use ai_sdk_provider::FinishReason;

/// Trait for determining when to stop the tool execution loop
pub trait StopCondition: Send + Sync {
    fn should_stop(&self, step: u32, finish_reason: &FinishReason) -> bool;
}

/// Stop after a maximum number of steps
pub struct StopAfterSteps {
    max_steps: u32,
}

impl StopAfterSteps {
    pub fn new(max_steps: u32) -> Self {
        Self { max_steps }
    }
}

impl StopCondition for StopAfterSteps {
    fn should_stop(&self, step: u32, _finish_reason: &FinishReason) -> bool {
        step >= self.max_steps
    }
}

/// Stop when model returns a finish reason other than ToolCalls
pub struct StopOnFinish;

impl StopCondition for StopOnFinish {
    fn should_stop(&self, _step: u32, finish_reason: &FinishReason) -> bool {
        !matches!(finish_reason, FinishReason::ToolCalls)
    }
}

/// Helper functions to create stop conditions
pub fn stop_after_steps(max_steps: u32) -> Box<dyn StopCondition> {
    Box::new(StopAfterSteps::new(max_steps))
}

pub fn stop_on_finish() -> Box<dyn StopCondition> {
    Box::new(StopOnFinish)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stop_after_steps() {
        let condition = StopAfterSteps::new(3);
        assert!(!condition.should_stop(0, &FinishReason::Stop));
        assert!(!condition.should_stop(2, &FinishReason::Stop));
        assert!(condition.should_stop(3, &FinishReason::Stop));
        assert!(condition.should_stop(4, &FinishReason::Stop));
    }

    #[test]
    fn test_stop_on_finish() {
        let condition = StopOnFinish;
        assert!(condition.should_stop(0, &FinishReason::Stop));
        assert!(condition.should_stop(0, &FinishReason::Length));
        assert!(!condition.should_stop(0, &FinishReason::ToolCalls));
    }
}
