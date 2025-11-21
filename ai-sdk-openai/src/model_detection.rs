//! Model detection utilities for OpenAI models.
//!
//! This module provides functions to detect model types and capabilities
//! based on model IDs. Different model families require different API
//! parameters and behaviors.

/// Check if model is a reasoning model (o1, o3, o4, etc.)
///
/// Reasoning models use `max_completion_tokens` instead of `max_tokens`.
///
/// # Examples
///
/// ```
/// use ai_sdk_openai::model_detection::is_reasoning_model;
///
/// assert!(is_reasoning_model("o1"));
/// assert!(is_reasoning_model("o1-preview"));
/// assert!(is_reasoning_model("o3-mini"));
/// assert!(is_reasoning_model("o4-mini"));
/// assert!(!is_reasoning_model("gpt-4o"));
/// ```
pub fn is_reasoning_model(model_id: &str) -> bool {
    model_id.starts_with("o1") || model_id.starts_with("o3") || model_id.starts_with("o4")
}

/// Check if model is a search preview model
///
/// Search preview models do not support the `temperature` parameter.
///
/// # Examples
///
/// ```
/// use ai_sdk_openai::model_detection::is_search_preview_model;
///
/// assert!(is_search_preview_model("gpt-4o-search-preview"));
/// assert!(is_search_preview_model("gpt-4-search-preview"));
/// assert!(!is_search_preview_model("gpt-4o"));
/// assert!(!is_search_preview_model("o4-mini"));
/// ```
pub fn is_search_preview_model(model_id: &str) -> bool {
    model_id.contains("search-preview")
}

/// Check if model supports flex processing
///
/// Only certain models support the `service_tier: "flex"` option.
/// Supported models: o3, o4-mini, gpt-5
///
/// # Examples
///
/// ```
/// use ai_sdk_openai::model_detection::supports_flex_processing;
///
/// assert!(supports_flex_processing("o3"));
/// assert!(supports_flex_processing("o3-mini"));
/// assert!(supports_flex_processing("o4-mini"));
/// assert!(supports_flex_processing("gpt-5"));
/// assert!(!supports_flex_processing("gpt-4o"));
/// assert!(!supports_flex_processing("gpt-4o-mini"));
/// ```
pub fn supports_flex_processing(model_id: &str) -> bool {
    model_id.starts_with("o3")
        || model_id.starts_with("o4")
        || (model_id.starts_with("gpt-5") && !model_id.starts_with("gpt-5-chat"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_reasoning_model() {
        // O1 models
        assert!(is_reasoning_model("o1"));
        assert!(is_reasoning_model("o1-preview"));
        assert!(is_reasoning_model("o1-mini"));

        // O3 models
        assert!(is_reasoning_model("o3"));
        assert!(is_reasoning_model("o3-mini"));

        // O4 models
        assert!(is_reasoning_model("o4"));
        assert!(is_reasoning_model("o4-mini"));

        // Non-reasoning models
        assert!(!is_reasoning_model("gpt-4"));
        assert!(!is_reasoning_model("gpt-4o"));
        assert!(!is_reasoning_model("gpt-4o-mini"));
        assert!(!is_reasoning_model("gpt-3.5-turbo"));
        assert!(!is_reasoning_model("gpt-5"));
    }

    #[test]
    fn test_is_search_preview_model() {
        // Search preview models
        assert!(is_search_preview_model("gpt-4o-search-preview"));
        assert!(is_search_preview_model("gpt-4-search-preview"));
        assert!(is_search_preview_model("gpt-4o-mini-search-preview"));

        // Not search preview models
        assert!(!is_search_preview_model("gpt-4o"));
        assert!(!is_search_preview_model("gpt-4"));
        assert!(!is_search_preview_model("o4-mini"));
        assert!(!is_search_preview_model("gpt-3.5-turbo"));
    }

    #[test]
    fn test_supports_flex_processing() {
        // Supported models
        assert!(supports_flex_processing("o3"));
        assert!(supports_flex_processing("o3-mini"));
        assert!(supports_flex_processing("o4-mini"));
        assert!(supports_flex_processing("gpt-5"));
        assert!(supports_flex_processing("gpt-5-turbo"));

        // Not supported models
        assert!(!supports_flex_processing("gpt-4o"));
        assert!(!supports_flex_processing("gpt-4o-mini"));
        assert!(!supports_flex_processing("gpt-4"));
        assert!(!supports_flex_processing("o1"));
        assert!(!supports_flex_processing("o1-mini"));
    }
}
