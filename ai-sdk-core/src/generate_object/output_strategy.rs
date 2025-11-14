//! Output strategy pattern for different object generation modes.
//!
//! This module defines the `OutputStrategy` trait and provides implementations for:
//! - Object output (single structured object)
//! - Array output (array of elements with validation)
//! - Enum output (single enum value)
//! - No-schema output (unvalidated JSON)

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::marker::PhantomData;
use thiserror::Error;

use ai_sdk_provider::language_model::{ResponseMetadata, Usage};

/// The type of output being generated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputType {
    /// Single structured object
    Object,
    /// Array of elements
    Array,
    /// Single enum value
    Enum,
    /// Unvalidated JSON
    NoSchema,
}

/// Result of partial validation during streaming.
#[derive(Debug, Clone)]
pub struct PartialValidation<P> {
    /// The partial value that was validated
    pub partial: P,
    /// Text delta that was processed
    pub text_delta: String,
}

/// Context for final validation.
#[derive(Debug, Clone)]
pub struct ValidationContext {
    /// Complete generated text
    pub text: String,
    /// Response metadata from the model
    pub response: Option<ResponseMetadata>,
    /// Token usage statistics
    pub usage: Usage,
}

/// Result of validation (either partial or final).
#[derive(Debug)]
pub enum ValidationResult<T> {
    /// Validation succeeded
    Success {
        /// The validated value
        value: T,
        /// Raw JSON value before validation
        raw_value: Value,
    },
    /// Validation failed
    Failure {
        /// The validation error
        error: ValidationError,
        /// Raw JSON value that failed validation
        raw_value: Value,
    },
}

/// Errors that can occur during validation.
#[derive(Debug, Error)]
pub enum ValidationError {
    /// JSON parsing error
    #[error("JSON parsing error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Schema validation error
    #[error("Schema validation error: {0}")]
    SchemaError(String),

    /// Type mismatch error
    #[error("Type mismatch: expected {expected}, got {got}")]
    TypeMismatch {
        /// Expected type
        expected: String,
        /// Actual type
        got: String,
    },

    /// Custom validation error
    #[error("Validation error: {0}")]
    Custom(String),
}

/// Strategy trait for handling different output types.
///
/// This trait defines how to:
/// - Generate JSON schema for the output
/// - Validate partial results during streaming
/// - Validate final results
/// - Transform streaming output
#[async_trait]
pub trait OutputStrategy: Send + Sync {
    /// The type of partial values during streaming
    type Partial: Clone + Send;

    /// The type of final results
    type Result: Send;

    /// Returns the output type for this strategy
    fn output_type(&self) -> OutputType;

    /// Returns the JSON schema for this output type, if any
    async fn json_schema(&self) -> Option<Value>;

    /// Validates a partial result during streaming
    async fn validate_partial_result(
        &self,
        value: Value,
        text_delta: String,
        is_first_delta: bool,
        is_final_delta: bool,
        latest_object: Option<&Self::Partial>,
    ) -> ValidationResult<PartialValidation<Self::Partial>>;

    /// Validates the final result after generation completes
    async fn validate_final_result(
        &self,
        value: Option<Value>,
        context: ValidationContext,
    ) -> ValidationResult<Self::Result>;
}

/// Strategy for generating single structured objects.
///
/// This strategy validates objects against a JSON schema and returns
/// the complete object as the final result.
pub struct ObjectOutputStrategy<T> {
    schema: Value,
    _phantom: PhantomData<T>,
}

impl<T> ObjectOutputStrategy<T> {
    /// Creates a new object output strategy with the given JSON schema.
    pub fn new(schema: Value) -> Self {
        Self {
            schema,
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<T> OutputStrategy for ObjectOutputStrategy<T>
where
    T: DeserializeOwned + Clone + Send + Sync + 'static,
{
    type Partial = T;
    type Result = T;

    fn output_type(&self) -> OutputType {
        OutputType::Object
    }

    async fn json_schema(&self) -> Option<Value> {
        Some(self.schema.clone())
    }

    async fn validate_partial_result(
        &self,
        value: Value,
        text_delta: String,
        _is_first_delta: bool,
        _is_final_delta: bool,
        _latest_object: Option<&Self::Partial>,
    ) -> ValidationResult<PartialValidation<Self::Partial>> {
        // For partial results, we don't validate against schema
        // Just try to deserialize as T (allowing partial data)
        match serde_json::from_value::<T>(value.clone()) {
            Ok(partial) => ValidationResult::Success {
                value: PartialValidation {
                    partial,
                    text_delta,
                },
                raw_value: value,
            },
            Err(e) => ValidationResult::Failure {
                error: ValidationError::JsonError(e),
                raw_value: value,
            },
        }
    }

    async fn validate_final_result(
        &self,
        value: Option<Value>,
        _context: ValidationContext,
    ) -> ValidationResult<Self::Result> {
        let Some(value) = value else {
            return ValidationResult::Failure {
                error: ValidationError::Custom("No value provided".to_string()),
                raw_value: Value::Null,
            };
        };

        // Validate against schema (for now, just try to deserialize)
        // TODO: Implement full JSON schema validation
        match serde_json::from_value::<T>(value.clone()) {
            Ok(result) => ValidationResult::Success {
                value: result,
                raw_value: value,
            },
            Err(e) => ValidationResult::Failure {
                error: ValidationError::JsonError(e),
                raw_value: value,
            },
        }
    }
}

/// Strategy for generating arrays of elements.
///
/// This strategy wraps the array in `{"elements": [...]}` for better LLM reliability
/// and validates each element (except the last incomplete one during streaming).
pub struct ArrayOutputStrategy<T> {
    element_schema: Value,
    _phantom: PhantomData<T>,
}

impl<T> ArrayOutputStrategy<T> {
    /// Creates a new array output strategy with the given element schema.
    pub fn new(element_schema: Value) -> Self {
        Self {
            element_schema,
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<T> OutputStrategy for ArrayOutputStrategy<T>
where
    T: DeserializeOwned + Clone + Send + Sync + 'static,
{
    type Partial = Vec<T>;
    type Result = Vec<T>;

    fn output_type(&self) -> OutputType {
        OutputType::Array
    }

    async fn json_schema(&self) -> Option<Value> {
        // Wrap element schema in array wrapper
        Some(serde_json::json!({
            "type": "object",
            "properties": {
                "elements": {
                    "type": "array",
                    "items": self.element_schema
                }
            },
            "required": ["elements"],
            "additionalProperties": false
        }))
    }

    async fn validate_partial_result(
        &self,
        value: Value,
        text_delta: String,
        _is_first_delta: bool,
        is_final_delta: bool,
        _latest_object: Option<&Self::Partial>,
    ) -> ValidationResult<PartialValidation<Self::Partial>> {
        // Extract elements array from wrapper
        let elements = match value.get("elements") {
            Some(Value::Array(arr)) => arr,
            _ => {
                return ValidationResult::Failure {
                    error: ValidationError::TypeMismatch {
                        expected: "object with elements array".to_string(),
                        got: format!("{:?}", value),
                    },
                    raw_value: value,
                };
            }
        };

        // For streaming, validate all elements except possibly the last
        let validate_count = if is_final_delta {
            elements.len()
        } else {
            elements.len().saturating_sub(1)
        };

        let mut validated_elements = Vec::new();
        for elem_value in elements.iter().take(validate_count) {
            match serde_json::from_value::<T>(elem_value.clone()) {
                Ok(elem) => validated_elements.push(elem),
                Err(e) => {
                    return ValidationResult::Failure {
                        error: ValidationError::JsonError(e),
                        raw_value: value,
                    };
                }
            }
        }

        // Add last element without validation if not final
        if !is_final_delta && elements.len() > validate_count {
            if let Ok(elem) = serde_json::from_value::<T>(elements[validate_count].clone()) {
                validated_elements.push(elem);
            }
        }

        ValidationResult::Success {
            value: PartialValidation {
                partial: validated_elements,
                text_delta,
            },
            raw_value: value,
        }
    }

    async fn validate_final_result(
        &self,
        value: Option<Value>,
        _context: ValidationContext,
    ) -> ValidationResult<Self::Result> {
        let Some(value) = value else {
            return ValidationResult::Failure {
                error: ValidationError::Custom("No value provided".to_string()),
                raw_value: Value::Null,
            };
        };

        // Extract and validate all elements
        let elements = match value.get("elements") {
            Some(Value::Array(arr)) => arr,
            _ => {
                return ValidationResult::Failure {
                    error: ValidationError::TypeMismatch {
                        expected: "object with elements array".to_string(),
                        got: format!("{:?}", value),
                    },
                    raw_value: value,
                };
            }
        };

        let mut validated_elements = Vec::new();
        for elem_value in elements {
            match serde_json::from_value::<T>(elem_value.clone()) {
                Ok(elem) => validated_elements.push(elem),
                Err(e) => {
                    return ValidationResult::Failure {
                        error: ValidationError::JsonError(e),
                        raw_value: value,
                    };
                }
            }
        }

        ValidationResult::Success {
            value: validated_elements,
            raw_value: value,
        }
    }
}

/// Strategy for generating enum values.
///
/// This strategy wraps the enum value in `{"result": "value"}` and validates
/// against a list of allowed values.
pub struct EnumOutputStrategy {
    enum_values: Vec<String>,
}

impl EnumOutputStrategy {
    /// Creates a new enum output strategy with the given allowed values.
    pub fn new(enum_values: Vec<String>) -> Self {
        Self { enum_values }
    }
}

#[async_trait]
impl OutputStrategy for EnumOutputStrategy {
    type Partial = String;
    type Result = String;

    fn output_type(&self) -> OutputType {
        OutputType::Enum
    }

    async fn json_schema(&self) -> Option<Value> {
        Some(serde_json::json!({
            "type": "object",
            "properties": {
                "result": {
                    "type": "string",
                    "enum": self.enum_values
                }
            },
            "required": ["result"],
            "additionalProperties": false
        }))
    }

    async fn validate_partial_result(
        &self,
        value: Value,
        text_delta: String,
        _is_first_delta: bool,
        is_final_delta: bool,
        _latest_object: Option<&Self::Partial>,
    ) -> ValidationResult<PartialValidation<Self::Partial>> {
        let result_value = match value.get("result") {
            Some(Value::String(s)) => s.clone(),
            _ => {
                return ValidationResult::Failure {
                    error: ValidationError::TypeMismatch {
                        expected: "object with result string".to_string(),
                        got: format!("{:?}", value),
                    },
                    raw_value: value,
                };
            }
        };

        // For partial results, check if it's a prefix of any enum value
        if is_final_delta {
            // Final: must match exactly
            if !self.enum_values.contains(&result_value) {
                return ValidationResult::Failure {
                    error: ValidationError::SchemaError(format!(
                        "Value '{}' is not one of: {:?}",
                        result_value, self.enum_values
                    )),
                    raw_value: value,
                };
            }
        } else {
            // Partial: check if it's a prefix
            if !self
                .enum_values
                .iter()
                .any(|ev| ev.starts_with(&result_value))
            {
                return ValidationResult::Failure {
                    error: ValidationError::SchemaError(format!(
                        "Value '{}' is not a prefix of any enum value: {:?}",
                        result_value, self.enum_values
                    )),
                    raw_value: value,
                };
            }
        }

        ValidationResult::Success {
            value: PartialValidation {
                partial: result_value,
                text_delta,
            },
            raw_value: value,
        }
    }

    async fn validate_final_result(
        &self,
        value: Option<Value>,
        _context: ValidationContext,
    ) -> ValidationResult<Self::Result> {
        let Some(value) = value else {
            return ValidationResult::Failure {
                error: ValidationError::Custom("No value provided".to_string()),
                raw_value: Value::Null,
            };
        };

        let result_value = match value.get("result") {
            Some(Value::String(s)) => s.clone(),
            _ => {
                return ValidationResult::Failure {
                    error: ValidationError::TypeMismatch {
                        expected: "object with result string".to_string(),
                        got: format!("{:?}", value),
                    },
                    raw_value: value,
                };
            }
        };

        if !self.enum_values.contains(&result_value) {
            return ValidationResult::Failure {
                error: ValidationError::SchemaError(format!(
                    "Value '{}' is not one of: {:?}",
                    result_value, self.enum_values
                )),
                raw_value: value,
            };
        }

        ValidationResult::Success {
            value: result_value,
            raw_value: value,
        }
    }
}

/// Strategy for generating unvalidated JSON.
///
/// This strategy performs no validation and returns raw JSON values.
pub struct NoSchemaOutputStrategy;

#[async_trait]
impl OutputStrategy for NoSchemaOutputStrategy {
    type Partial = Value;
    type Result = Value;

    fn output_type(&self) -> OutputType {
        OutputType::NoSchema
    }

    async fn json_schema(&self) -> Option<Value> {
        None
    }

    async fn validate_partial_result(
        &self,
        value: Value,
        text_delta: String,
        _is_first_delta: bool,
        _is_final_delta: bool,
        _latest_object: Option<&Self::Partial>,
    ) -> ValidationResult<PartialValidation<Self::Partial>> {
        ValidationResult::Success {
            value: PartialValidation {
                partial: value.clone(),
                text_delta,
            },
            raw_value: value,
        }
    }

    async fn validate_final_result(
        &self,
        value: Option<Value>,
        _context: ValidationContext,
    ) -> ValidationResult<Self::Result> {
        let Some(value) = value else {
            return ValidationResult::Failure {
                error: ValidationError::Custom("No value provided".to_string()),
                raw_value: Value::Null,
            };
        };

        ValidationResult::Success {
            value: value.clone(),
            raw_value: value,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestObject {
        name: String,
        age: u32,
    }

    #[tokio::test]
    async fn test_object_strategy() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            }
        });

        let strategy = ObjectOutputStrategy::<TestObject>::new(schema);
        assert_eq!(strategy.output_type(), OutputType::Object);

        let value = serde_json::json!({"name": "Alice", "age": 30});
        let result = strategy
            .validate_final_result(
                Some(value),
                ValidationContext {
                    text: String::new(),
                    response: None,
                    usage: Usage::default(),
                },
            )
            .await;

        match result {
            ValidationResult::Success { value, .. } => {
                assert_eq!(value.name, "Alice");
                assert_eq!(value.age, 30);
            }
            ValidationResult::Failure { error, .. } => {
                panic!("Validation failed: {:?}", error);
            }
        }
    }

    #[tokio::test]
    async fn test_array_strategy() {
        let element_schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        });

        let strategy = ArrayOutputStrategy::<TestObject>::new(element_schema);
        assert_eq!(strategy.output_type(), OutputType::Array);

        let value = serde_json::json!({
            "elements": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25}
            ]
        });

        let result = strategy
            .validate_final_result(
                Some(value),
                ValidationContext {
                    text: String::new(),
                    response: None,
                    usage: Usage::default(),
                },
            )
            .await;

        match result {
            ValidationResult::Success { value, .. } => {
                assert_eq!(value.len(), 2);
                assert_eq!(value[0].name, "Alice");
                assert_eq!(value[1].name, "Bob");
            }
            ValidationResult::Failure { error, .. } => {
                panic!("Validation failed: {:?}", error);
            }
        }
    }

    #[tokio::test]
    async fn test_enum_strategy() {
        let strategy = EnumOutputStrategy::new(vec![
            "option1".to_string(),
            "option2".to_string(),
            "option3".to_string(),
        ]);

        assert_eq!(strategy.output_type(), OutputType::Enum);

        let value = serde_json::json!({"result": "option2"});
        let result = strategy
            .validate_final_result(
                Some(value),
                ValidationContext {
                    text: String::new(),
                    response: None,
                    usage: Usage::default(),
                },
            )
            .await;

        match result {
            ValidationResult::Success { value, .. } => {
                assert_eq!(value, "option2");
            }
            ValidationResult::Failure { error, .. } => {
                panic!("Validation failed: {:?}", error);
            }
        }
    }

    #[tokio::test]
    async fn test_no_schema_strategy() {
        let strategy = NoSchemaOutputStrategy;
        assert_eq!(strategy.output_type(), OutputType::NoSchema);

        let value = serde_json::json!({"any": "value", "works": true});
        let result = strategy
            .validate_final_result(
                Some(value.clone()),
                ValidationContext {
                    text: String::new(),
                    response: None,
                    usage: Usage::default(),
                },
            )
            .await;

        match result {
            ValidationResult::Success {
                value: result_value,
                ..
            } => {
                assert_eq!(result_value, value);
            }
            ValidationResult::Failure { error, .. } => {
                panic!("Validation failed: {:?}", error);
            }
        }
    }
}
