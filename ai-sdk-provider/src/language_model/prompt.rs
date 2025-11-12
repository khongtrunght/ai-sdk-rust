use super::content::*;
use serde::{Deserialize, Serialize};

/// A prompt is a sequence of messages
pub type Prompt = Vec<Message>;

/// A message in the conversation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    /// System message providing instructions
    System {
        /// The system message content
        content: String,
    },
    /// User message from the human
    User {
        /// The user message content parts
        content: Vec<UserContentPart>,
    },
    /// Assistant message from the AI
    Assistant {
        /// The assistant message content parts
        content: Vec<AssistantContentPart>,
    },
    /// Tool message containing tool execution results
    Tool {
        /// The tool result content parts
        content: Vec<ToolResultPart>,
    },
}

/// Content part in a user message
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum UserContentPart {
    /// Text content part
    Text {
        /// The text content
        text: String,
    },
    /// File content part (images, audio, etc.)
    File {
        /// Binary file data
        data: Vec<u8>,
        /// MIME type of the file
        media_type: String,
    },
}

/// Content part in an assistant message
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum AssistantContentPart {
    /// Text content part
    Text(TextPart),
    /// Reasoning content part (for reasoning models like o1)
    Reasoning(ReasoningPart),
    /// File content part
    File(FilePart),
    /// Tool call content part
    ToolCall(ToolCallPart),
    /// Tool result content part
    ToolResult(ToolResultPart),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_system() {
        let msg = Message::System {
            content: "You are helpful".into(),
        };
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["role"], "system");
        assert_eq!(json["content"], "You are helpful");
    }

    #[test]
    fn test_message_user() {
        let msg = Message::User {
            content: vec![UserContentPart::Text {
                text: "Hello".into(),
            }],
        };
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["role"], "user");
        assert_eq!(json["content"][0]["type"], "text");
    }

    #[test]
    fn test_user_content_file() {
        let part = UserContentPart::File {
            data: vec![1, 2, 3],
            media_type: "image/png".into(),
        };
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(json["type"], "file");
        assert_eq!(json["media_type"], "image/png");
    }
}
