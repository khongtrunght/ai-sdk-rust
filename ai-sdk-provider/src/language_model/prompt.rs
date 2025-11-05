use super::content::*;
use serde::{Deserialize, Serialize};

pub type Prompt = Vec<Message>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    System { content: String },
    User { content: Vec<UserContentPart> },
    Assistant { content: Vec<AssistantContentPart> },
    Tool { content: Vec<ToolResultPart> },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum UserContentPart {
    Text { text: String },
    File { data: Vec<u8>, media_type: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum AssistantContentPart {
    Text(TextPart),
    Reasoning(ReasoningPart),
    File(FilePart),
    ToolCall(ToolCallPart),
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
