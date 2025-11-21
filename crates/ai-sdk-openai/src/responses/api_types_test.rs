#[cfg(test)]
mod tests {
    use super::super::api_types::*;
    use serde_json::json;

    #[test]
    fn test_deserialize_responses_chunk_output_text_delta() {
        let json = json!({
            "type": "response.output_text.delta",
            "item_id": "msg_123",
            "delta": "Hello",
            "logprobs": [
                {
                    "token": "Hello",
                    "logprob": -0.1,
                    "top_logprobs": []
                }
            ]
        });

        let chunk: ResponsesChunk = serde_json::from_value(json).unwrap();
        match chunk {
            ResponsesChunk::OutputTextDelta {
                item_id,
                delta,
                logprobs,
            } => {
                assert_eq!(item_id, "msg_123");
                assert_eq!(delta, "Hello");
                assert!(logprobs.is_some());
                assert_eq!(logprobs.unwrap()[0].token, "Hello");
            }
            _ => panic!("Unexpected chunk type"),
        }
    }

    #[test]
    fn test_deserialize_responses_chunk_output_item_added_web_search() {
        let json = json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "web_search_call",
                "id": "call_123",
                "status": "pending",
                "action": {
                    "type": "search",
                    "query": "rust lang"
                }
            }
        });

        let chunk: ResponsesChunk = serde_json::from_value(json).unwrap();
        match chunk {
            ResponsesChunk::OutputItemAdded { output_index, item } => {
                assert_eq!(output_index, 0);
                match item {
                    OutputItemData::WebSearchCall { id, status, action } => {
                        assert_eq!(id, "call_123");
                        assert_eq!(status.unwrap(), "pending");
                        match action.unwrap() {
                            ResponsesWebSearchAction::Search { query, .. } => {
                                assert_eq!(query.unwrap(), "rust lang");
                            }
                            _ => panic!("Unexpected action type"),
                        }
                    }
                    _ => panic!("Unexpected item type"),
                }
            }
            _ => panic!("Unexpected chunk type"),
        }
    }

    #[test]
    fn test_deserialize_responses_chunk_output_item_done_mcp_call() {
        let json = json!({
            "type": "response.output_item.done",
            "output_index": 1,
            "item": {
                "type": "mcp_call",
                "id": "mcp_123",
                "status": "completed"
            }
        });

        let chunk: ResponsesChunk = serde_json::from_value(json).unwrap();
        match chunk {
            ResponsesChunk::OutputItemDone { output_index, item } => {
                assert_eq!(output_index, 1);
                match item {
                    OutputItemData::McpCall { id, status } => {
                        assert_eq!(id, "mcp_123");
                        assert_eq!(status.unwrap(), "completed");
                    }
                    _ => panic!("Unexpected item type"),
                }
            }
            _ => panic!("Unexpected chunk type"),
        }
    }

    #[test]
    fn test_serialize_responses_request() {
        let request = ResponsesRequest {
            model: "gpt-4o".to_string(),
            input: vec![ResponsesInputItem::Message(ResponsesMessage {
                role: "user".to_string(),
                content: Some(ResponsesContent::Text("Hello".to_string())),
                id: None,
            })],
            temperature: Some(0.5),
            top_p: None,
            max_output_tokens: None,
            stream: Some(true),
            conversation: None,
            include: None,
            instructions: None,
            max_tool_calls: None,
            metadata: None,
            parallel_tool_calls: None,
            previous_response_id: None,
            prompt_cache_key: None,
            prompt_cache_retention: None,
            safety_identifier: None,
            service_tier: None,
            store: None,
            top_logprobs: None,
            truncation: None,
            user: None,
        };

        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["model"], "gpt-4o");
        assert_eq!(json["temperature"], 0.5);
        assert_eq!(json["stream"], true);
        assert_eq!(json["input"][0]["role"], "user");
        assert_eq!(json["input"][0]["content"], "Hello");
    }
}
