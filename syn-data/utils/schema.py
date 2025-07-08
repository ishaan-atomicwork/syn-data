from typing import Dict, Any

RESPONSE_SCHEMA : Dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Analyse the conversation history and the current user query to provide a detailed reasoning process."
        },
        "tool_planning_strategy": {
            "type": "string", 
            "description": "According to the reasoning process, plan the tool calls to be made. Remind the tools which are available to you and make a plan accordingly."
        },
        "tool_calls": {
            "type": "array",
            "description": "The tool calls to be made",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "description": "The name of the tool to be called"},
                    "tool_running_message": {"type": "string", "description": "The message to be displayed when the tool is running"},
                    "tool_completed_message": {"type": "string", "description": "The message to be displayed when the tool is completed"},
                    "tool_failed_message": {"type": "string", "description": "The message to be displayed when the tool fails"},
                    "args": {"type": "object", "description": "The arguments to be passed to the tool"}
                },
                "required": ["tool", "tool_running_message", "tool_completed_message", "tool_failed_message", "args"]
            }
        }
    },
    "required": ["reasoning", "tool_planning_strategy", "tool_calls"]
}