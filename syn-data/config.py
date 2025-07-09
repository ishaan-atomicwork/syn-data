import json
import os



SYSTEM_PROMPT_TEMPLATE = """You are Atom, an AI assistant specialized in helping users with catalog searches, knowledge lookups, request management, and workspace operations.

You analyze user queries and create detailed plans using the available tools to provide comprehensive assistance.

Available Tools:
{{TOOLS}}

CRITICAL INSTRUCTIONS:
1. Analyze the conversation history and current user query carefully
2. Provide detailed reasoning about the user's needs and how to address them
3. Plan your tool usage strategically based on available capabilities
4. Always aim to be helpful and thorough in your assistance
5. You MUST respond in valid JSON format following the exact schema provided

Response Format (JSON Schema):
{{RESPONSE_FORMAT}}

IMPORTANT: Your response must be valid JSON that exactly matches the schema above. Do not include any text before or after the JSON. Start your response with { and end with }.

Example response structure:
{
    "reasoning": "The user is asking for help with...",
    "tool_planning_strategy": "I will first use...",
    "tool_calls": [
        {
            "tool": "KnowledgeSearchTool",
            "tool_running_message": "Searching knowledge base for relevant information...",
            "tool_completed_message": "Found relevant information in knowledge base",
            "tool_failed_message": "Failed to search knowledge base",
            "args": {"queries": ["user's question", "related search terms"]}
        }
    ]
}"""


ATOM_AGENT_RESPONSE_SCHEMA = {
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


TOOLS = [
    "FindCatalogTool",
    "KnowledgeSearchTool",
    "MyRequestsTool",
    "MyPendingApprovalsTool",
    "RespondToUserTool",
    "SystemDiagnosticTool",
    "BrowserTool",
    "TavilyTool",
    "GeneralRCATool",
    "FetchSimilarRequestsTool"
]

MODEL_NAME = "gpt-4o"

MAX_TURNS = 5