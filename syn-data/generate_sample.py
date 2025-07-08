import json
import random
import logging
from typing import List, Dict, Any
import yaml
from pydantic import BaseModel, Field
from datasets import Dataset

from bespokelabs import curator
from config import TOOLS, MAX_TURNS
from utils.schema import RESPONSE_SCHEMA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToolCall(BaseModel):
    tool: str = Field(description="Tool name")
    tool_running_message: str = Field(description="Message shown while tool is running")
    tool_completed_message: str = Field(description="Message shown when tool completes")
    tool_failed_message: str = Field(description="Message shown when tool fails")
    args: Dict[str, Any] = Field(description="Tool arguments")

class AssistantResponse(BaseModel):
    reasoning: str = Field(description="Assistant's reasoning process")
    tool_planning_strategy: str = Field(description="Strategy for using tools")
    tool_calls: List[ToolCall] = Field(description="List of tool calls to make")

class ConversationTurn(BaseModel):
    role: str = Field(description="Role: user or assistant")
    content: Any = Field(description="Turn content")

class MultiTurnConversation(BaseModel):
    conversation: List[ConversationTurn] = Field(description="Complete conversation")

def load_taxonomy() -> List[Dict[str, Any]]:
    with open('syn-data/taxonomy.yaml', 'r') as f:
        return yaml.safe_load(f)

def get_failure_scenarios() -> List[str]:
    return [
        "That didn't work, no results found.",
        "The search returned empty results.",
        "I couldn't find what I was looking for.",
        "That form doesn't seem to exist.",
        "The catalog search failed.",
        "No relevant articles found.",
        "That service isn't available.",
        "I'm getting an error with that request.",
        "The tool seems to be down.",
        "Those results aren't helpful."
    ]

def get_success_scenarios() -> List[str]:
    return [
        "Perfect! That's exactly what I needed.",
        "Great, I found the right form.",
        "Thanks! The search results were helpful.",
        "That worked perfectly.",
        "Excellent! I can proceed now.",
        "That's the information I was looking for.",
        "Perfect! I'll use that service.",
        "Great! I can see the relevant articles.",
        "Thanks for finding that for me."
    ]

def get_follow_up_prompts() -> List[str]:
    return [
        "Can you help me with something else?",
        "What about alternatives?",
        "Is there anything else I should know?",
        "What are the next steps?",
        "Can you provide more details?",
        "Are there other options?",
        "What if that doesn't work?",
        "I need help with something related.",
        "Can you explain the process?",
        "What documentation should I check?"
    ]

class ConversationGenerator(curator.LLM):
    response_format = AssistantResponse

    def prompt(self, input_data: Dict) -> str:
        seed_question = input_data["seed_question"]
        conversation_history = input_data.get("conversation_history", [])
        
        system_prompt = f"""You are an AI assistant that must respond in JSON format with the following structure:
{{
    "reasoning": "Your reasoning process",
    "tool_planning_strategy": "Your strategy for using tools",
    "tool_calls": [
        {{
            "tool": "ToolName",
            "tool_running_message": "Message while running",
            "tool_completed_message": "Message when complete",
            "tool_failed_message": "Message when failed",
            "args": {{"key": "value"}}
        }}
    ]
}}

Available tools: {', '.join(TOOLS)}

You must always use appropriate tools to help the user. Be helpful and provide structured responses.

Conversation so far:"""
        
        for turn in conversation_history:
            if turn["role"] == "user":
                system_prompt += f"\nUser: {turn['content']}"
            else:
                system_prompt += f"\nAssistant: {json.dumps(turn['content'])}"
        
        system_prompt += f"\nUser: {seed_question}\n\nRespond in JSON format:"
        return system_prompt

    def parse(self, input_data: Dict, response: AssistantResponse) -> List[Dict]:
        return [{
            "seed_question": input_data["seed_question"],
            "conversation_history": input_data.get("conversation_history", []),
            "assistant_response": {
                "reasoning": response.reasoning,
                "tool_planning_strategy": response.tool_planning_strategy,
                "tool_calls": [
                    {
                        "tool": tc.tool,
                        "tool_running_message": tc.tool_running_message,
                        "tool_completed_message": tc.tool_completed_message,
                        "tool_failed_message": tc.tool_failed_message,
                        "args": tc.args
                    } for tc in response.tool_calls
                ]
            }
        }]

class MultiTurnGenerator(curator.LLM):
    response_format = MultiTurnConversation

    def prompt(self, input_data: Dict) -> str:
        seed_question = input_data["seed_question"]
        
        return f"""Generate a realistic multi-turn conversation between a user and an AI assistant. The conversation should:

1. Start with this user question: "{seed_question}"
2. Have 3-6 turns total
3. Include realistic user follow-ups, including some tool failures (15% chance)
4. Show the assistant using appropriate tools from: {', '.join(TOOLS)}
5. Include varied user responses like successes, failures, and follow-up questions

IMPORTANT: The assistant can use multiple tools in a single turn, then respond to the user after using all tools.

User response patterns to include:
- Success responses: {random.sample(get_success_scenarios(), 3)}
- Failure responses: {random.sample(get_failure_scenarios(), 2)}
- Follow-ups: {random.sample(get_follow_up_prompts(), 3)}

Available Tools:
- FindCatalogTool: Search service & incident catalog for forms (requires 'queries' array of 2-4 search phrases)
- KnowledgeSearchTool: Search knowledge base for information (requires 'queries' array of 2-4 search phrases)  
- MyRequestsTool: Get user's requests/tickets (requires 'status' array, optional 'priority' array)
- MyPendingApprovalsTool: Get pending approvals for user (requires 'status' array, optional 'priority' array)
- RespondToUserTool: Reply to user (requires 'success' boolean and 'response' string in Markdown)

The assistant must ALWAYS respond in this EXACT JSON structure:
{{
    "reasoning": "detailed reasoning process",
    "tool_planning_strategy": "strategy for using tools",
    "tool_calls": [
        {{
            "tool": "FindCatalogTool",
            "tool_running_message": "Searching catalog for relevant forms...",
            "tool_completed_message": "Found matching forms in catalog!",
            "tool_failed_message": "No matching forms found in catalog.",
            "args": {{"queries": ["laptop request form", "hardware request", "equipment order", "new laptop service"]}}
        }},
        {{
            "tool": "KnowledgeSearchTool", 
            "tool_running_message": "Searching knowledge base...",
            "tool_completed_message": "Retrieved relevant information!",
            "tool_failed_message": "No relevant information found.",
            "args": {{"queries": ["parental leave policy", "family leave benefits", "maternity leave", "paternity leave"]}}
        }},
        {{
            "tool": "RespondToUserTool",
            "tool_running_message": "Preparing response...",
            "tool_completed_message": "Response sent to user.",
            "tool_failed_message": "Failed to send response.",
            "args": {{"success": true, "response": "I found the laptop request form for you. According to the policy, new laptop requests are processed within 3-5 business days. Would you like me to help you with anything else?"}}
        }}
    ]
}}

Generate a complete conversation in this format:
{{
    "conversation": [
        {{"role": "user", "content": "user message"}},
        {{"role": "assistant", "content": {{assistant_json_response}}}},
        ...
    ]
}}"""

    def parse(self, input_data: Dict, response: MultiTurnConversation) -> List[Dict]:
        return [{
            "seed_question": input_data["seed_question"],
            "full_conversation": json.dumps([
                {
                    "role": turn.role,
                    "content": turn.content
                } for turn in response.conversation
            ])
        }]

def main():
    import os 
    os.environ["CURATOR_VIEWER"]="1"
    os.environ["OPENAI_API_KEY"] = ""
    
    taxonomy = load_taxonomy()
    all_seeds = []
    for intent in taxonomy:
        for seed in intent["seed_questions"]:
            all_seeds.append(seed)
    
    random.shuffle(all_seeds)
    
    generator = MultiTurnGenerator(
        model_name="gpt-4.1-nano-2025-04-14",
        backend="openai",
        backend_params={
            "max_retries": 3,
            "max_requests_per_minute": 60,
            "max_tokens_per_minute": 100000
        },
        generation_params={"temperature": 0.9, "max_tokens": 2048}
    )
    
    input_data = [{"seed_question": seed} for seed in all_seeds[:10]]
    input_dataset = Dataset.from_list(input_data)
    
    logger.info(f"Generating conversations for {len(input_data)} seed questions...")
    
    results = generator(input_dataset)
    
    conversations = []
    
    try:
        if hasattr(results, 'dataset') and results.dataset is not None:
            dataset = results.dataset
            if hasattr(dataset, 'to_list'):
                result_list = dataset.to_list()
            elif hasattr(dataset, '__iter__'):
                result_list = list(dataset)
            else:
                logger.warning("Dataset doesn't have expected methods")
                result_list = []
        else:
            logger.warning("Results doesn't have dataset attribute")
            result_list = []
        
        for result in result_list:
            if isinstance(result, dict) and "full_conversation" in result:
                try:
                    # Parse the JSON string back to a list
                    conversation = json.loads(result["full_conversation"])
                    if len(conversation) >= 3:
                        conversations.append(conversation)
                except (json.JSONDecodeError, TypeError):
                    # If it's already a list, use it directly
                    if isinstance(result["full_conversation"], list) and len(result["full_conversation"]) >= 3:
                        conversations.append(result["full_conversation"])
                
    except Exception as e:
        logger.error(f"Error processing results: {e}")
        logger.error(f"Results type: {type(results)}")
        if hasattr(results, '__dict__'):
            logger.error(f"Results attributes: {list(results.__dict__.keys())}")
    
    # Save to JSON
    with open('data/sample.json', 'w') as f:
        json.dump(conversations, f, indent=2)
    
    logger.info(f"Generated {len(conversations)} conversations and saved to sample.json")
    
    print(f"\n{'='*50}")
    print(f"SAMPLE GENERATION COMPLETE")
    print(f"{'='*50}")
    print(f"Generated: {len(conversations)} conversations")
    print(f"Saved to: sample.json")

if __name__ == "__main__":
    main()