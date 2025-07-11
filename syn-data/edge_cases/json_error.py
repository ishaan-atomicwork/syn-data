"""
this script is for generating those cases where the LLM generates invalid JSON.

the LLM will be given a seed question and a list of error types.
the LLM will then generate a conversation that will be used to test the self-repairing ability of the LLM.

the conversation will be generated in the following format:

{
    "conversation": [
        {"role": "user", "content": "seed question"},
        {"role": "assistant", "content": "faulty json"},
        {"role": "user", "content": "validation error message"},
"""
import json
import random
import argparse
import yaml
import os
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datasets import Dataset
from bespokelabs import curator

from tool_response_formats import TOOL_RESPONSE_FORMATS, AVAILABLE_TOOLS

class ConversationTurn(BaseModel):
    role: str = Field(description="Role: user or assistant")
    content: Any = Field(description="Turn content")


class FullConversation(BaseModel):
    conversation: List[ConversationTurn] = Field(description="Complete conversation")


def load_taxonomy(file_path: str = "taxonomy.yaml") -> List[Dict[str, Any]]:
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


class InvalidJSONSelfRepairGenerator(curator.LLM):
    response_format = FullConversation

    def prompt(self, input_data: Dict) -> str:
        seed_question = input_data["seed_question"]
        
        # Determine request type based on seed question keywords
        request_type = "general"
        if any(word in seed_question.lower() for word in ["laptop", "computer", "hardware", "equipment", "docking"]):
            request_type = "hardware"
        elif any(word in seed_question.lower() for word in ["software", "license", "application", "tool"]):
            request_type = "software"
        elif any(word in seed_question.lower() for word in ["access", "permission", "login", "account", "vpn"]):
            request_type = "access"
        elif any(word in seed_question.lower() for word in ["policy", "procedure", "rule", "guideline"]):
            request_type = "policy"
        elif any(word in seed_question.lower() for word in ["incident", "problem", "issue", "error", "bug"]):
            request_type = "incident"
        elif any(word in seed_question.lower() for word in ["benefit", "hr", "payroll", "vacation", "leave", "parental"]):
            request_type = "hr"
        elif any(word in seed_question.lower() for word in ["expense", "reimburse", "cost", "budget", "travel"]):
            request_type = "expense"
        
        json_error_types = [
            "missing_required_field",
            "non_utf8_chars", 
            "trailing_commas",
            "malformed_structure"
        ]
        
        error_type = random.choice(json_error_types)
        
        error_examples = {
            "missing_required_field": {
                "faulty_json": '{"reasoning": "User needs help with their request", "tool_calls": [{"tool": "KnowledgeSearchTool", "args": {"queries": ["help request"]}}]}',
                "error_msg": "ValidationError: Missing required field 'tool_planning_strategy' in JSON response",
                "description": "Missing required field in JSON structure"
            },
            "non_utf8_chars": {
                "faulty_json": '{"reasoning": "User needs help with their request", "tool_planning_strategy": "I will search for relevant informacion", "tool_calls": [{"tool": "KnowledgeSearchTool", "tool_running_message": "Searching...", "tool_completed_message": "Done!", "tool_failed_message": "Failed", "args": {"queries": ["help request"]}}]}',
                "error_msg": "JSONDecodeError: Invalid UTF-8 character sequence at position 127",
                "description": "Non-UTF-8 characters in JSON"
            },
            "trailing_commas": {
                "faulty_json": '{"reasoning": "User needs help with their request", "tool_planning_strategy": "I will search for information", "tool_calls": [{"tool": "KnowledgeSearchTool", "tool_running_message": "Searching", "tool_completed_message": "Done", "tool_failed_message": "Failed", "args": {"queries": ["help request"]},}],}',
                "error_msg": "JSONDecodeError: Trailing comma at position 245",
                "description": "Trailing commas in JSON structure"
            },
            "malformed_structure": {
                "faulty_json": '{"reasoning": "User needs help", "tool_planning_strategy": "Search for info" "tool_calls": [{"tool": "KnowledgeSearchTool", "args": {"queries": ["help"]}}]}',
                "error_msg": "JSONDecodeError: Expected ',' delimiter at position 78",
                "description": "Malformed JSON structure with missing comma"
            }
        }
        
        selected_error = error_examples[error_type]
        
        reasoning_map = {
            "hardware": "The user is requesting hardware equipment. In a multi-turn scenario, I need to thoroughly verify current hardware policies, check available models and specifications, review approval workflows, and confirm budget allocations. I should also check for any existing hardware requests to avoid duplicates and understand replacement vs. new request scenarios. This comprehensive approach will help me handle follow-up questions about specifications, alternatives, or urgent needs.",
            "software": "This is a software license or application request. For multi-turn conversations, I need to research licensing policies, available software packages, security compliance requirements, and approval processes. I should also verify compatibility requirements and check for existing licenses or alternatives. This thorough analysis will prepare me for follow-up questions about alternatives, installation, or licensing details.",
            "access": "The user needs access permissions or account setup. In multi-turn scenarios, I need to understand the permission structure, security requirements, approval chains, and compliance protocols. I should verify current access levels and check for any existing requests or security restrictions. This comprehensive understanding will help me address follow-up questions about permissions, security, or access levels.",
            "policy": "This is a policy or procedure inquiry. For multi-turn conversations, I need to locate authoritative documentation, current procedures, and any recent updates. I should search for related forms, compliance requirements, and ensure I'm providing the most current information. This thorough approach will help me handle follow-up questions about specific scenarios or exceptions.",
            "incident": "The user is reporting an incident or technical issue. In multi-turn scenarios, I need to gather diagnostic information, review troubleshooting procedures, check for known issues, and identify escalation paths. I should also prepare support request forms and contact information. This comprehensive approach will help me handle follow-up questions about diagnostics, alternatives, or escalation.",
            "hr": "This is an HR-related request about benefits, payroll, or personnel matters. For multi-turn conversations, I need to find current HR policies, required forms, processing timelines, and appropriate contacts. I should verify eligibility requirements and approval processes. This thorough understanding will help me address follow-up questions about eligibility, timelines, or specific scenarios.",
            "expense": "The user has an expense or reimbursement request. In multi-turn scenarios, I need to review expense policies, required documentation, approval workflows, and processing timelines. I should check for specific expense categories and compliance requirements. This comprehensive analysis will help me handle follow-up questions about documentation, limits, or processing.",
            "general": "This is a general inquiry requiring comprehensive information gathering. For multi-turn conversations, I need to cast a wide net to understand the request context, find relevant policies or procedures, and identify appropriate forms or services. This broad approach will help me handle diverse follow-up questions."
        }
        
        tool_planning_map = {
            "hardware": "Strategic multi-tool approach for hardware requests using KnowledgeSearchTool, FindCatalogTool, and MyRequestsTool: I'll prioritize comprehensive hardware policy verification, then explore the catalog for available models and specifications, audit existing requests, and map approval workflows. My strategy: (1) KnowledgeSearchTool for hardware policies and approval workflows, (2) FindCatalogTool for available models and technical specifications, (3) MyRequestsTool for existing request audit to avoid duplicates, (4) KnowledgeSearchTool for budget and procurement procedures, (5) FindCatalogTool for request forms and submission processes.",
            "software": "Multi-tool strategy for software requests using KnowledgeSearchTool, FindCatalogTool, and MyRequestsTool: I'll focus on software licensing policies, security compliance, available applications, and approval processes. Approach: (1) KnowledgeSearchTool for software licensing policies and compliance, (2) FindCatalogTool for available software packages and alternatives, (3) KnowledgeSearchTool for security requirements and approval processes, (4) FindCatalogTool for installation and support procedures, (5) MyRequestsTool for compatibility verification.",
            "access": "Comprehensive access request strategy using KnowledgeSearchTool, FindCatalogTool, and MyRequestsTool: I'll examine permission structures, security protocols, current access status, and request procedures. Strategy: (1) KnowledgeSearchTool for permission documentation and security policies, (2) FindCatalogTool for access request forms and approval chains, (3) MyRequestsTool for current access audit and restrictions, (4) KnowledgeSearchTool for compliance and security verification, (5) MyRequestsTool for historical access patterns.",
            "policy": "Multi-source policy research strategy using KnowledgeSearchTool and FindCatalogTool: I'll search for authoritative policy documents, current procedures, and recent updates. Approach: (1) KnowledgeSearchTool for comprehensive policy documentation search, (2) KnowledgeSearchTool for related procedures and compliance requirements, (3) FindCatalogTool for forms and submission processes, (4) KnowledgeSearchTool for recent updates and changes, (5) KnowledgeSearchTool for cross-reference with related policies.",
            "incident": "Systematic incident response strategy using KnowledgeSearchTool, FindCatalogTool, and MyRequestsTool: I'll gather diagnostic information, troubleshooting steps, known issues, and support procedures. Strategy: (1) KnowledgeSearchTool for system diagnostics and error analysis, (2) KnowledgeSearchTool for troubleshooting procedures and solutions, (3) KnowledgeSearchTool for known issues and workarounds, (4) FindCatalogTool for support request forms and escalation paths, (5) MyRequestsTool for historical incident patterns.",
            "hr": "Comprehensive HR inquiry strategy using KnowledgeSearchTool, FindCatalogTool, and MyRequestsTool: I'll research HR policies, benefits information, processing procedures, and contacts. Approach: (1) KnowledgeSearchTool for HR policies and employee handbook, (2) KnowledgeSearchTool for benefits and compensation information, (3) FindCatalogTool for required forms and approval processes, (4) KnowledgeSearchTool for processing timelines and contacts, (5) MyRequestsTool for eligibility verification.",
            "expense": "Multi-faceted expense strategy using KnowledgeSearchTool, FindCatalogTool, and MyRequestsTool: I'll review expense policies, documentation requirements, approval workflows, and processing timelines. Strategy: (1) KnowledgeSearchTool for expense and reimbursement policies, (2) KnowledgeSearchTool for required documentation and receipts, (3) FindCatalogTool for approval workflows and limits, (4) KnowledgeSearchTool for processing timelines and payment methods, (5) MyRequestsTool for category-specific requirements.",
            "general": "Comprehensive multi-tool approach using KnowledgeSearchTool, FindCatalogTool, and MyRequestsTool: I'll use a broad multi-faceted strategy to gather relevant information. Strategy: (1) KnowledgeSearchTool for broad knowledge base search for context, (2) FindCatalogTool for catalog search for relevant forms and services, (3) KnowledgeSearchTool for policy and procedure verification, (4) FindCatalogTool for support and contact information, (5) MyRequestsTool for historical request patterns."
        }
        
        return f"""Generate a realistic JSON SELF-REPAIR conversation with EXACTLY 6 turns following this MANDATORY pattern:

SEED QUESTION: {seed_question}

EDGE CASE: Invalid JSON → Self-repair
ERROR TYPE: {selected_error['description']}

CONVERSATION FLOW:
1. User: {seed_question}
2. Assistant: FAULTY JSON (with {error_type})
3. User: JSON validation error message
4. Assistant: CORRECTED JSON (fixed version)
5. User: Tool execution results
6. Assistant: ONLY RespondToUserTool (successful completion)

TURN 2 ASSISTANT FAULTY JSON:
{selected_error['faulty_json']}

TURN 3 USER VALIDATION ERROR:
{selected_error['error_msg']}

TURN 4 ASSISTANT CORRECTED JSON FORMAT:
{{
    "reasoning": "I apologize for the JSON formatting error. Let me provide a properly structured response. {reasoning_map.get(request_type, reasoning_map['general'])}",
    "tool_planning_strategy": "Now I'll properly format my tool strategy: {tool_planning_map.get(request_type, tool_planning_map['general'])}",
    "tool_calls": [
        {{
            "tool": "KnowledgeSearchTool",
            "tool_running_message": "Searching knowledge base for relevant information...",
            "tool_completed_message": "Successfully retrieved relevant information!",
            "tool_failed_message": "Knowledge search encountered issues, but continuing with available information.",
            "args": {{"queries": ["{request_type} policy and procedures", "{request_type} requirements", "approval workflow"]}}
        }},
        {{
            "tool": "FindCatalogTool",
            "tool_running_message": "Locating relevant forms and services in catalog...",
            "tool_completed_message": "Found applicable forms and services!",
            "tool_failed_message": "Catalog search had limitations, but proceeding with found options.",
            "args": {{"queries": ["{request_type} request form", "{request_type} service catalog"]}}
        }}
    ]
}}

TURN 5 TOOL EXECUTION RESULTS:
[
    {{
        "tool_name": "KnowledgeSearchTool",
        "tool_args": {{"queries": ["policy information", "relevant procedures", "helpful guidance"]}},
        "tool_result": {{
            "results": {{
                "topic_responses": [
                    {{
                        "answer": {{
                            "summary": "Comprehensive policy information and procedures relevant to the user's request with clear steps and requirements.",
                            "title": "Relevant Policy and Procedures"
                        }},
                        "chunks": [
                            {{
                                "title": "Policy_Document.pdf",
                                "url": "https://company.com/policies/relevant-policy"
                            }}
                        ]
                    }}
                ]
            }}
        }}
    }},
    {{
        "tool_name": "FindCatalogTool",
        "tool_args": {{"queries": ["request form", "relevant service"]}},
        "tool_result": {{
            "results": [
                {{
                    "item_id": "REQ-001",
                    "workspace_id": "8",
                    "item_name": "Relevant Request Form",
                    "category": "Appropriate Category",
                    "category_type": "REQUEST",
                    "preamble": "Description of the service or form",
                    "description": "Complete description of what this form/service provides",
                    "tags": ["relevant", "tags", "here"]
                }}
            ]
        }}
    }}
]

TURN 6 ASSISTANT RESPONSE (ONLY RespondToUserTool):
{{
    "reasoning": "Now that I have the complete information from my corrected tool calls, I can provide a comprehensive response about the user's {request_type} request. The knowledge search provided policy details and requirements, while the catalog search found the appropriate forms and services.",
    "tool_planning_strategy": "I'll deliver a structured response that combines the policy information with practical next steps, including the specific form needed and clear submission guidance for this {request_type} request.",
    "tool_calls": [
        {{
            "tool": "RespondToUserTool",
            "tool_running_message": "Compiling comprehensive response with retrieved information...",
            "tool_completed_message": "Successfully provided complete solution!",
            "tool_failed_message": "Encountered issues in response compilation, but provided available guidance.",
            "args": {{
                "success": true,
                "response": "Here's how to proceed with your {request_type} request! \\n\\n**Policy Information:**\\n• {request_type.title()} requests require proper documentation and approval\\n• Processing typically takes 3-5 business days\\n• All requests must include business justification\\n\\n**Next steps:**\\n1. Complete the {request_type} request form below\\n2. Provide required documentation\\n3. Submit for manager approval\\n\\n**Ready to submit your request?**\\n\\n<form>\\n{{\\n  \\"form_type\\": \\"service_request\\",\\n  \\"form_cta\\": \\"📋 Submit {request_type.title()} Request\\",\\n  \\"item_id\\": \\"{request_type.upper()}-001\\",\\n  \\"workspace_id\\": \\"8\\"\\n}}\\n</form>\\n\\nLet me know if you need any clarification about the process!"
            }}
        }}
    ]
}}

Generate the conversation in this EXACT format:
{{
    "conversation": [
        {{"role": "user", "content": "{seed_question}"}},
        {{"role": "assistant", "content": "FAULTY_JSON_STRING"}},
        {{"role": "user", "content": "VALIDATION_ERROR_MESSAGE"}},
        {{"role": "assistant", "content": {{CORRECTED_JSON_OBJECT}}}},
        {{"role": "tool", "content": {{TOOL_EXECUTION_RESULTS_LIST}}}},
        {{"role": "assistant", "content": {{RESPOND_TO_USER_TOOL_ONLY}}}}
    ]
}}

ABSOLUTE REQUIREMENTS:
- Use seed question from taxonomy.yaml exactly as provided
- EXACTLY 6 turns total
- Turn 2: Faulty JSON as STRING (not object)
- Turn 3: Validation error message as STRING
- Turn 4: Properly corrected JSON as OBJECT
- Turn 5: Tool execution results as LIST
- Turn 6: ONLY RespondToUserTool with success: true
- Include self-repair acknowledgment in Turn 4 reasoning
- All JSON must be structurally valid after correction
- Follow the exact error type: {error_type}

Available tools: {', '.join(AVAILABLE_TOOLS)}"""

    def parse(self, input_data: Dict, response: FullConversation) -> Dict:
        return {
            "seed_question": input_data["seed_question"],
            "full_conversation": json.dumps([
                {
                    "role": turn.role,
                    "content": turn.content
                } for turn in response.conversation
            ])
        }


def generate_sample_data(output_file: str, num_conversations: int = 500, api_key: Optional[str] = None):
    
    os.environ["CURATOR_VIEWER"] = "1"
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    taxonomy = load_taxonomy()
    all_seeds = []
    for intent in taxonomy:
        for seed in intent["seed_questions"]:
            all_seeds.append(seed)
    
    random.shuffle(all_seeds)
    
    generator = InvalidJSONSelfRepairGenerator(
        model_name="o3-2025-04-16",
        backend="openai",
        backend_params={
            "max_retries": 3,
            "max_requests_per_minute": 60,
            "max_tokens_per_minute": 100000
        },
        generation_params={"max_completion_tokens": 4096}
    )
    
    dataset_items = []
    for i in range(num_conversations):
        seed_question = all_seeds[i % len(all_seeds)]
        dataset_items.append({
            "seed_question": seed_question
        })
    
    input_dataset = Dataset.from_list(dataset_items)
    
    print(f"Generating {num_conversations} Invalid JSON Self-repair conversations...")
    results = generator(input_dataset)
    
    conversations = []
    
    if hasattr(results, 'dataset') and results.dataset is not None:
        result_dataset = results.dataset
        if hasattr(result_dataset, 'to_list'):
            result_list = result_dataset.to_list()
        elif hasattr(result_dataset, '__iter__'):
            result_list = list(result_dataset)
        else:
            result_list = []
    else:
        result_list = []
    
    for result in result_list:
        if isinstance(result, dict) and "full_conversation" in result:
            try:
                conversation = json.loads(result["full_conversation"])
                if len(conversation) == 6:
                    conversations.append(conversation)
            except (json.JSONDecodeError, TypeError):
                if isinstance(result["full_conversation"], list) and len(result["full_conversation"]) == 6:
                    conversations.append(result["full_conversation"])
    
    dataset = {
        "conversations": conversations,
    }
    
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated {len(conversations)} Invalid JSON Self-repair conversations and saved to {output_file}")
    
    if conversations:
        total_turns = sum(len(conv) for conv in conversations)
        avg_turns = total_turns / len(conversations)
        print(f"Average turns per conversation: {avg_turns:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Invalid JSON Self-repair edge case conversations")
    parser.add_argument("--output", default="data/json_error/conversations.json", help="Output file")
    parser.add_argument("--num", type=int, default=2, help="Number of conversations to generate")
    parser.add_argument("--api-key", help="OpenAI API key")
    
    args = parser.parse_args()
    
    generate_sample_data(args.output, args.num, args.api_key) 