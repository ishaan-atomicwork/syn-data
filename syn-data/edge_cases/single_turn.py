import json
import random
import argparse
import yaml
import os
from datetime import datetime
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


class CategoryAGenerator(curator.LLM):
    response_format = FullConversation

    def prompt(self, input_data: Dict) -> str:
        seed_question = input_data["seed_question"]
        
        request_type = "general"
        if any(word in seed_question.lower() for word in ["laptop", "computer", "hardware", "equipment"]):
            request_type = "hardware"
        elif any(word in seed_question.lower() for word in ["software", "license", "application", "tool"]):
            request_type = "software"
        elif any(word in seed_question.lower() for word in ["access", "permission", "login", "account"]):
            request_type = "access"
        elif any(word in seed_question.lower() for word in ["policy", "procedure", "rule", "guideline"]):
            request_type = "policy"
        elif any(word in seed_question.lower() for word in ["incident", "problem", "issue", "error", "bug"]):
            request_type = "incident"
        elif any(word in seed_question.lower() for word in ["benefit", "hr", "payroll", "vacation", "leave"]):
            request_type = "hr"
        elif any(word in seed_question.lower() for word in ["expense", "reimburse", "cost", "budget"]):
            request_type = "expense"
        
        reasoning_map = {
            "hardware": "The user is requesting hardware equipment. I need to verify current hardware policies, check available models and specifications, review approval workflows, and confirm budget allocations. I should also check for any existing hardware requests to avoid duplicates and understand replacement vs. new request scenarios.",
            "software": "This is a software license or application request. I need to research licensing policies, available software packages, security compliance requirements, and approval processes. I should also verify compatibility requirements and check for existing licenses or alternatives.",
            "access": "The user needs access permissions or account setup. I need to understand the permission structure, security requirements, approval chains, and compliance protocols. I should verify current access levels and check for any existing requests or security restrictions.",
            "policy": "This is a policy or procedure inquiry. I need to locate authoritative documentation, current procedures, and any recent updates. I should search for related forms, compliance requirements, and ensure I'm providing the most current information.",
            "incident": "The user is reporting an incident or technical issue. I need to gather diagnostic information, review troubleshooting procedures, check for known issues, and identify escalation paths. I should also prepare support request forms and contact information.",
            "hr": "This is an HR-related request about benefits, payroll, or personnel matters. I need to find current HR policies, required forms, processing timelines, and appropriate contacts. I should verify eligibility requirements and approval processes.",
            "expense": "The user has an expense or reimbursement request. I need to review expense policies, required documentation, approval workflows, and processing timelines. I should check for specific expense categories and compliance requirements.",
            "general": "This is a general inquiry requiring comprehensive information gathering. I need to cast a wide net to understand the request context, find relevant policies or procedures, and identify appropriate forms or services."
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
        
        return f"""Generate a realistic HAPPY-PATH conversation with EXACTLY 4 turns following this MANDATORY pattern:

SEED QUESTION: {seed_question}

CORRECT CONVERSATION FLOW:
1. User: {seed_question}
2. Assistant: JSON with information-gathering tools (KnowledgeSearchTool, FindCatalogTool, etc.)
3. User: Tool execution results (just the list, not a question)
4. Assistant: JSON with ONLY RespondToUserTool (successful answer)

CATEGORY A REQUIREMENTS (Happy-path fulfillment):
- Straightforward queries where agent successfully locates information
- Information is found and user's request can be fulfilled
- Agent successfully guides user to solutions
- NO failures or errors - everything works smoothly
- EXACTLY 4 turns, no more, no less
- TURN 4 must contain ONLY RespondToUserTool with success: true

TURN 2 ASSISTANT FORMAT (information gathering):
{{
    "reasoning": "{reasoning_map.get(request_type, reasoning_map['general'])}",
    "tool_planning_strategy": "{tool_planning_map.get(request_type, tool_planning_map['general'])}",
    "tool_calls": [
        {{
            "tool": "KnowledgeSearchTool",
            "tool_running_message": "Searching company knowledge base for policies, procedures, and guidelines related to your request...",
            "tool_completed_message": "Successfully retrieved relevant documentation and procedures!",
            "tool_failed_message": "Knowledge search encountered issues, but continuing with available information.",
            "args": {{"queries": ["comprehensive query based on request type", "related policy or procedure", "troubleshooting or process steps"]}}
        }},
        {{
            "tool": "FindCatalogTool",
            "tool_running_message": "Locating relevant request forms, services, and available options in our catalog...",
            "tool_completed_message": "Found applicable forms and services for your request!",
            "tool_failed_message": "Catalog search had limitations, but proceeding with found options.",
            "args": {{"queries": ["request form or service type", "related catalog items"]}}
        }}
    ]
}}

TURN 3 TOOL EXECUTION RESULTS (just the list, not a question):
[
    {{
        "tool_name": "KnowledgeSearchTool",
        "tool_args": {{"queries": ["VPN setup guide", "How to set up VPN", "VPN configuration instructions"]}},
        "tool_result": {{
            "results": {{
                "topic_responses": [
                    {{
                        "answer": {{
                            "summary": "To get access to VPN or remote access for working from outside the office, please submit a request to the IT department. They will provide you with the VPN details and guide you through the installation and connection process.",
                            "title": "VPN Setup Guide"
                        }},
                        "chunks": [
                            {{
                                "title": "Basic_IT_FAQ.pdf",
                                "url": "https://company.com/vpn-setup"
                            }}
                        ]
                    }}
                ],
                "conversation_responses": [],
                "verified_answer_responses": [],
                "default_knowledge_responses": []
            }}
        }}
    }},
    {{
        "tool_name": "FindCatalogTool",
        "tool_args": {{"queries": ["Request VPN access", "VPN setup service"]}},
        "tool_result": {{
            "results": [
                {{
                    "item_id": "VPN-001",
                    "workspace_id": "8",
                    "item_name": "VPN Setup Request",
                    "category": "Security and Network Access",
                    "category_type": "REQUEST",
                    "preamble": "Need VPN access? Submit a request for secure remote access setup.",
                    "description": "Enable secure remote access through VPN setup, ensuring safe and private connections for your work",
                    "llm_description": "The VPN setup service provides secure remote access, ensuring safe and private connections for work from any location. This service supports various devices and offers reliable deployment of secure connections.",
                    "tags": ["remote access", "network security", "privacy", "secure connection"]
                }}
            ]
        }}
    }}
]

TURN 4 ASSISTANT FORMAT (ONLY RespondToUserTool):
{{
    "reasoning": "Analyzing the comprehensive tool results: The knowledge base search revealed specific policies, procedures, and requirements relevant to the user's request. The catalog search identified available forms, services, and processing workflows. Cross-referencing this information, I can now provide a complete solution that addresses the user's immediate need while ensuring compliance with company policies. I've identified the specific steps required, any prerequisites or approvals needed, estimated timelines, and potential alternatives or escalation paths.",
    "tool_planning_strategy": "With all necessary information gathered, I'll deliver a structured response that: (1) Directly addresses the user's question with actionable steps, (2) Provides relevant forms or services with clear submission instructions, (3) Sets appropriate expectations for timelines and approval processes, (4) Includes helpful context about policies or procedures, (5) Offers proactive guidance for potential follow-up questions or next steps.",
    "tool_calls": [
        {{
            "tool": "RespondToUserTool",
            "tool_running_message": "Synthesizing information and preparing comprehensive solution tailored to your specific request...",
            "tool_completed_message": "Complete solution with actionable steps and relevant resources provided successfully!",
            "tool_failed_message": "Encountered difficulties in delivering complete solution, but provided available guidance.",
            "args": {{
                "success": true,
                "response": "Here's how to get VPN access! 🌐\\n\\n**How to get VPN access:**\\n- Submit a request to the IT department for VPN access\\n- They'll provide you with the details and guide you through installation\\n- You can download the VPN client from the IT portal and log in using your credentials\\n\\n**Ready to request VPN access?**\\n\\n<form>\\n{{\\n  \\"form_type\\": \\"service_request\\",\\n  \\"form_cta\\": \\"🔒 Request VPN Setup\\",\\n  \\"item_id\\": \\"VPN-001\\",\\n  \\"workspace_id\\": \\"8\\"\\n}}\\n</form>\\n\\nThis will get you set up with secure remote access for working from anywhere! 😊"
            }}
        }}
    ]
}}

Generate the conversation in this EXACT format:
{{
    "conversation": [
        {{"role": "user", "content": "{seed_question}"}},
        {{"role": "assistant", "content": {{INFORMATION_GATHERING_TOOLS}}}},
        {{"role": "user", "content": {{TOOL_EXECUTION_RESULTS_LIST}}}},
        {{"role": "assistant", "content": {{RESPOND_TO_USER_TOOL_ONLY}}}}
    ]
}}

ABSOLUTE REQUIREMENTS:
- Use seed question from taxonomy.yaml exactly as provided
- 4 turns total
- Turn 3 is just tool execution results (not a question)
- Turn 4 must contain ONLY RespondToUserTool with success: true
- Include detailed, realistic tool parameters and comprehensive responses
- ALL scenarios must be successful (happy path)
- Follow the exact flow shown in the example

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


def generate_sample_data(output_file: str, num_conversations: int = 2000, api_key: Optional[str] = None):
    
    os.environ["CURATOR_VIEWER"] = "1"
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    taxonomy = load_taxonomy()
    all_seeds = []
    for intent in taxonomy:
        for seed in intent["seed_questions"]:
            all_seeds.append(seed)
    
    random.shuffle(all_seeds)
    
    generator = CategoryAGenerator(
        model_name="o3-2025-04-16",
        backend="openai",
        backend_params={
            "max_retries": 3,
            "max_requests_per_minute": 60,
            "max_tokens_per_minute": 100000
        },
        generation_params={"temperature": 0.9, "max_completion_tokens": 4096}
    )
    
    dataset_items = []
    for i in range(num_conversations):
        seed_question = all_seeds[i % len(all_seeds)]
        dataset_items.append({
            "seed_question": seed_question
        })
    
    input_dataset = Dataset.from_list(dataset_items)
    
    print(f"Generating {num_conversations} Category A (Happy-path) conversations...")
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
                if len(conversation) >= 4:
                    conversations.append(conversation)
            except (json.JSONDecodeError, TypeError):
                if isinstance(result["full_conversation"], list) and len(result["full_conversation"]) >= 4:
                    conversations.append(result["full_conversation"])
    
    dataset = {
        "conversations": conversations,
    }
    
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated {len(conversations)} Category A conversations and saved to {output_file}")
    
    if conversations:
        total_turns = sum(len(conv) for conv in conversations)
        avg_turns = total_turns / len(conversations)
        print(f"Average turns per conversation: {avg_turns:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Category A: Happy-path fulfillment conversations")
    parser.add_argument("--output", default="data/single_turn/conversations.json", help="Output file")
    parser.add_argument("--num", type=int, default=2, help="Number of conversations to generate")
    parser.add_argument("--api-key", help="OpenAI API key")
    
    args = parser.parse_args()
    
    generate_sample_data(args.output, args.num, args.api_key) 