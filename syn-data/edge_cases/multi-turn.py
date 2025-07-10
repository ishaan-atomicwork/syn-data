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


class CategoryBGenerator(curator.LLM):
    response_format = FullConversation

    def prompt(self, input_data: Dict) -> str:
        seed_question = input_data["seed_question"]
        
        # Determine request type based on seed question keywords
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
        
        # Dynamic reasoning based on request type
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
        
        return f"""Generate a realistic MULTI-TOOL ORCHESTRATION conversation with VARIABLE turns (8-10 turns) based on multiple user queries.

SEED QUESTION: {seed_question}

CORRECT CONVERSATION FLOW EXAMPLE:
1. User: "Help me with VPN setup"
2. Assistant: JSON with KnowledgeSearchTool + FindCatalogTool (information gathering)
3. User: [tool execution results list] (just the results, not a question)
4. Assistant: JSON with ONLY RespondToUserTool (actual answer to user)
5. User: "What if I'm working from China?" (follow-up question)
6. Assistant: JSON with ONLY RespondToUserTool (answer to follow-up)

PATTERN A (8 turns - 2 User Questions):
TURN 1 - USER: {seed_question}
TURN 2 - ASSISTANT: Use information-gathering tools (KnowledgeSearchTool, FindCatalogTool, MyRequestsTool, etc.)
TURN 3 - USER: Tool execution results (just the list, not a question)
TURN 4 - ASSISTANT: Use ONLY RespondToUserTool to answer the original question
TURN 5 - USER: Realistic follow-up question #1
TURN 6 - ASSISTANT: Use information-gathering tools if needed OR use ONLY RespondToUserTool
TURN 7 - USER: Tool execution results (if Turn 6 had info gathering) OR second follow-up question
TURN 8 - ASSISTANT: Use ONLY RespondToUserTool to answer the final question

PATTERN B (10 turns - 3 User Questions):
TURN 1 - USER: {seed_question}
TURN 2 - ASSISTANT: Use information-gathering tools (KnowledgeSearchTool, FindCatalogTool, MyRequestsTool, etc.)
TURN 3 - USER: Tool execution results (just the list, not a question)
TURN 4 - ASSISTANT: Use ONLY RespondToUserTool to answer the original question
TURN 5 - USER: Realistic follow-up question #1
TURN 6 - ASSISTANT: Use information-gathering tools if needed OR use ONLY RespondToUserTool
TURN 7 - USER: Tool execution results (if Turn 6 had info gathering) OR second follow-up question
TURN 8 - ASSISTANT: Use ONLY RespondToUserTool to answer the follow-up question
TURN 9 - USER: Realistic follow-up question #2
TURN 10 - ASSISTANT: Use ONLY RespondToUserTool to answer the final question

ASSISTANT INFORMATION-GATHERING FORMAT:
{{
    "reasoning": "{reasoning_map.get(request_type, reasoning_map['general'])}",
    "tool_planning_strategy": "{tool_planning_map.get(request_type, tool_planning_map['general'])}",
    "tool_calls": [
        {{
            "tool": "KnowledgeSearchTool",
            "tool_running_message": "Performing comprehensive search across company knowledge base for policies, procedures, and technical documentation relevant to your request...",
            "tool_completed_message": "Successfully retrieved detailed documentation, policies, and procedural guidance!",
            "tool_failed_message": "Knowledge search faced some limitations, but gathered available information to proceed.",
            "args": {{"queries": ["comprehensive query tailored to request type", "related policies and procedures", "troubleshooting steps or compliance requirements"]}}
        }},
        {{
            "tool": "FindCatalogTool",
            "tool_running_message": "Scanning service catalog for relevant request forms, available services, and processing options specific to your needs...",
            "tool_completed_message": "Located applicable forms, services, and request options with detailed specifications!",
            "tool_failed_message": "Catalog search encountered limitations, but found relevant options to proceed.",
            "args": {{"queries": ["specific request form or service type", "related catalog offerings and alternatives"]}}
        }},
        {{
            "tool": "MyRequestsTool",
            "tool_running_message": "Reviewing your request history to identify patterns, previous submissions, and current status of related requests...",
            "tool_completed_message": "Completed comprehensive review of your request history and current submissions!",
            "tool_failed_message": "Request history review had technical issues, but proceeding with available data.",
            "args": {{"status": ["pending", "approved", "in_progress"]}}
        }}
    ]
}}

TOOL EXECUTION RESULTS (just the list, not a question):
[
    {{
        "tool_name": "KnowledgeSearchTool",
        "tool_args": {{"queries": ["laptop replacement policy", "hardware request process", "IT equipment approval"]}},
        "tool_result": {{
            "results": {{
                "topic_responses": [
                    {{
                        "answer": {{
                            "summary": "All hardware requests must be submitted through the IT Service Portal with manager approval. Processing takes 3-5 business days. Standard laptop models include Dell Latitude 7420, MacBook Pro 13-inch, and ThinkPad X1 Carbon.",
                            "title": "Hardware Request Process"
                        }},
                        "chunks": [
                            {{
                                "title": "IT_Hardware_Policy.pdf",
                                "url": "https://company.com/policies/hardware"
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
        "tool_args": {{"queries": ["laptop request", "hardware request form"]}},
        "tool_result": {{
            "results": [
                {{
                    "item_id": "HW-001",
                    "workspace_id": "8",
                    "item_name": "Hardware Request Form",
                    "category": "IT Equipment",
                    "category_type": "REQUEST",
                    "preamble": "Need new hardware? Submit a request through our hardware request form.",
                    "description": "Request laptops, monitors, keyboards, and other IT equipment with manager approval",
                    "llm_description": "The Hardware Request Form allows employees to request IT equipment including laptops, monitors, keyboards, and accessories. Requires manager approval and business justification. Processing time is 3-5 business days.",
                    "tags": ["hardware", "laptop", "equipment", "IT"]
                }}
            ]
        }}
    }},
    {{
        "tool_name": "MyRequestsTool",
        "tool_args": {{"status": ["pending", "approved", "in_progress"]}},
        "tool_result": {{
            "success": true,
            "response": {{
                "page": 1,
                "page_size": 10,
                "total_pages": 1,
                "total_count": 0,
                "data": []
            }}
        }}
    }}
]

ASSISTANT RESPONSE FORMAT (answering user with ONLY RespondToUserTool):
{{
    "reasoning": "Synthesizing comprehensive tool results from multi-source analysis: The knowledge base provided detailed policies, procedures, and compliance requirements. The catalog search revealed available services, forms, and processing workflows with specific timelines. The request history audit showed relevant patterns and current status. Cross-referencing all sources, I can now deliver a complete solution that not only addresses the immediate request but also anticipates potential follow-up needs, ensures policy compliance, and provides proactive guidance for the user's specific situation.",
    "tool_planning_strategy": "Delivering a structured, multi-layered response that: (1) Immediately addresses the user's primary question with clear, actionable steps, (2) Integrates relevant policy context and compliance requirements, (3) Provides specific forms/services with detailed submission guidance, (4) Sets realistic expectations for timelines and approval processes, (5) Includes relevant alternatives or escalation paths, (6) Anticipates logical follow-up questions and provides proactive guidance, (7) Ensures the response enables user self-service while maintaining support availability.",
    "tool_calls": [
        {{
            "tool": "RespondToUserTool",
            "tool_running_message": "Integrating multi-source analysis and crafting comprehensive, actionable response tailored to your specific needs and situation...",
            "tool_completed_message": "Delivered complete solution with step-by-step guidance, relevant resources, and proactive support information!",
            "tool_failed_message": "Encountered challenges in response delivery, but provided comprehensive guidance based on available information.",
            "args": {{
                "success": true,
                "response": "Here's how to get your laptop! 💻\\n\\n**Process:**\\n1. Submit request through IT Service Portal\\n2. Get manager approval\\n3. Wait 3-5 business days for processing\\n\\n**Available Models:**\\n- Dell Latitude 7420\\n- MacBook Pro 13-inch\\n- ThinkPad X1 Carbon\\n\\n**Ready to request?**\\n\\n<form>\\n{{\\n  \\"form_type\\": \\"service_request\\",\\n  \\"form_cta\\": \\"🖥️ Request Hardware\\",\\n  \\"item_id\\": \\"HW-001\\",\\n  \\"workspace_id\\": \\"8\\"\\n}}\\n</form>\\n\\nNeed help with anything specific about the request process?"
            }}
        }}
    ]
}}

REALISTIC FOLLOW-UP QUESTIONS:
- "What if I need it urgently?"
- "Can I get a higher-spec model?"
- "How long does manager approval take?"
- "What happens to my old laptop?"
- "Are there any accessories included?"

Generate the conversation using ONE of these EXACT formats:

PATTERN A (8 turns - 2 User Questions):
{{
    "conversation": [
        {{"role": "user", "content": "{seed_question}"}},
        {{"role": "assistant", "content": {{INFORMATION_GATHERING_TOOLS}}}},
        {{"role": "user", "content": {{TOOL_EXECUTION_RESULTS_LIST}}}},
        {{"role": "assistant", "content": {{RESPOND_TO_USER_TOOL_ONLY}}}},
        {{"role": "user", "content": "{{REALISTIC_FOLLOW_UP_QUESTION_1}}"}},
        {{"role": "assistant", "content": {{RESPOND_TO_USER_TOOL_ONLY_OR_INFO_GATHERING}}}},
        {{"role": "user", "content": "{{TOOL_RESULTS_OR_SECOND_FOLLOW_UP_QUESTION}}"}},
        {{"role": "assistant", "content": {{RESPOND_TO_USER_TOOL_ONLY}}}}
    ]
}}

PATTERN B (10 turns - 3 User Questions):
{{
    "conversation": [
        {{"role": "user", "content": "{seed_question}"}},
        {{"role": "assistant", "content": {{INFORMATION_GATHERING_TOOLS}}}},
        {{"role": "user", "content": {{TOOL_EXECUTION_RESULTS_LIST}}}},
        {{"role": "assistant", "content": {{RESPOND_TO_USER_TOOL_ONLY}}}},
        {{"role": "user", "content": "{{REALISTIC_FOLLOW_UP_QUESTION_1}}"}},
        {{"role": "assistant", "content": {{RESPOND_TO_USER_TOOL_ONLY_OR_INFO_GATHERING}}}},
        {{"role": "user", "content": "{{TOOL_RESULTS_OR_SECOND_FOLLOW_UP_QUESTION}}"}},
        {{"role": "assistant", "content": {{RESPOND_TO_USER_TOOL_ONLY}}}},
        {{"role": "user", "content": "{{REALISTIC_FOLLOW_UP_QUESTION_2}}"}},
        {{"role": "assistant", "content": {{RESPOND_TO_USER_TOOL_ONLY}}}}
    ]
}}

ABSOLUTE REQUIREMENTS:
- Use seed question from taxonomy.yaml exactly as provided
- Generate 2-3 realistic user questions (not tool execution results)
- Tool execution results are just the list, not questions
- When answering user: Use ONLY RespondToUserTool
- When gathering info: Use KnowledgeSearchTool, FindCatalogTool, MyRequestsTool, etc.
- Include detailed, realistic tool parameters and comprehensive responses
- Variable turns: 8 turns (Pattern A) OR 10 turns (Pattern B)
- ALWAYS ends with RespondToUserTool with success: true
- 50% distribution between Pattern A and Pattern B
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


def generate_sample_data(output_file: str, num_conversations: int = 1200, api_key: Optional[str] = None):
    
    os.environ["CURATOR_VIEWER"] = "1"
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    taxonomy = load_taxonomy()
    all_seeds = []
    for intent in taxonomy:
        for seed in intent["seed_questions"]:
            all_seeds.append(seed)
    
    random.shuffle(all_seeds)
    
    generator = CategoryBGenerator(
        model_name="o3-2025-04-16",
        backend="openai",
        backend_params={
            "max_retries": 3,
            "max_requests_per_minute": 60,
            "max_tokens_per_minute": 100000
        },
        generation_params={"max_completion_tokens": 8192}
    )
    
    dataset_items = []
    for i in range(num_conversations):
        seed_question = all_seeds[i % len(all_seeds)]
        dataset_items.append({
            "seed_question": seed_question
        })
    
    input_dataset = Dataset.from_list(dataset_items)
    
    print(f"Generating {num_conversations} Category B (Multi-tool orchestration) conversations...")
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
                if len(conversation) >= 8:  # Accept 8 or 10 turns
                    conversations.append(conversation)
            except (json.JSONDecodeError, TypeError):
                if isinstance(result["full_conversation"], list) and len(result["full_conversation"]) >= 8:
                    conversations.append(result["full_conversation"])
    
    dataset = {
        "conversations": conversations,
    }
    
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated {len(conversations)} Category B conversations and saved to {output_file}")
    
    if conversations:
        total_turns = sum(len(conv) for conv in conversations)
        avg_turns = total_turns / len(conversations)
        print(f"Average turns per conversation: {avg_turns:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Category B: Multi-tool orchestration conversations")
    parser.add_argument("--output", default="data/multi_turn/conversations.json", help="Output file")
    parser.add_argument("--num", type=int, default=2, help="Number of conversations to generate")
    parser.add_argument("--api-key", help="OpenAI API key")
    
    args = parser.parse_args()
    
    generate_sample_data(args.output, args.num, args.api_key) 