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


class AmbiguityClarificationGenerator(curator.LLM):
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
        
        ambiguity_types = [
            "multiple_forms_available",
            "unclear_department",
            "vague_request_type",
            "multiple_interpretations"
        ]
        
        ambiguity_type = random.choice(ambiguity_types)
        
        clarification_patterns = {
            "multiple_forms_available": {
                "description": "Multiple relevant forms found in catalog",
                "clarification_question": "I found several relevant forms that might address your request. Could you clarify which specific type of [request/service] you need?",
                "options": ["Option A: [Specific form type]", "Option B: [Alternative form type]", "Option C: [Third form type]"]
            },
            "unclear_department": {
                "description": "Request could apply to multiple departments",
                "clarification_question": "Your request could be handled by different departments. Could you specify which area this relates to?",
                "options": ["Option A: [Department/Area 1]", "Option B: [Department/Area 2]", "Option C: [Department/Area 3]"]
            },
            "vague_request_type": {
                "description": "Request type is too general",
                "clarification_question": "I'd like to help you with the most relevant information. Could you be more specific about what type of [service/support] you're looking for?",
                "options": ["Option A: [Specific service type]", "Option B: [Alternative service type]", "Option C: [Third service type]"]
            },
            "multiple_interpretations": {
                "description": "Question has multiple possible meanings",
                "clarification_question": "Your question could have a few different interpretations. Could you clarify which specific aspect you're asking about?",
                "options": ["Option A: [First interpretation]", "Option B: [Second interpretation]", "Option C: [Third interpretation]"]
            }
        }
        
        selected_pattern = clarification_patterns[ambiguity_type]
        
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
        
        return f"""Generate a realistic AMBIGUITY & CLARIFICATION conversation with EXACTLY 10 turns following this MANDATORY pattern:

SEED QUESTION: {seed_question}

EDGE CASE: Ambiguity & clarification
AMBIGUITY TYPE: {selected_pattern['description']}

CONVERSATION FLOW:
1. User: {seed_question} (vague/ambiguous initial request)
2. Assistant: Initial information gathering tools
3. User: Tool execution results (showing multiple options/ambiguous results)
4. Assistant: ONLY RespondToUserTool (asks for clarification due to ambiguity)
5. User: Clarification response (clearer, more specific request)
6. Assistant: Targeted information gathering based on clarification
7. User: Tool execution results (more specific)
8. Assistant: ONLY RespondToUserTool (comprehensive answer)
9. User: Follow-up question
10. Assistant: ONLY RespondToUserTool (final response)

TURN 2 ASSISTANT INITIAL TOOL CALLS:
{{
    "reasoning": "{reasoning_map.get(request_type, reasoning_map['general'])}",
    "tool_planning_strategy": "{tool_planning_map.get(request_type, tool_planning_map['general'])}",
    "tool_calls": [
        {{
            "tool": "KnowledgeSearchTool",
            "tool_running_message": "Searching knowledge base for comprehensive information about your request...",
            "tool_completed_message": "Successfully retrieved relevant information from knowledge base!",
            "tool_failed_message": "Knowledge search encountered some issues, but continuing with available information.",
            "args": {{"queries": ["{request_type} policies and procedures", "{request_type} requirements", "general {request_type} information"]}}
        }},
        {{
            "tool": "FindCatalogTool",
            "tool_running_message": "Searching catalog for relevant forms and services...",
            "tool_completed_message": "Found multiple relevant options in catalog!",
            "tool_failed_message": "Catalog search had some limitations, but proceeding with found options.",
            "args": {{"queries": ["{request_type} forms and services", "{request_type} request options"]}}
        }}
    ]
}}

TURN 3 TOOL EXECUTION RESULTS (showing ambiguity):
[
    {{
        "tool_name": "KnowledgeSearchTool",
        "tool_args": {{"queries": ["broad search terms related to request", "general policy information", "relevant procedures"]}},
        "tool_result": {{
            "results": {{
                "topic_responses": [
                    {{
                        "answer": {{
                            "summary": "Multiple policies and procedures found that could apply to your request, depending on the specific type and department involved.",
                            "title": "Multiple Relevant Policies Found"
                        }},
                        "chunks": [
                            {{
                                "title": "General_Policy_A.pdf",
                                "url": "https://company.com/policies/policy-a"
                            }},
                            {{
                                "title": "Alternative_Policy_B.pdf", 
                                "url": "https://company.com/policies/policy-b"
                            }},
                            {{
                                "title": "Specialized_Policy_C.pdf",
                                "url": "https://company.com/policies/policy-c"
                            }}
                        ]
                    }}
                ]
            }}
        }}
    }},
    {{
        "tool_name": "FindCatalogTool",
        "tool_args": {{"queries": ["general service search", "related forms", "relevant catalog items"]}},
        "tool_result": {{
            "results": [
                {{
                    "item_id": "FORM-A-001",
                    "workspace_id": "8",
                    "item_name": "General Request Form Type A",
                    "category": "General Services",
                    "category_type": "REQUEST",
                    "preamble": "Submit requests for Type A services",
                    "description": "Form for requesting Type A services and support",
                    "tags": ["type-a", "general", "request"]
                }},
                {{
                    "item_id": "FORM-B-002",
                    "workspace_id": "8",
                    "item_name": "Alternative Request Form Type B",
                    "category": "Alternative Services",
                    "category_type": "REQUEST",
                    "preamble": "Submit requests for Type B services",
                    "description": "Form for requesting Type B services and support",
                    "tags": ["type-b", "alternative", "request"]
                }},
                {{
                    "item_id": "FORM-C-003",
                    "workspace_id": "8",
                    "item_name": "Specialized Request Form Type C",
                    "category": "Specialized Services",
                    "category_type": "REQUEST",
                    "preamble": "Submit requests for Type C services",
                    "description": "Form for requesting Type C specialized services",
                    "tags": ["type-c", "specialized", "request"]
                }}
            ]
        }}
    }}
]

TURN 4 ASSISTANT CLARIFICATION REQUEST:
{{
    "reasoning": "The search results show multiple relevant policies and forms that could address the user's request. To provide the most accurate and helpful assistance, I need to clarify which specific type of service or support they need.",
    "tool_planning_strategy": "I'll present the user with the different options I found and ask for clarification so I can provide the most relevant information and appropriate form.",
    "tool_calls": [
        {{
            "tool": "RespondToUserTool",
            "tool_running_message": "Preparing clarification request with available options...",
            "tool_completed_message": "Clarification request prepared with multiple options!",
            "tool_failed_message": "Had difficulty preparing clarification, but provided available options.",
            "args": {{
                "success": true,
                "response": "I found multiple {request_type}-related options that could address your request. To provide you with the most accurate information, could you clarify which specific type of {request_type} assistance you need?\\n\\n**Available {request_type} options:**\\n• Option A: [Specific {request_type} service type]\\n• Option B: [Alternative {request_type} service type]\\n• Option C: [Third {request_type} service type]\\n\\nOnce you clarify which option best matches your {request_type} needs, I can provide you with the specific policies, procedures, and forms that apply to your situation. Which {request_type} option would you like me to focus on?"
            }}
        }}
    ]
}}

TURN 5 USER CLARIFICATION RESPONSE:
"I need [specific {request_type} clarification that matches Option A from the choices]"

TURN 6 ASSISTANT TARGETED INFORMATION GATHERING:
{{
    "reasoning": "Now that the user has clarified their specific {request_type} needs, I can conduct a more targeted search to provide precise {request_type} information, policies, and forms that directly address their request.",
    "tool_planning_strategy": "I'll search for specific {request_type} information related to their clarified request, focusing on the exact {request_type} policies, procedures, and forms that apply to their situation.",
    "tool_calls": [
        {{
            "tool": "KnowledgeSearchTool",
            "tool_running_message": "Searching for specific information based on your clarification...",
            "tool_completed_message": "Retrieved targeted information for your specific needs!",
            "tool_failed_message": "Targeted search had some issues, but found relevant information.",
            "args": {{"queries": ["specific {request_type} policies", "targeted {request_type} procedures", "{request_type} compliance requirements"]}}
        }},
        {{
            "tool": "FindCatalogTool",
            "tool_running_message": "Locating the most relevant form and services for your specific request...",
            "tool_completed_message": "Found the exact form and services you need!",
            "tool_failed_message": "Catalog search had limitations, but found relevant options.",
            "args": {{"queries": ["specific {request_type} request form", "targeted {request_type} service"]}}
        }}
    ]
}}

TURN 7 TOOL EXECUTION RESULTS:
[
    {{
        "tool_name": "KnowledgeSearchTool",
        "tool_args": {{"queries": ["specific query based on clarification", "targeted policy search", "relevant procedures for specific case"]}},
        "tool_result": {{
            "results": {{
                "topic_responses": [
                    {{
                        "answer": {{
                            "summary": "Specific policy and procedures for your type of request, including requirements, timelines, and approval processes.",
                            "title": "Targeted Policy Information"
                        }},
                        "chunks": [
                            {{
                                "title": "Specific_Policy_Document.pdf",
                                "url": "https://company.com/policies/specific-policy"
                            }}
                        ]
                    }}
                ]
            }}
        }}
    }},
    {{
        "tool_name": "FindCatalogTool",
        "tool_args": {{"queries": ["specific form search", "targeted service lookup"]}},
        "tool_result": {{
            "results": [
                {{
                    "item_id": "SPEC-001",
                    "workspace_id": "8",
                    "item_name": "Specific Request Form",
                    "category": "Targeted Services",
                    "category_type": "REQUEST",
                    "preamble": "Submit your specific type of request here",
                    "description": "Specialized form for your specific type of request with all required fields and approval workflow",
                    "tags": ["specific", "targeted", "request"]
                }}
            ]
        }}
    }}
]

TURN 8 ASSISTANT COMPREHENSIVE RESPONSE:
{{
    "reasoning": "Now I have the specific {request_type} information the user needs based on their clarification. I can provide a comprehensive response with exact {request_type} policies, procedures, and forms that apply to their situation.",
    "tool_planning_strategy": "I'll deliver a complete {request_type} solution with specific steps, requirements, timelines, and the exact form they need to use for their {request_type} request.",
    "tool_calls": [
        {{
            "tool": "RespondToUserTool",
            "tool_running_message": "Compiling comprehensive response with specific information for your request...",
            "tool_completed_message": "Delivered complete solution tailored to your specific needs!",
            "tool_failed_message": "Had difficulty compiling complete response, but provided relevant guidance.",
            "args": {{
                "success": true,
                "response": "Perfect! Now I can provide you with specific information for your {request_type} request.\\n\\n**Here's what you need to know about {request_type} requests:**\\n• {request_type.title()} policy requires proper documentation and approval\\n• Processing time: 3-5 business days for {request_type} requests\\n• Manager approval required for all {request_type} submissions\\n• Business justification must be provided\\n\\n**Next steps for your {request_type} request:**\\n1. Complete the {request_type} request form below\\n2. Attach required {request_type} documentation\\n3. Submit for manager and department approval\\n\\n**Ready to submit your {request_type} request?**\\n\\n<form>\\n{{\\n  \\"form_type\\": \\"service_request\\",\\n  \\"form_cta\\": \\"📋 Submit {request_type.title()} Request\\",\\n  \\"item_id\\": \\"{request_type.upper()}-SPEC-001\\",\\n  \\"workspace_id\\": \\"8\\"\\n}}\\n</form>\\n\\nThis form is specifically designed for {request_type} requests and includes all the necessary fields and approval workflow. Let me know if you need any clarification about the {request_type} process!"
            }}
        }}
    ]
}}

TURN 9 USER FOLLOW-UP QUESTION:
"How long does the approval process typically take for this type of request?"

TURN 10 ASSISTANT FINAL RESPONSE:
{{
    "reasoning": "The user is asking about timelines for the approval process, which I can answer based on the policy information I retrieved earlier.",
    "tool_planning_strategy": "I'll provide specific timeline information and explain the approval workflow, including any factors that might affect processing time.",
    "tool_calls": [
        {{
            "tool": "RespondToUserTool",
            "tool_running_message": "Providing timeline and approval process details...",
            "tool_completed_message": "Timeline and approval process information delivered!",
            "tool_failed_message": "Had difficulty providing timeline details, but gave general guidance.",
            "args": {{
                "success": true,
                "response": "Great question! For your specific type of request, here's the typical approval timeline:\\n\\n**Standard processing time:**\\n• Initial review: [X] business days\\n• Manager approval: [Y] business days\\n• Final processing: [Z] business days\\n• **Total time: [Total] business days**\\n\\n**Factors that can affect timing:**\\n• [Factor 1 that might delay or speed up]\\n• [Factor 2 that might affect timeline]\\n• [Factor 3 to consider]\\n\\n**To expedite your request:**\\n• Ensure all required fields are complete\\n• Attach any necessary documentation\\n• Follow up with your manager if needed\\n\\n**Status tracking:**\\nYou'll receive email notifications at each stage, and you can check the status in your requests dashboard.\\n\\nIs there anything else about the process you'd like me to explain?"
            }}
        }}
    ]
}}

Generate the conversation in this EXACT format:
{{
    "conversation": [
        {{"role": "user", "content": "{seed_question}"}},
        {{"role": "assistant", "content": {{INITIAL_TOOL_CALLS}}}},
        {{"role": "tool", "content": {{TOOL_EXECUTION_RESULTS_SHOWING_AMBIGUITY}}}},
        {{"role": "assistant", "content": {{CLARIFICATION_REQUEST}}}},
        {{"role": "user", "content": "I need [specific clarification that matches Option A from the choices]"}},
        {{"role": "assistant", "content": {{TARGETED_INFORMATION_GATHERING}}}},
        {{"role": "tool", "content": {{TARGETED_TOOL_EXECUTION_RESULTS}}}},
        {{"role": "assistant", "content": {{COMPREHENSIVE_RESPONSE}}}},
        {{"role": "user", "content": "How long does the approval process typically take for this type of request?"}},
        {{"role": "assistant", "content": {{FINAL_RESPONSE_WITH_TIMELINE}}}}
    ]
}}

ABSOLUTE REQUIREMENTS:
- Use seed question from taxonomy.yaml exactly as provided
- EXACTLY 10 turns total
- Show realistic ambiguity: {ambiguity_type}
- Assistant must ask for clarification when multiple options exist
- User provides clarification in Turn 5
- Second tool search is more targeted based on clarification
- Final responses are comprehensive and specific
- Include realistic forms and policies
- Timeline question in Turn 9 is always about approval process timing
- All responses maintain helpful tone while managing ambiguity

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
    
    generator = AmbiguityClarificationGenerator(
        model_name="o3-2025-04-16",
        backend="openai",
        backend_params={
            "max_retries": 3,
            "max_requests_per_minute": 60,
            "max_tokens_per_minute": 100000
        },
        generation_params={"max_completion_tokens": 6144}
    )
    
    dataset_items = []
    for i in range(num_conversations):
        seed_question = all_seeds[i % len(all_seeds)]
        dataset_items.append({
            "seed_question": seed_question
        })
    
    input_dataset = Dataset.from_list(dataset_items)
    
    print(f"Generating {num_conversations} Ambiguity & Clarification conversations...")
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
                if len(conversation) == 10:
                    conversations.append(conversation)
            except (json.JSONDecodeError, TypeError):
                if isinstance(result["full_conversation"], list) and len(result["full_conversation"]) == 10:
                    conversations.append(result["full_conversation"])
    
    dataset = {
        "conversations": conversations,
    }
    
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated {len(conversations)} Ambiguity & Clarification conversations and saved to {output_file}")
    
    if conversations:
        total_turns = sum(len(conv) for conv in conversations)
        avg_turns = total_turns / len(conversations)
        print(f"Average turns per conversation: {avg_turns:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Ambiguity & Clarification edge case conversations")
    parser.add_argument("--output", default="data/ambiguity_clarification/conversations.json", help="Output file")
    parser.add_argument("--num", type=int, default=2, help="Number of conversations to generate")
    parser.add_argument("--api-key", help="OpenAI API key")
    
    args = parser.parse_args()
    
    generate_sample_data(args.output, args.num, args.api_key) 