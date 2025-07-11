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


class ToolFailureRetryGenerator(curator.LLM):
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
        
        failure_types = [
            "timeout_error",
            "server_error_500", 
            "empty_result",
            "connection_error"
        ]
        
        failure_type = random.choice(failure_types)
        
        retry_strategies = [
            "automatic_retry_same_args",
            "reformulate_queries",
            "graceful_fallback_message"
        ]
        
        retry_strategy = random.choice(retry_strategies)
        
        failure_examples = {
            "timeout_error": {
                "error_msg": "Request timed out after 30 seconds",
                "description": "Tool execution timeout"
            },
            "server_error_500": {
                "error_msg": "Internal server error (500) - service temporarily unavailable",
                "description": "Server error response"
            },
            "empty_result": {
                "error_msg": "Tool completed successfully but returned empty results",
                "description": "Empty or no results returned"
            },
            "connection_error": {
                "error_msg": "Connection error - unable to reach service endpoint",
                "description": "Network connection failure"
            }
        }
        
        selected_failure = failure_examples[failure_type]
        
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
        
        return f"""Generate a realistic TOOL FAILURE & RETRY conversation with EXACTLY 8 turns following this MANDATORY pattern:

SEED QUESTION: {seed_question}

EDGE CASE: Tool call failure & retry loop
FAILURE TYPE: {selected_failure['description']}
RETRY STRATEGY: {retry_strategy}

CONVERSATION FLOW:
1. User: {seed_question}
2. Assistant: Initial tool calls
3. User: Tool execution results (with failures)
4. Assistant: Retry with {retry_strategy}
5. User: Second tool execution results (mixed success/partial failure)
6. Assistant: ONLY RespondToUserTool (addresses original request despite limitations)
7. User: Follow-up question
8. Assistant: ONLY RespondToUserTool (final response)

TURN 2 ASSISTANT INITIAL TOOL CALLS:
{{
    "reasoning": "{reasoning_map.get(request_type, reasoning_map['general'])}",
    "tool_planning_strategy": "{tool_planning_map.get(request_type, tool_planning_map['general'])}",
    "tool_calls": [
        {{
            "tool": "KnowledgeSearchTool",
            "tool_running_message": "Searching knowledge base for comprehensive information...",
            "tool_completed_message": "Successfully retrieved relevant information!",
            "tool_failed_message": "Knowledge search encountered issues, but continuing with available information.",
            "args": {{"queries": ["{request_type} policy and procedures", "{request_type} requirements and approval", "compliance guidelines"]}}
        }},
        {{
            "tool": "FindCatalogTool",
            "tool_running_message": "Locating relevant forms and services in catalog...",
            "tool_completed_message": "Found applicable forms and services!",
            "tool_failed_message": "Catalog search had limitations, but proceeding with found options.",
            "args": {{"queries": ["{request_type} request form", "{request_type} catalog services"]}}
        }}
    ]
}}

TURN 3 TOOL EXECUTION RESULTS (with failures):
[
    {{
        "tool_name": "KnowledgeSearchTool",
        "tool_args": {{"queries": ["comprehensive query based on request", "related policy information", "relevant procedures"]}},
        "tool_result": {{
            "error": "{selected_failure['error_msg']}",
            "success": false
        }}
    }},
    {{
        "tool_name": "FindCatalogTool", 
        "tool_args": {{"queries": ["relevant form or service", "related catalog items"]}},
        "tool_result": {{
            "error": "{selected_failure['error_msg']}",
            "success": false
        }}
    }}
]

TURN 4 ASSISTANT RETRY RESPONSE:
{self._generate_retry_response(retry_strategy)}

TURN 5 TOOL EXECUTION RESULTS (mixed success):
[
    {{
        "tool_name": "KnowledgeSearchTool",
        "tool_args": {{"queries": ["basic query", "simple information"]}},
        "tool_result": {{
            "results": {{
                "topic_responses": [
                    {{
                        "answer": {{
                            "summary": "Basic information found in knowledge base with limited details about the user's request.",
                            "title": "Limited Information Available"
                        }},
                        "chunks": [
                            {{
                                "title": "Basic_Guide.pdf",
                                "url": "https://company.com/basic-info"
                            }}
                        ]
                    }}
                ]
            }}
        }}
    }},
    {{
        "tool_name": "FindCatalogTool",
        "tool_args": {{"queries": ["general form", "basic service"]}},
        "tool_result": {{
            "results": [
                {{
                    "item_id": "GEN-001",
                    "workspace_id": "8",
                    "item_name": "General Request Form",
                    "category": "General Services",
                    "category_type": "REQUEST",
                    "preamble": "Submit general requests here",
                    "description": "Basic request form for various services",
                    "tags": ["general", "request", "basic"]
                }}
            ]
        }}
    }}
]

TURN 6 ASSISTANT RESPONSE (addresses limitations):
{{
    "reasoning": "Despite the initial tool failures, I was able to retrieve some basic information about {request_type} requests on the retry. While the results are limited due to the technical issues, I can still provide helpful guidance for {request_type} procedures based on what I found.",
    "tool_planning_strategy": "I'll provide the best assistance possible with the available {request_type} information, acknowledge the technical limitations, and offer alternative paths to get complete {request_type} support.",
    "tool_calls": [
        {{
            "tool": "RespondToUserTool",
            "tool_running_message": "Compiling response with available information and acknowledging limitations...",
            "tool_completed_message": "Provided response with available information and alternatives!",
            "tool_failed_message": "Encountered issues in response compilation, but provided what guidance was available.",
            "args": {{
                "success": true,
                "response": "I apologize, but I encountered some technical difficulties accessing our full knowledge base and catalog. However, I was able to gather some basic information about {request_type} requests to help you.\\n\\n**What I found for {request_type} requests:**\\n• Basic {request_type} procedures and requirements\\n• General approval workflow information\\n• Standard processing timelines\\n\\n**Due to technical limitations, I recommend:**\\n1. Contact the {request_type} support team directly\\n2. Check the main portal when systems are restored\\n3. Submit a general request form and specify {request_type} details\\n\\n**Available option:**\\n\\n<form>\\n{{\\n  \\"form_type\\": \\"service_request\\",\\n  \\"form_cta\\": \\"📋 Submit General Request\\",\\n  \\"item_id\\": \\"GEN-001\\",\\n  \\"workspace_id\\": \\"8\\"\\n}}\\n</form>\\n\\nI apologize for the limited {request_type} information due to technical issues. Is there anything specific about the {request_type} process I can help clarify?"
            }}
        }}
    ]
}}

TURN 7 USER FOLLOW-UP:
"Can you try to get more specific information about [specific aspect of original request]?"

TURN 8 ASSISTANT FINAL RESPONSE:
{{
    "reasoning": "The user is asking for more specific information, but given the ongoing technical issues with our information systems, I need to be honest about the limitations while still being as helpful as possible.",
    "tool_planning_strategy": "I'll acknowledge the request, explain the current technical limitations, and provide the best guidance I can offer along with alternative resources for getting complete information.",
    "tool_calls": [
        {{
            "tool": "RespondToUserTool",
            "tool_running_message": "Preparing response about current system limitations and alternative resources...",
            "tool_completed_message": "Provided transparent response about limitations and alternatives!",
            "tool_failed_message": "Had difficulty preparing response, but provided available guidance.",
            "args": {{
                "success": true,
                "response": "I understand you need more specific information, and I wish I could provide more detailed guidance. Unfortunately, our knowledge base and catalog systems are experiencing technical difficulties that are limiting my ability to access comprehensive information.\\n\\n**For more specific details, I recommend:**\\n• Contact our help desk directly at [contact information]\\n• Check our main portal at [portal URL] when systems are restored\\n• Submit the general request form and someone will follow up with detailed information\\n\\n**System status:**\\nOur IT team is working to resolve these technical issues. Normal service should be restored soon.\\n\\nI apologize for not being able to provide the complete information you need right now. Is there anything else I can help with using the limited information currently available?"
            }}
        }}
    ]
}}

Generate the conversation in this EXACT format:
{{
    "conversation": [
        {{"role": "user", "content": "{seed_question}"}},
        {{"role": "assistant", "content": {{INITIAL_TOOL_CALLS}}}},
        {{"role": "tool", "content": {{TOOL_EXECUTION_RESULTS_WITH_FAILURES}}}},
        {{"role": "assistant", "content": {{RETRY_RESPONSE}}}},
        {{"role": "tool", "content": {{SECOND_TOOL_EXECUTION_RESULTS}}}},
        {{"role": "assistant", "content": {{RESPOND_TO_USER_ADDRESSING_LIMITATIONS}}}},
        {{"role": "user", "content": "Can you try to get more specific information about [specific aspect of original request]?"}},
        {{"role": "assistant", "content": {{FINAL_RESPONSE_WITH_LIMITATIONS}}}}
    ]
}}

ABSOLUTE REQUIREMENTS:
- Use seed question from taxonomy.yaml exactly as provided
- EXACTLY 8 turns total
- Show realistic tool failure patterns: {failure_type}
- Demonstrate retry strategy: {retry_strategy}
- Assistant must acknowledge limitations gracefully
- Second attempt shows partial success (mixed results)
- Final responses must be helpful despite technical issues
- Include appropriate error handling and user communication
- All responses maintain professional tone while being transparent about issues

Available tools: {', '.join(AVAILABLE_TOOLS)}"""

    def _generate_retry_response(self, retry_strategy: str) -> str:
        if retry_strategy == "automatic_retry_same_args":
            return '''{{
    "reasoning": "The initial tool calls failed due to technical issues. I'll retry the same searches to see if the services are back online, as these temporary failures often resolve quickly.",
    "tool_planning_strategy": "I'll attempt the same tool calls again with identical parameters. If the services have recovered, this should provide the information needed.",
    "tool_calls": [
        {{
            "tool": "KnowledgeSearchTool",
            "tool_running_message": "Retrying knowledge base search...",
            "tool_completed_message": "Retry successful - retrieved information!",
            "tool_failed_message": "Second attempt also failed, but continuing with available options.",
            "args": {{"queries": ["comprehensive query based on request", "related policy information", "relevant procedures"]}}
        }},
        {{
            "tool": "FindCatalogTool",
            "tool_running_message": "Retrying catalog search...",
            "tool_completed_message": "Retry successful - found forms and services!",
            "tool_failed_message": "Second catalog search also failed, but proceeding with alternatives.",
            "args": {{"queries": ["relevant form or service", "related catalog items"]}}
        }}
    ]
}}'''
        elif retry_strategy == "reformulate_queries":
            return '''{{
    "reasoning": "The initial searches failed, possibly due to overly complex queries. I'll reformulate with simpler, more targeted search terms that might be more likely to succeed.",
    "tool_planning_strategy": "I'll use simpler, more basic search terms and try alternative tools if available. This approach often works better when services are experiencing issues.",
    "tool_calls": [
        {{
            "tool": "KnowledgeSearchTool",
            "tool_running_message": "Searching with simplified queries...",
            "tool_completed_message": "Simplified search successful!",
            "tool_failed_message": "Even simplified search failed, but continuing with available resources.",
            "args": {{"queries": ["basic query", "simple information"]}}
        }},
        {{
            "tool": "FindCatalogTool",
            "tool_running_message": "Looking for general forms and services...",
            "tool_completed_message": "Found general options!",
            "tool_failed_message": "General search also failed, but proceeding with standard options.",
            "args": {{"queries": ["general form", "basic service"]}}
        }}
    ]
}}'''
        else:  # graceful_fallback_message
            return '''{{
    "reasoning": "The tool failures indicate system issues that may persist. Rather than continue retrying and potentially wasting time, I'll provide a helpful response based on general knowledge and direct the user to alternative resources.",
    "tool_planning_strategy": "I'll acknowledge the technical difficulties and provide the best assistance I can with general knowledge, while directing the user to reliable alternative resources for complete information.",
    "tool_calls": [
        {{
            "tool": "RespondToUserTool",
            "tool_running_message": "Preparing response with general guidance and alternative resources...",
            "tool_completed_message": "Provided helpful response despite technical limitations!",
            "tool_failed_message": "Had difficulty even with fallback response, but provided basic guidance.",
            "args": {{
                "success": true,
                "response": "I apologize, but I'm experiencing technical difficulties accessing our knowledge base and catalog systems. While I can't provide the comprehensive information I'd normally offer, I can give you some general guidance.\\n\\n**General approach for your request:**\\n• [General guidance based on request type]\\n• [Common procedures or requirements]\\n• [Typical next steps]\\n\\n**Alternative resources:**\\n• Contact our support team directly\\n• Check our main portal when systems are restored\\n• Visit our help center for general information\\n\\n**Contact information:**\\n• Help desk: [contact details]\\n• Portal: [portal URL]\\n\\nI apologize for the technical issues. Once our systems are restored, I'll be able to provide more detailed assistance. Is there anything else I can help with in the meantime?"
            }}
        }}
    ]
}}'''

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
    
    generator = ToolFailureRetryGenerator(
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
    
    print(f"Generating {num_conversations} Tool Failure & Retry conversations...")
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
                if len(conversation) == 8:
                    conversations.append(conversation)
            except (json.JSONDecodeError, TypeError):
                if isinstance(result["full_conversation"], list) and len(result["full_conversation"]) == 8:
                    conversations.append(result["full_conversation"])
    
    dataset = {
        "conversations": conversations,
    }
    
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated {len(conversations)} Tool Failure & Retry conversations and saved to {output_file}")
    
    if conversations:
        total_turns = sum(len(conv) for conv in conversations)
        avg_turns = total_turns / len(conversations)
        print(f"Average turns per conversation: {avg_turns:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Tool Failure & Retry edge case conversations")
    parser.add_argument("--output", default="data/tool_failure_retry/conversations.json", help="Output file")
    parser.add_argument("--num", type=int, default=2, help="Number of conversations to generate")
    parser.add_argument("--api-key", help="OpenAI API key")
    
    args = parser.parse_args()
    
    generate_sample_data(args.output, args.num, args.api_key) 