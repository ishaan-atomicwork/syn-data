from typing import Dict, List, Any
import random

class ToolResponseFormats:
    """Defines realistic response formats for each tool based on actual implementations"""
    
    @staticmethod
    def knowledge_search_tool_response() -> Dict[str, Any]:
        """Format for KnowledgeSearchTool based on kb_tool.py"""
        return {
            "results": {
                "topic_responses": [
                    {
                        "answer": {
                            "summary": "To get access to VPN or remote access for working from outside the office, please submit a request to the IT department. They will provide you with the VPN details and guide you through the installation and connection process.",
                            "title": ""
                        },
                        "chunks": [
                            {
                                "title": f"IT_FAQ_{random.randint(1, 100)}.pdf",
                                "url": ""
                            }
                        ]
                    }
                ],
                "conversation_responses": [],
                "verified_answer_responses": [],
                "default_knowledge_responses": []
            }
        }
    
    @staticmethod
    def find_catalog_tool_response() -> Dict[str, Any]:
        """Format for FindCatalogTool based on find_catalog_tool.py"""
        item_id = str(random.randint(100, 1200))
        item_name = random.choice(["VPN Setup", "Laptop Request", "Software License", "Access Request"])
        
        return {
            "results": [
                {
                    "item_id": item_id,
                    "workspace_id": "8",
                    "item_name": item_name,
                    "category": "Security and Network Access",
                    "category_type": "UNKNOWN",
                    "preamble": f"We've found the {item_name} in our catalog, would you like to raise a request to access it?",
                    "description": f"Service for {item_name.lower()} with secure access and support",
                    "llm_description": f"The {item_name} service provides secure access and support. This service ensures reliable connectivity and proper setup for your needs.",
                    "tags": ["security", "access", "service"]
                }
            ]
        }
    
    @staticmethod
    def my_requests_tool_response() -> Dict[str, Any]:
        """Format for MyRequestsTool based on esd_tools/request_status/base_req.py"""
        return {
            "success": True,
            "response": {
                "page": 1,
                "page_size": 5,
                "total_pages": 1,
                "total_count": 2,
                "next_page_token": None,
                "data": [
                    {
                        "id": f"REQ-{random.randint(1000, 9999)}",
                        "subject": "VPN Access Request",
                        "display_id": f"DISP-{random.randint(100, 999)}",
                        "approval_task_title": "VPN Access Approval",
                        "entity_type": "service_request",
                        "request_type": "access",
                        "requester": {
                            "name": "John Doe",
                            "email": "john.doe@company.com"
                        },
                        "description_text": "Request for VPN access to work remotely",
                        "pending_since": "2025-01-15T10:00:00Z",
                        "current_user_approver": False
                    }
                ]
            }
        }
    
    @staticmethod
    def my_pending_approvals_tool_response() -> Dict[str, Any]:
        """Format for MyPendingApprovalsTool based on esd_tools/request_status/base_req.py"""
        return {
            "success": True,
            "response": {
                "page": 1,
                "page_size": 5,
                "total_pages": 1,
                "total_count": 1,
                "next_page_token": None,
                "data": [
                    {
                        "id": f"APP-{random.randint(1000, 9999)}",
                        "subject": "Software License Request",
                        "display_id": f"DISP-{random.randint(100, 999)}",
                        "approval_task_title": "Software License Approval",
                        "entity_type": "service_request",
                        "request_type": "license",
                        "requester": {
                            "name": "Jane Smith",
                            "email": "jane.smith@company.com"
                        },
                        "description_text": "Request for Adobe Creative Cloud license",
                        "pending_since": "2025-01-14T14:30:00Z",
                        "current_user_approver": True
                    }
                ]
            }
        }
    
    @staticmethod
    def respond_to_user_tool_response() -> Dict[str, Any]:
        """Format for RespondToUserTool based on finish_tool/atom_finish_tool.py"""
        return {
            "execution_status": "success",
            "results": {
                "process_status": True,
                "response": "Generated response to user"
            }
        }
    
    @staticmethod
    def system_diagnostic_tool_response() -> Dict[str, Any]:
        """Format for SystemDiagnosticTool based on mac_diagnosis_tools/system_diagnostic_tool.py"""
        return {
            "status": "healthy",
            "details": {
                "diagnostic_results": {
                    "cpu": {
                        "status": "normal",
                        "details": {"cpu_usage": f"{random.randint(20, 60)}%"}
                    },
                    "memory": {
                        "status": "normal", 
                        "details": {"memory_usage": f"{random.randint(30, 70)}%"}
                    },
                    "disk": {
                        "status": "normal",
                        "details": {"disk_usage": f"{random.randint(40, 80)}%"}
                    }
                },
                "critical_issues": []
            }
        }
    
    @staticmethod
    def browser_tool_response() -> Dict[str, Any]:
        """Format for BrowserTool based on browser_tool/browser_tool.py"""
        return {
            "execution_status": "success",
            "results": "Data from various websites: \n1. URL: https://example.com/help Data from the website: Helpful information about the topic\n2. URL: https://support.example.com Data from the website: Additional support documentation\n"
        }
    
    @staticmethod
    def tavily_tool_response() -> Dict[str, Any]:
        """Format for TavilyTool based on tavily_tool/tavily_tool.py"""
        return {
            "execution_status": "success",
            "selected_websites": [
                {
                    "name": "Apple Support",
                    "description": "Official Apple support documentation",
                    "url": "https://support.apple.com"
                }
            ],
            "search_results": [
                {
                    "url": "https://support.apple.com/help",
                    "text": "Detailed information about troubleshooting and support"
                }
            ]
        }

# Tool name to response format mapping
TOOL_RESPONSE_FORMATS = {
    "KnowledgeSearchTool": ToolResponseFormats.knowledge_search_tool_response,
    "FindCatalogTool": ToolResponseFormats.find_catalog_tool_response,
    "MyRequestsTool": ToolResponseFormats.my_requests_tool_response,
    "MyPendingApprovalsTool": ToolResponseFormats.my_pending_approvals_tool_response,
    "RespondToUserTool": ToolResponseFormats.respond_to_user_tool_response,
    "SystemDiagnosticTool": ToolResponseFormats.system_diagnostic_tool_response,
    "BrowserTool": ToolResponseFormats.browser_tool_response,
    "TavilyTool": ToolResponseFormats.tavily_tool_response
}

# Available tools list
AVAILABLE_TOOLS = [
    "KnowledgeSearchTool",
    "FindCatalogTool", 
    "MyRequestsTool",
    "MyPendingApprovalsTool",
    "RespondToUserTool",
    "SystemDiagnosticTool",
    "BrowserTool",
    "TavilyTool"
] 