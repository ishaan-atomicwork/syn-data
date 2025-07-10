import json
import argparse
import logging
import hashlib
from typing import List, Dict, Any, Set
import jsonschema

from utils.schema import RESPONSE_SCHEMA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_turn_count(conversation: List[Dict[str, Any]]) -> bool:
    return len(conversation) >= 5

def check_tool_calls(conversation: List[Dict[str, Any]]) -> bool:
    for turn in conversation: # check if the assistant has called a tool
        if turn.get("role") == "assistant":
            content = turn.get("content", {})
            tool_calls = content.get("tool_calls", [])
            if tool_calls:
                return True
    return False

def check_reasoning_coherence(conversation: List[Dict[str, Any]]) -> bool:
    """
    goal is to check if the assistant's reasoning mentions every tool that was actually called
    """

    for turn in conversation:
        if turn.get("role") == "assistant":
            content = turn.get("content", {})
            reasoning = content.get("reasoning", "").lower()
            tool_calls = content.get("tool_calls", [])
            
            called_tools = set()
            for call in tool_calls:
                tool_name = call.get("tool", "")
                called_tools.add(tool_name.lower())
            
            for tool in called_tools:
                tool_variations = [
                    tool.lower(),
                    tool.lower().replace("tool", ""),
                    tool.lower().replace("_", " "),
                    tool.lower().replace("_", "")
                ]
                
                found = False
                for variation in tool_variations:
                    if variation in reasoning:
                        found = True
                        break
                
                if not found:
                    logger.debug(f"tool {tool} not mentioned in reasoning")
                    return False
    return True

def get_first_user_message_hash(conversation: List[Dict[str, Any]]) -> str:
    for turn in conversation:
        if turn.get("role") == "user":
            content = turn.get("content", "")
            return hashlib.sha1(content.encode()).hexdigest()
    return ""

def check_invalid_assistant_turns(conversation: List[Dict[str, Any]]) -> bool:
    assistant_turns = [turn for turn in conversation if turn.get("role") == "assistant"]
    
    if not assistant_turns:
        return False
    
    invalid_count = 0
    for turn in assistant_turns:
        try:
            content = turn.get("content", {})
            jsonschema.validate(content, RESPONSE_SCHEMA)
        except (jsonschema.ValidationError, Exception):
            invalid_count += 1
    
    invalid_rate = invalid_count / len(assistant_turns)
    return invalid_rate <= 0.25

def check_ends_with_respond_to_user(conversation: List[Dict[str, Any]]) -> bool:
    """
    very imp to check if conversation ends with RespondToUserTool
    """
    if not conversation:
        return False
    
    last_turn = conversation[-1]
    if last_turn.get("role") != "assistant":
        return False
    
    content = last_turn.get("content", {})
    tool_calls = content.get("tool_calls", [])
    
    if not tool_calls:
        return False
    
    # Check if the last tool call is RespondToUserTool
    last_tool_call = tool_calls[-1]
    return last_tool_call.get("tool") == "RespondToUserTool"

def filter_convos(infile: str, outfile: str) -> None:
    """
    Filter conversations based on curation heuristics.
    
    Args:
        infile: Input JSON file path
        outfile: Output JSON file path
    """
    seen_hashes: Set[str] = set()
    kept_count = 0
    total_count = 0
    
    # Load the input JSON dataset
    with open(infile, 'r') as f:
        dataset = json.load(f)
    
    input_conversations = dataset.get("conversations", [])
    total_count = len(input_conversations)
    
    filtered_conversations = []
    
    for i, conversation in enumerate(input_conversations):
        
        if not check_turn_count(conversation):
            logger.debug(f"Conversation {i+1}: Failed turn count check")
            continue
        
        if not check_tool_calls(conversation):
            logger.debug(f"Conversation {i+1}: Failed tool calls check")
            continue
        
        if not check_reasoning_coherence(conversation):
            logger.debug(f"Conversation {i+1}: Failed reasoning coherence check")
            continue
        
        first_msg_hash = get_first_user_message_hash(conversation)
        if first_msg_hash in seen_hashes:
            logger.debug(f"Conversation {i+1}: Duplicate first message hash")
            continue
        seen_hashes.add(first_msg_hash)
        
        if not check_invalid_assistant_turns(conversation):
            logger.debug(f"Conversation {i+1}: Failed invalid assistant turns check")
            continue
        
        if not check_ends_with_respond_to_user(conversation):
            logger.debug(f"Conversation {i+1}: Failed ends with RespondToUserTool check")
            continue
        
        # Keep the conversation
        filtered_conversations.append(conversation)
        kept_count += 1
    
    output_dataset = {
        "conversations": filtered_conversations
    }
    
    with open(outfile, 'w') as f:
        json.dump(output_dataset, f, indent=2)
    
    logger.info(f"Kept {kept_count} out of {total_count} conversations")
    if total_count > 0:
        keep_rate = (kept_count / total_count) * 100
        logger.info(f"Keep rate: {keep_rate:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Filter and curate conversations for tool usage training")
    parser.add_argument("infile", help="Input JSON file")
    parser.add_argument("outfile", help="Output JSON file")
    args = parser.parse_args()
    
    try:
        filter_convos(args.infile, args.outfile)
    except Exception as e:
        logger.error(f"Error during curation: {e}")

if __name__ == "__main__":
    main() 