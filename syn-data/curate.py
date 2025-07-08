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
    return len(conversation) >= 3

def check_tool_calls(conversation: List[Dict[str, Any]]) -> bool:
    for turn in conversation: # check if the assistant has called a tool
        if turn.get("role") == "assistant":
            content = turn.get("content", {})
            tool_calls = content.get("tool_calls", [])
            if tool_calls:
                return True
    return False

def check_reasoning_coherence(conversation: List[Dict[str, Any]]) -> bool:
    """Check if reasoning mentions every tool that was actually called."""
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
                    logger.debug(f"Tool {tool} not mentioned in reasoning: {reasoning[:100]}...")
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

def check_conversation_quality(conversation: List[Dict[str, Any]]) -> bool:
    
    user_messages = [turn.get("content", "") for turn in conversation if turn.get("role") == "user"]
    if len(user_messages) < 2:
        return False
    
    failure_keywords = ["didn't work", "no results", "failed", "error", "down", "not helpful"]
    success_keywords = ["perfect", "great", "excellent", "thanks", "helpful", "worked"]
    
    user_text = " ".join(user_messages).lower()
    has_failure_scenario = any(keyword in user_text for keyword in failure_keywords)
    has_success_scenario = any(keyword in user_text for keyword in success_keywords)
    
    return has_failure_scenario or has_success_scenario

def filter_convos(infile: str, outfile: str) -> None:
    """
    Filter conversations based on curation heuristics.
    
    Args:
        infile: Input JSONL file path
        outfile: Output JSONL file path
    """
    seen_hashes: Set[str] = set()
    kept_count = 0
    total_count = 0
    
    with open(infile, 'r') as f_in, open(outfile, 'w') as f_out:
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
            
            total_count += 1
            
            try:
                conversation = json.loads(line)
                
                if not check_turn_count(conversation):
                    logger.debug(f"Line {line_num}: Failed turn count check")
                    continue
                
                if not check_tool_calls(conversation):
                    logger.debug(f"Line {line_num}: Failed tool calls check")
                    continue
                
                if not check_reasoning_coherence(conversation):
                    logger.debug(f"Line {line_num}: Failed reasoning coherence check")
                    continue
                
                first_msg_hash = get_first_user_message_hash(conversation)
                if first_msg_hash in seen_hashes:
                    logger.debug(f"Line {line_num}: Duplicate first message hash")
                    continue
                seen_hashes.add(first_msg_hash)
                
                if not check_invalid_assistant_turns(conversation):
                    logger.debug(f"Line {line_num}: Failed invalid assistant turns check")
                    continue
                
                if not check_conversation_quality(conversation):
                    logger.debug(f"Line {line_num}: Failed conversation quality check")
                    continue
                
                f_out.write(json.dumps(conversation) + '\n')
                kept_count += 1
                
            except json.JSONDecodeError as e:
                logger.debug(f"Line {line_num}: JSON decode error: {e}")
                continue
            except Exception as e:
                logger.debug(f"Line {line_num}: Unexpected error: {e}")
                continue
    
    logger.info(f"Kept {kept_count} out of {total_count} conversations")
    if total_count > 0:
        keep_rate = (kept_count / total_count) * 100
        logger.info(f"Keep rate: {keep_rate:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Filter and curate conversations for tool usage training")
    parser.add_argument("infile", help="Input JSONL file")
    parser.add_argument("outfile", help="Output JSONL file")
    args = parser.parse_args()
    
    try:
        filter_convos(args.infile, args.outfile)
    except FileNotFoundError:
        logger.error(f"Input file {args.infile} not found")
    except Exception as e:
        logger.error(f"Error during curation: {e}")

if __name__ == "__main__":
    main() 