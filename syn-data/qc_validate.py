import json
import argparse
import logging
from typing import List, Dict, Any
import jsonschema

from utils.schema import RESPONSE_SCHEMA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_conversation(conversation: List[Dict[str, Any]]) -> bool:
    """
    Validate that all assistant turns in a conversation follow the JSON schema.
    
    Args:
        conversation: List of conversation turns
        
    Returns:
        True if all assistant turns are valid, False otherwise
    """
    assistant_turns = [turn for turn in conversation if turn.get("role") == "assistant"]
    
    if not assistant_turns:
        return False
    
    for turn in assistant_turns:
        try:
            content = turn.get("content", {})
            jsonschema.validate(content, RESPONSE_SCHEMA)
        except jsonschema.ValidationError as e:
            logger.debug(f"Validation error: {e}")
            return False
        except Exception as e:
            logger.debug(f"Unexpected error during validation: {e}")
            return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Validate JSON schema compliance")
    parser.add_argument("input_file", help="Input JSONL file to validate")
    args = parser.parse_args()
    
    valid_count = 0
    invalid_count = 0
    total_count = 0
    
    try:
        with open(args.input_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                total_count += 1
                
                try:
                    conversation = json.loads(line)
                    if validate_conversation(conversation):
                        valid_count += 1
                    else:
                        invalid_count += 1
                        logger.debug(f"Invalid conversation at line {line_num}")
                except json.JSONDecodeError as e:
                    invalid_count += 1
                    logger.debug(f"JSON decode error at line {line_num}: {e}")
                except Exception as e:
                    invalid_count += 1
                    logger.debug(f"Unexpected error at line {line_num}: {e}")
    
    except FileNotFoundError:
        logger.error(f"Input file {args.input_file} not found")
        return
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        return
    
    print(f"Valid {valid_count} / Invalid {invalid_count}")
    
    if total_count > 0:
        validity_rate = (valid_count / total_count) * 100
        logger.info(f"Validation rate: {validity_rate:.1f}% ({valid_count}/{total_count})")
    else:
        logger.info("No conversations found in input file")

if __name__ == "__main__":
    main() 