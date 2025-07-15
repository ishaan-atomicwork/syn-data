#!/usr/bin/env python3
"""
Combined dataset generation script for edge cases.

This script generates different types of edge case conversations with specified proportions:
- 30% single turns (happy path)
- 40% multi turns (complex conversations)
- 10% ambiguity clarification
- 10% tool failure retry
- 10% JSON error handling

All conversations are combined into a single JSON file.
"""

import json
import argparse
import os
import tempfile
from typing import Optional
import math

from single_turn import generate_sample_data as generate_single_turn
from multi_turn import generate_sample_data as generate_multi_turn
from ambiguity_clarification import generate_sample_data as generate_ambiguity
from tool_failure_retry import generate_sample_data as generate_tool_failure
from json_error import generate_sample_data as generate_json_error


def generate_combined_dataset(
    total_conversations: int,
    output_file: str,
    api_key: Optional[str] = None
):
    """
    Generate a combined dataset with different edge case types.
    
    Args:
        total_conversations: Total number of conversations to generate
        output_file: Path to save the combined dataset
        api_key: OpenAI API key for generation
    """
    
    single_turn_count = math.ceil(total_conversations * 0.3)
    multi_turn_count = math.ceil(total_conversations * 0.4)
    ambiguity_count = math.ceil(total_conversations * 0.1)
    tool_failure_count = math.ceil(total_conversations * 0.1)
    json_error_count = math.ceil(total_conversations * 0.1)
    
    actual_total = single_turn_count + multi_turn_count + ambiguity_count + tool_failure_count + json_error_count
    if actual_total > total_conversations:
        single_turn_count -= (actual_total - total_conversations)
    
    print(f"Generating {total_conversations} total conversations:")
    print(f"  - Single turn (30%): {single_turn_count}")
    print(f"  - Multi turn (40%): {multi_turn_count}")
    print(f"  - Ambiguity (10%): {ambiguity_count}")
    print(f"  - Tool failure (10%): {tool_failure_count}")
    print(f"  - JSON error (10%): {json_error_count}")
    print()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_files = {}
        all_conversations = []
        
        generators = [
            ("single_turn", generate_single_turn, single_turn_count),
            ("multi_turn", generate_multi_turn, multi_turn_count),
            ("ambiguity", generate_ambiguity, ambiguity_count),
            ("tool_failure", generate_tool_failure, tool_failure_count),
            ("json_error", generate_json_error, json_error_count)
        ]
        
        for edge_case_type, generator_func, count in generators:
            if count > 0:
                temp_file = os.path.join(temp_dir, f"{edge_case_type}.json")
                temp_files[edge_case_type] = temp_file
                
                print(f"Generating {count} {edge_case_type} conversations...")
                try:
                    print(f"  → Calling generator for {edge_case_type}...")
                    generator_func(temp_file, count, api_key)
                    
                    print(f"  → Generator completed, checking file: {temp_file}")
                    if not os.path.exists(temp_file):
                        print(f"  ✗ Output file was not created: {temp_file}")
                        continue
                        
                    with open(temp_file, 'r') as f:
                        data = json.load(f)
                        conversations = data.get('conversations', [])
                        
                        for conv in conversations:
                            all_conversations.append(conv)
                        
                        print(f"  ✓ Generated {len(conversations)} {edge_case_type} conversations")
                        
                except Exception as e:
                    import traceback
                    print(f"  ✗ Error generating {edge_case_type}: {e}")
                    print(f"  ✗ Full traceback: {traceback.format_exc()}")
                    continue
        
        combined_dataset = {
            "conversations": all_conversations
        }
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(combined_dataset, f, indent=2)
        
        print(f"Combined dataset saved to: {output_file}")
        print(f"Total conversations generated: {len(all_conversations)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate combined edge case dataset with specified proportions"
    )
    parser.add_argument(
        "--total", 
        type=int, 
        required=True,
        help="Total number of conversations to generate"
    )
    parser.add_argument(
        "--output", 
        default="/Users/ishaankumar/Documents/syn-data/data/sft_data_100.json",
        help="Output file path"
    )
    parser.add_argument(
        "--api-key", 
        help="OpenAI API key for generation"
    )
    
    args = parser.parse_args()
    
    if args.total <= 0:
        print("Error: Total conversations must be greater than 0")
        return 1
    
    try:
        generate_combined_dataset(
            total_conversations=args.total,
            output_file=args.output,
            api_key=args.api_key
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 