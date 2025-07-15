#!/usr/bin/env python3

import json

def extract_assistant_response(turn):
    try:
        if turn["role"] == "assistant" and "tool_calls" in turn["content"]:
            for tool_call in turn["content"]["tool_calls"]:
                if tool_call["tool"] == "RespondToUserTool":
                    return tool_call["args"]["response"]
    except:
        pass
    return ""

def add_conversation_history(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    for conversation in data["conversations"]:
        user_indices = []
        for i, turn in enumerate(conversation):
            if turn["role"] == "user":
                user_indices.append(i)
        
        if len(user_indices) <= 1:
            continue
        
        conversation_history = []
        
        for i, user_idx in enumerate(user_indices):
            if i == 0:
                user_msg = conversation[user_idx]["content"]
                
                for j in range(user_idx + 1, len(conversation)):
                    if conversation[j]["role"] == "assistant":
                        response = extract_assistant_response(conversation[j])
                        if response:
                            conversation_history.append(f"User: {user_msg}")
                            conversation_history.append(f"Assistant: {response}")
                            break
            else:
                original_content = conversation[user_idx]["content"]
                history_text = "\n".join(conversation_history)
                
                conversation[user_idx]["content"] = f"<conversation_history>\n{history_text}\n</conversation_history> {original_content}"
                
                user_msg = original_content
                for j in range(user_idx + 1, len(conversation)):
                    if conversation[j]["role"] == "assistant":
                        response = extract_assistant_response(conversation[j])
                        if response:
                            conversation_history.append(f"User: {user_msg}")
                            conversation_history.append(f"Assistant: {response}")
                            break
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Processed {len(data['conversations'])} conversations")

if __name__ == "__main__":
    add_conversation_history("/Users/ishaankumar/Documents/syn-data/data/sft_data_100.json", "/Users/ishaankumar/Documents/syn-data/data/sft_data_100_with_history.json") 