import json

def extract_info_from_jsonl(jsonl_file):
    unique_entries = set()
    results = []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            
            image_paths = data.get("image", [])
            if image_paths:
                image_number = image_paths[0].split('/')[1]
            
            conversations = data.get("conversations", [])
            task_description = ""
            for conversation in conversations:
                if conversation.get("from") == "human":
                    value = conversation.get("value", "")
                    start = value.find("Robot's Task:") + len("Robot's Task:")
                    end = value.find("Robot's history:") if "Robot's history:" in value else value.find("Your output format")
                    task_description = value[start:end].strip()
                    break
            
            entry = {
                "image_number": image_number,
                "task_description": task_description
            }
            
            entry_str = json.dumps(entry, sort_keys=True)
            if entry_str not in unique_entries:
                unique_entries.add(entry_str)
                results.append(entry)
    return results

annotation_jsonl_path = "" #the path of annotation jsonl(like "sat_DATASET_demo_head_rgb/xxx.jsonl")
output = extract_info_from_jsonl(annotation_jsonl_path)
task_prompt_json_path = "" #the path of task prompt json,should be like "sat_DATASET_demo_head_rgb/task_prompt.json"
with open(task_prompt_json_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=4, ensure_ascii=False)
