from PIL import Image
import numpy as np
import os
import json
from pivot_package.run_pivot import run_pivot
from tqdm import tqdm
from PIL import Image
import argparse
import io
from io import BytesIO
import base64
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from openai import OpenAI
import openai
from multiprocessing import Pool, cpu_count
import ast
import re
import pdb
import time
def encode_image_base64_with_max_size(image_path: str, max_size: int = 512,reshape_path="reshaped_image") -> str:
    """
    Encode an image to base64 with a maximum size.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    image = Image.open(image_path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    width, height = image.size
    if width > height:
        left = (width - height) / 2
        top = 0
        right = (width + height) / 2
        bottom = height
    else:
        left = 0
        top = (height - width) / 2
        right = width
        bottom = (height + width) / 2
    image = image.crop((left, top, right, bottom))
    image = image.resize((max_size, max_size), Image.Resampling.LANCZOS)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    os.makedirs(reshape_path,exist_ok=True)
    image.save(os.path.join(reshape_path,os.path.basename(image_path)),format="PNG")
    img_bytes = buffered.getvalue()
    image_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return image_base64
def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((512, 512))
    image_array = np.array(image)
    return image_array
def process_message(image_list,prompt):
    content = [
        {
            "type":"text",
            "text": prompt
        },
    ]
    for image_path in image_list:
        temp_image_info = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encode_image_base64_with_max_size(image_path)}",
            }
        }
        content.append(temp_image_info)
    return content
def query_and_receive(content):
    response = openai.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[{
            'role':'user',
            'content': content
        }],
        temperature=0.0,
        top_p=0.5,
        response_format={"type":"json_object"}
        )
    return response
def process_prompt_for_pivot(image,query_text,gpt_action):
    task_match = re.search(
        r"Robot's Task:\s*(.*?)\s*(?=Robot's history:|Your output format should be)",
        query_text,
        re.DOTALL
    )
    if task_match:
        task_content = task_match.group(1).strip()
    else:
        raise ValueError("Robot's Task not found")
    history_match = re.search(
        r"Robot's history:\s*(.*?)\s*(?=Your output format should be)",
        query_text,
        re.DOTALL
    )
    if history_match:
        history_content = history_match.group(1).strip()
    else:
        history_content = None
    robot_history = "" if not history_content else f"Robot's history: {history_content}"
    pivot_prompt = f"""The robot need to {task_content}.Now the robot need to {gpt_action}.{robot_history}"""
    return load_and_preprocess_image(image[-1]),pivot_prompt
def process_item(item):
    image_list,query_text,query_dict = item
    try:
        content = process_message(image_list,query_text)
        start_time = time.time()
        response = query_and_receive(content)
        gpt_dm = json.loads(response.choices[0].message.content)
        prompt_tokens_dm = response.usage.prompt_tokens
        completion_tokens_dm = response.usage.completion_tokens
        if gpt_dm["action"] == "search_scene_frame":
            answer_info = gpt_dm["action_information"]
            prompt_tokens = prompt_tokens_dm
            completion_tokens = completion_tokens_dm
        else:
            image_tensor,query_pivot = process_prompt_for_pivot(image_list,query_text,gpt_dm["action"])
            answer_info,prompt_tokens_pivot,completion_tokens_pivot = run_pivot(
                    im=image_tensor,
                    query=query_pivot,
                    n_samples_init=10,
                    n_samples_opt=6,
                    n_iters=2,
                    n_parallel_trials=1,
                    openai_api_key=API_KEY,
                    openai_base_url=BASE_URL,
                )
            prompt_tokens = prompt_tokens_dm + prompt_tokens_pivot
            completion_tokens = completion_tokens_dm + completion_tokens_pivot
        end_time = time.time()
        time_total = end_time - start_time
        answer_return = {
            "answer_info":answer_info,
            "action":gpt_dm["action"]
        }
    except Exception as e:
        print(f"An error occurred: {e}")
        answer_return = [-1,-1]
        prompt_tokens = -1
        completion_tokens = -1
        time_total = -1
    return answer_return, query_dict,prompt_tokens,completion_tokens,time_total

if __name__ == '__main__':
    API_KEY = "abc"
    BASE_URL = "xxx"
    openai.api_key = API_KEY
    openai.base_url = BASE_URL
    num_processes = 3
    answer_jsonl_path = 'add10_real_test_pivot_gptvanilla.jsonl' #Inferenced json's path
    test_jsonl_paths = ['OWMM_real_test_add10/addto10_annotation.jsonl'] #Test set Path
    text_tuple = []
    for test_jsonl_path in test_jsonl_paths:
        root_dir = os.path.dirname(test_jsonl_path)
        with open(test_jsonl_path, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
            for i,line in enumerate(lines):
                query_dict = json.loads(line)
                image_list = [os.path.join(root_dir,item)
                               for item in query_dict["image"]] #Change the path to match your own testset path
                query_text = query_dict["query"]
                text_tuple.append((image_list,query_text,query_dict))
    pre_and_anno = []
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_item, text_tuple), total=len(text_tuple)))

    with open(answer_jsonl_path, 'w') as f:
        for item in results:
            answer_point, query_dict,prompt_tokens,completion_tokens,time_total = item
            try:
                for key, value in answer_point.items():
                    if isinstance(value, list):
                        answer_point[key] = [int(x) if isinstance(x, np.int64) else x for x in value]
                    elif isinstance(value, np.int64):
                        answer_point[key] = int(value)
                query_dict["prediction"] = json.dumps(answer_point)
            except:
                query_dict["prediction"] = answer_point
            query_dict["prompt_tokens"] = prompt_tokens
            query_dict["completion_tokens"] = completion_tokens
            query_dict["time_total"] = time_total        
            for key, value in query_dict.items():
                if isinstance(value, list):
                    query_dict[key] = [int(x) if isinstance(x, np.int64) else x for x in value]
                elif isinstance(value, np.int64):
                    query_dict[key] = int(value)
            json_line = json.dumps(query_dict)
            f.write(json_line + '\n')

