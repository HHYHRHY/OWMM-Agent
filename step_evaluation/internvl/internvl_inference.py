from PIL import Image
import numpy as np
import os
import json
from tqdm import tqdm
from PIL import Image
import argparse
from io import BytesIO
import base64
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from openai import OpenAI
from multiprocessing import Pool, cpu_count
import ast
import pdb
import time

def process_message(image_list,prompt):
    prompt = prompt.replace('<image>','<IMAGE_TOKEN>')
    content = [
        {
            "type":"text",
            "text": prompt
        },
    ]
    for image_path in image_list:
        with open(image_path,"rb") as image_token:
            temp_image_info = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64.b64encode(image_token.read()).decode('utf-8')}",
                }
            }
            content.append(temp_image_info)
    return content
def query_and_receive(content,client):
    model_name = client.models.list().data[0].id
    response = client.chat.completions.create(
        model=model_name,
        messages=[{
            'role':'user',
            'content': content
        }],
        temperature=0.0,
        top_p=0.5)
    return response

def process_item(item):
    image_list,query_text,query_dict = item
    try:
        content = process_message(image_list,query_text)
        start_time = time.time()
        response = query_and_receive(content,client)
        answer_info = response.choices[0].message.content
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        end_time = time.time()
        time_total = end_time - start_time
    except Exception as e:
        print(f"An error occurred: {e}")
        answer_info = [-1,-1]
        prompt_tokens = -1
        completion_tokens = -1
        time_total = -1
    return answer_info, query_dict,prompt_tokens,completion_tokens,time_total

if __name__ == '__main__':
    num_processes = 4
    client = OpenAI(api_key='123', base_url='http://0.0.0.0:33336/v1') #change the url to match your deployment
    answer_jsonl_path = 'add10_real_test_internvl38B.jsonl' #Inferenced json's path
    test_jsonl_paths = ['OWMM_real_test/real_test.jsonl'] #Test set Path
    text_tuple = []
    for test_jsonl_path in test_jsonl_paths: #pre process
        root_dir = os.path.dirname(test_jsonl_path)
        with open(test_jsonl_path, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
            for i,line in enumerate(lines):
                query_dict = json.loads(line)
                image_list = [os.path.join(root_dir,item)
                               for item in query_dict["image"]] #Change the path to match your own testset path
                query_text = query_dict["query"]
                text_tuple.append((image_list,query_text,query_dict))
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_item, text_tuple), total=len(text_tuple))) #inference

    with open(answer_jsonl_path, 'w') as f:
        for item in results:
            answer_point, query_dict,prompt_tokens,completion_tokens,time_total = item
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

