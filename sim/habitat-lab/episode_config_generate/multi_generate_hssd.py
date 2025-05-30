import os
import multiprocessing
import subprocess
import time
import argparse
from tqdm import tqdm
from threading import Timer
import gc
import psutil,sys

def check_memory_usage(threshold_mb):
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / (1024 * 1024)  # 转换为MB
    for child in process.children(recursive=True):
        memory_usage_mb += child.memory_info().rss / (1024 * 1024)
    print(f"Memory: {memory_usage_mb:.2f} MB")

    if memory_usage_mb > threshold_mb:
        print(f"Memory >  {threshold_mb} MB, Exit")
        sys.exit()
def terminate_pool(pool):
    if pool is not None:
        pool.terminate()
        pool.close()
        pool.join()
        # print("Terminated pool due to timeout.")
def run_episode_generator(args):
    process_name = multiprocessing.current_process().name
    
    data_name,gpu_id,item,yaml_dir,output_dir = args
    output_file = f"./{output_dir}/{item}/{data_name}.json.gz"
    command = [
        "python",
        "./habitat-lab/habitat/datasets/rearrange/run_episode_generator.py",
        "--run",
        "--config", f"{yaml_dir}/{item}.yaml",
        "--gpu_id", f"{gpu_id}",
        "--num-episodes", f"{batch_num}",
        "--out", f"{output_file}",
        "--type", "manipulation",
        "--resume", "habitat-mas/habitat_mas/data/robot_resume/FetchRobot_default.json"
    ]
    log_file = f"./log/sample/{data_name}.log"
    # with open(log_file, "w") as f:
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # with open(log_file, 'w') as log:
    #     subprocess.run(command, stdout=log, stderr=log)
    time.sleep(0.9)
def get_numbers_from_filenames(directory):
    numbers = []
    for filename in os.listdir(directory):
        if filename.endswith('.scene_instance.json'):
            number = filename.split('.scene_instance')[0]
            numbers.append(number)
    return numbers
def extract_keys_from_txt(file_path):
    keys = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or not line.startswith('(') or not line.endswith(')'):
                continue
            try:
                content = line[1:-1].split(',', 1)
                key = content[0].strip().strip("'")
                keys.append(key)
            except Exception as e:
                print(f"Error processing line: {line}, Error: {e}")
    return keys
if __name__ == '__main__':
    sum_episode = 400
    process_num = 50
    batch_num = 4
    parser = argparse.ArgumentParser(description="Setup dataset and log paths.")
    parser.add_argument('--start_dir', type=int, required=True, help='Start')
    parser.add_argument('--end_dir', type=int, required=True, help='End')
    parser.add_argument('--gpu_num', type=int, default=5)
    args = parser.parse_args()
    start_dir = args.start_dir
    end_dir = args.end_dir
    gpu_num = args.gpu_num
    num = int(sum_episode / batch_num)
    yaml_dir = "./allgoogle_dir"
    output_dir = 'hssd_scene_google_test'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    zip_files = [f"data_{i}" for i in range(0,int(sum_episode/batch_num))]
    
    timeout = 1500

    file_path = 'test_id.txt'
    keys = extract_keys_from_txt(file_path)
    log_path = "./log/sample"
    scene_sample = keys[start_dir:end_dir]
    os.makedirs(log_path,exist_ok=True)
    memory_threshold = 450976
    for a,item in enumerate(scene_sample):
        if not os.path.exists(os.path.join(output_dir, item)):
            os.makedirs(os.path.join(output_dir, item))
        start_time = time.time()
        check_memory_usage(memory_threshold)
        print(f"START--{item}//i---{str(a)}")
        with multiprocessing.Pool(processes=process_num) as pool:
            args = [(f"data_{i}",int(i%gpu_num),item,yaml_dir,output_dir) for i in range(0,int(sum_episode/batch_num))]
            results = []
            async_results = [pool.apply_async(run_episode_generator, (arg,), error_callback=lambda e: print(f"Error: {e}")) for arg in args]
            timer = Timer(timeout, terminate_pool, [pool])
            timer.start()
            try:
                pool.close()
                pool.join()
            finally:
                timer.cancel()
                end_time = time.time()
                item_process_time = end_time - start_time
                terminate_pool(pool)
            time.sleep(0.9)
            print(f"FINISH--{item};TIME:{item_process_time}")
        gc.collect()
    
