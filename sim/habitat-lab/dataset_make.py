
import multiprocessing
import subprocess
import time
from tqdm import tqdm
from ruamel.yaml import YAML
from dataset_make_script import datatrans_2_end_sat_waypoint_closer
import os
from dataset_make_script import process_directory
import shutil,random
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED,TimeoutError
import zipfile
import concurrent.futures
from tqdm import tqdm
import time
import gzip
import shutil
import os,pdb
import time
from threading import Timer
import json
import argparse
yaml = YAML()
def terminate_pool(pool):
    if pool is not None:
        pool.terminate()
        pool.close()
        pool.join()
        print("Terminated pool due to timeout.")
class SkipCurrentThread(Exception):
    pass
def run_with_timeout(func, *args, timeout=2, process_dir=None, skip_len=40, sample_clip=500, retries=1, data_name=None):
    for attempt in range(retries):
        with ThreadPoolExecutor() as executor:
            future = executor.submit(func, *args, process_dir=process_dir, skip_len=skip_len, sample_clip=sample_clip)
            try:
                result = future.result(timeout=timeout)
                return result
            except:
                return -1
    
def unzip_gz_file(gz_file_path, extract_to):
    os.makedirs(extract_to, exist_ok=True)
    output_file_path = os.path.join(extract_to, os.path.basename(gz_file_path).replace('.gz', ''))
    with gzip.open(gz_file_path, 'rb') as gz_file:
        with open(output_file_path, 'wb') as out_file:
            shutil.copyfileobj(gz_file, out_file)
def clear_directory(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def run_script(args):
    file_path,skip_len, base_directory,gpu_id,scene,scene_dataset_dir = args
    a = ["disk"]
    seed = random.randint(1, 1000000)
    cmd = [
        "python", "-u", "-m", "habitat_baselines.run",
        "--config-name=social_rearrange/llm_fetch_stretch_manipulation.yaml",
        "habitat_baselines.evaluate=True",
        "habitat_baselines.num_environments=1",
        f"habitat_baselines.eval.json_option={a}",
        f"habitat.simulator.habitat_sim_v0.gpu_device_id={gpu_id}",
        f"habitat_baselines.torch_gpu_id={gpu_id}",
        f"habitat_baselines.eval.video_option={a}",
        f"habitat_baselines.eval.video_option_new=False",
        f"habitat_baselines.video_dir={base_directory}/video_dir_",
        f"habitat_baselines.image_dir={base_directory}/{scene}/{file_path}",
        f"habitat.seed={seed}",
        f"habitat.environment.max_episode_steps={max_step}",
        f"habitat.dataset.data_path=data/datasets/{scene_dataset_dir}/{scene}/{file_path}"
    ]
    log_file = f"{log_dir}/{scene}_example_{file_path}.log"
    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    seed_txt_path = os.path.join(f"{base_directory}/{scene}/{file_path}",'seed.txt')
    time.sleep(0.7)
    process_directory(os.path.join(base_directory,scene,file_path),skip_len=skip_len)
    print("finish_trans",flush=True)
    flag = 0
    with ThreadPoolExecutor() as executor:
        future_set = executor.submit(datatrans_2_end_sat_waypoint_closer, process_dir=f"./{base_directory}/{scene}/{file_path}", skip_len=skip_len, sample_clip=max_step)
        try:
            result = future_set.result(timeout=10)
            sample_info = result
        except:
            return (False, -1)
    if len(sample_info)==0 or sample_info == -1:
        print(f"no sample_info,Exiting process,from{file_path}")
        return (False, 0)
    sample = str(sample_info)
    print(f"{file_path}'s sample:",sample)
    sample_yaml_path = './habitat-lab/habitat/config/benchmark/single_agent/fetch_sample.yaml'
    with open(sample_yaml_path,'r') as file:
        data = yaml.load(file)
    data['habitat']['dataset']['data_path'] = f"data/datasets/{scene_dataset_dir}/{scene}/{file_path}"
    data['habitat']['simulator']['habitat_sim_v0']['gpu_device_id'] = gpu_id
    relative_yaml_dir_path = os.path.join(f"sample_frame_dataprocess_{base_dir_name}",scene)
    scene_yaml_dir_path = os.path.join('./habitat-lab/habitat/config/benchmark/single_agent/override',relative_yaml_dir_path)
    os.makedirs(scene_yaml_dir_path,exist_ok=True)
    yaml_path = os.path.join(scene_yaml_dir_path,f"{file_path}.yaml")
    with open(yaml_path,'w') as file:
        yaml.dump(data,file)
    obs_key = obs_key_set
    sample_info_str = json.dumps(sample_info)
    command = [
        "python",
        "-u",
        "habitat-lab/habitat/datasets/rearrange/generate_episode_graph_images.py",
        "--config", f"benchmark/single_agent/override/sample_frame_dataprocess_{base_dir_name}/{scene}/{file_path}.yaml",
        "--gpu_id", f"{gpu_id}",
        "--output_dir", f"{base_directory}/{scene}/{file_path}",
        "--generate_type", "sample_frame",
        "--obs_keys",obs_key,
        "--sample_info", sample_info_str
    ]
    
    log_file = f"{log_dir}/sampleframe_{scene}_example_{file_path}.log"
    with open(log_file, "w") as f:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT)
    time.sleep(0.5)
    try:
        with open(seed_txt_path,'w') as f:
            f.write(str(seed))
    except:
        pass
    return (True, 1)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Setup dataset and log paths.")
    parser.add_argument('--sum_episode', type=int, default=400, help='Total number of episodes')
    parser.add_argument('--epnum_per_gz', type=int, default=4, help='Number of episodes per gz')
    parser.add_argument('--gz_start', type=int, default=40, help='Starting gz index')
    parser.add_argument('--gz_end', type=int, default=50, help='Try Time')
    parser.add_argument('--skip_len', type=int, default=27, help='Length to skip')
    parser.add_argument('--base_directory', type=str, default='DATASET_YCB_0109', help='Base directory name')
    parser.add_argument('--process_num', type=int, default=10, help='Number of processes')
    parser.add_argument('--gpu_number', type=int, default=4, help='Number of GPUs')  # Updated default value
    parser.add_argument('--max_step', type=int, default=800, help='Maximum number of steps')
    parser.add_argument('--jump_gz', type=int, default=13, help='Jump gz value')
    parser.add_argument('--repeat_time', type=int, default=1, help='Repeat time')
    parser.add_argument('--scene_dataset_dir', type=str, default='hssd_scene_allycb_train_2', help='Scene dataset directory')
    parser.add_argument('--sample_scene', type=str, nargs='+', default=["106878960_174887073","108736851_177263586","108736824_177263559","107734254_176000121"], help='Sample scenes')
    parser.add_argument('--timeout', type=int, default=800, help='Sampling timeout setting')
    parser.add_argument('--restart_scene', type=str, default="")
    parser.add_argument('--start_scene',type=int,default=None)
    parser.add_argument('--end_scene',type=int,default=None)
    return parser.parse_args()
if __name__ == "__main__":
    obs_key_set = "arm_workspace_rgb"
    dp_config = parse_arguments()
    current_time = time.time()
    local_random = random.Random(current_time)
    random_number = local_random.randint(1, 100000)
    max_step = dp_config.max_step
    scene_dataset_dir = dp_config.scene_dataset_dir
    gz_start_gz = dp_config.gz_start
    gz_end = dp_config.gz_end
    base_dir_name = f"{dp_config.base_directory}_{gz_start_gz}_{random_number}"
    base_dir_path = f"./{dp_config.base_directory}_{gz_start_gz}_{random_number}"
    sum_episode = dp_config.sum_episode
    epnum_per_gz = dp_config.epnum_per_gz
    gz_sum = int(sum_episode/epnum_per_gz)
    repeat_time = dp_config.repeat_time
    skip_len = dp_config.skip_len
    gpu_num = dp_config.gpu_number
    # print("gpu_num:",gpu_num)
    sum_episode = dp_config.sum_episode
    process_num = dp_config.process_num
    log_dir = f'./log/{base_dir_name}'
    os.makedirs(log_dir, exist_ok=True)
    num_gz = int(gz_end - gz_start_gz)
    sample_scene_dir_path = os.path.join('data/datasets',scene_dataset_dir)
    sample_scene_dir = []
    for entry in os.scandir(sample_scene_dir_path):
        if entry.is_dir():
            sample_scene_dir.append(os.path.basename(entry.path))
    sample_scene_dir = sorted(sample_scene_dir)
    restart_scene = dp_config.restart_scene
    if len(restart_scene) > 0:
        if restart_scene in sample_scene_dir:
            restart_scene_id = sample_scene_dir.index(restart_scene)
            sample_scene_dir = sample_scene_dir[restart_scene_id:]
        else:
            raise ValueError(f"restart_scene:{restart_scene} not in sample_scene_dir")
    start_scene = dp_config.start_scene
    end_scene = dp_config.end_scene
    if end_scene:
        sample_scene_dir = sample_scene_dir[start_scene:end_scene]
    print("PROCESS_SCENE_NAME:",sample_scene_dir)
    for it in range(int(num_gz/process_num)):
        for scene in sample_scene_dir:
            gz_start = process_num*it
            zip_files = [(f"data_{i}.json.gz", int(i % gpu_num)) for i in range(gz_start_gz+gz_start,gz_start_gz+gz_start+process_num)]
            time.sleep(1.0)
            print(f"START--{scene}--{it*process_num}/{num_gz}--dir:{base_dir_name}")
            start_time = time.time()
            with multiprocessing.Pool(processes=process_num) as pool:
                args = [(file_path, skip_len, base_dir_path, gpu_id, scene, scene_dataset_dir) for file_path, gpu_id in zip_files]
                results = []
                async_results = [pool.apply_async(run_script, (arg,), error_callback=lambda e: print(f"Error: {e}")) for arg in args]
                timer = Timer(dp_config.timeout, terminate_pool, [pool])
                timer.start()
                try:
                    pool.close()
                    pool.join()
                finally:
                    timer.cancel()
            end_time = time.time()
            print(f"FINISH--{scene}--id:{sample_scene_dir.index(scene)}--{it*process_num}/{num_gz}----dir:{base_dir_name}--usetime:{end_time-start_time}")
            