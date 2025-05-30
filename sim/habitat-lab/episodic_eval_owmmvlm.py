import subprocess
import os
import random
from tqdm import tqdm
import multiprocessing


def run_command(i, gpu_id,dataset_path,episode_id):
    """Runs the command for a single i value on a specified GPU."""
    # seed = random.randint(0, 100000)
    seed = episode_id[i]
    data_path = os.path.join(dataset_path,"image",str(episode_id[i]),"scene_graph.gz")
    output_dir_name = os.path.basename(dataset_path)
    
    image_dir = f"eval_in_sim_info/{output_dir_name}/{episode_id[i]}"
    video_dir = f"eval_in_sim_info/{output_dir_name}/{episode_id[i]}"
    output_file = f"{image_dir}/output.log"

    os.makedirs(image_dir, exist_ok=True)

    command = [
        "python", "-u", "-m", "habitat_baselines.run",
        "--config-name=single_rearrange/vlm_fetch.yaml",
        f"habitat.dataset.data_path={data_path}",
        f"habitat.seed={seed}",
        f"habitat_baselines.image_dir={image_dir}",
        f"habitat_baselines.video_dir={video_dir}",
        f"habitat.simulator.habitat_sim_v0.gpu_device_id={gpu_id}",
        f"habitat_baselines.torch_gpu_id={gpu_id}",
    ]


    with open(output_file, "w") as f:
        process = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT)
        process.communicate()
    return i


def worker(params):
    """Helper function to unpack parameters for multiprocessing."""
    return run_command(*params)

if __name__ == "__main__":
    dataset_paths = ['data/datasets/sat_TEST_YCB_30scene_head_rgb']
    gpus = [0]
    num_processes = len(gpus)
    episode_id = []
    for dataset_path in dataset_paths:
        episode_id = []
        with open(os.path.join(dataset_path,"test_episode_id.txt")) as f: #put the id of episodes that you want to evaluate in this txt
            for line in f:
                stripped_line = line.strip()
                if stripped_line.isdigit():
                    episode_id.append(int(stripped_line))
        len_test = len(episode_id)
        tasks = [(i, gpus[i % len(gpus)],dataset_path,episode_id) for i in range(len_test)]

        with multiprocessing.Pool(processes=num_processes) as pool:
            for i in tqdm(pool.imap_unordered(worker, tasks), total=len_test):
                pass
