# Simulator 

We provide scripts for data generation and simulator evaluation.

## Data Generation

First, you need to follow the instructions in `habitat_mas` to download the dataset.

Additionally, you need to download the episode config for data collection. We provide the complete config used in our experiments, which you can download from [there](https://drive.google.com/drive/folders/1FIvCu28nRcz_tSsMLbii3pigFwLtwUQU?usp=sharing). After downloading, place the file in the `habitat-lab/data/dataset` directory like the following structure:
```kotlin
data/
└── dataset/
    ├── hssd_scene_ycb_trainset/
    ├── hssd_scene_google_trainset/
    ├── hssd_scene_ycb_testset/
    └── hssd_scene_google_testset/

```

We provide a script for parallel data collection on multiple GPUs. To run the script, navigate to `habitat-lab` and execute the following command:

```bash
python dataset_make.py --gz_start 0 --gz_end 10 --base_directory DATASET_demo --process_num 4 --gpu_number 1 --scene_dataset_dir hssd_scene_ycb_trainset
```

This means the script will perform data collection using a single GPU with four processes. The episode config used is `hssd_scene_ycb_trainset`, and for each scene, it will collect `.gz` files numbered from 0 to 10. The collected dataset will be stored in the `DATASET_demo` folder.This folder contains raw robot dataset collected from the simulator. The raw dataset need to be annotated in the [dataset_annotation](https://github.com/HHYHRHY/OWMM-Agent/dataset_annotation). Please follow [Readme](https://github.com/HHYHRHY/OWMM-Agent/dataset_annotation/README.md) in the annotation part to complete the labeling process.

After the raw dataset has been annotated, the dataset also needs to include scene graph images randomly sampled from the simulator. Please follow the command example below to collect this data:

```bash
python scenegraph_generation.py --dataset_name sat_DATASET_demo --start_dir 0 --end_dir 100 --gpu_num 1
```

This means the script will read episodes numbered from 0 to 100 from the annotated dataset "sat_DATASET_demo", and generate 8 random scene graph images for each corresponding episode scene. After running this script, the dataset will conform to the required format for our model's trainset and testset. Additionally, you can also use GPT to rewrite the textual annotations in the dataset.


## Simulator Evaluation 

To eval our models and baseline models(PIVOT agent and Robopoint agent), you should first follow the upon instructions to make test dataset. Then you can follow the scripts`episodic_eval_gptagent.py` and `episodic_eval_owmmvlm.py` to make episodic evaluation.

You should navigate to [dummy_agent.py](habitat-lab/habitat-mas/habitat_mas/agents/dummy_agent.py) to change the client's url to match your model's deployment. Before evaluate baseline models, you should first navigate to [pivot_agent.py](habitat-lab/habitat-mas/habitat_mas/agents/pivot_agent.py) to change API_KEY and BASE_URL for gpt-4o in line 334. Change `USE_PIVOT` in line 343 to match your baseline model. If you want to evaluate Robopoint agent, you should additionally change the url to match your deployment of robopoint model in line 294.

