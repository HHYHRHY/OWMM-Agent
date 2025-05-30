# Simulator 

We provide scripts for data generation and simulator evaluation.

## Data Generation

First, you need to follow the instructions in `habitat_mas` to download the dataset.

Additionally, you need to download the episode config for data collection. We provide the complete config used in our experiments, which you can download from [there](). After downloading, place the file in the `habitat-lab/data/dataset` directory.

We provide a script for parallel data collection on multiple GPUs. To run the script, navigate to `habitat-lab` and execute the following command:

```bash
python dataset_make.py --gz_start 0 --gz_end 10 --base_directory DATASET_demo --process_num 4 --gpu_number 1 --scene_dataset_dir hssd_scene_train
```

This means the script will perform data collection using a single GPU with four processes. The episode config used is `hssd_scene_train`, and for each scene, it will collect `.gz` files numbered from 0 to 10. The collected dataset will be stored in the `DATASET_demo` folder.

## Simulator Evaluation 

TBD

