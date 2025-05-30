# Dataset Annotation

After collect raw datasets from simulator, you should annotate this dataset by the following command:

```bash
python annotation.py --dataset_name DATASET_demo --gz_dir_name hssd_scene_ycb_trainset
```

This script will create an annotated dataset `sat_...`. Follow the instructions in [sim README](../sim/README.md) to collect scene graph images.
