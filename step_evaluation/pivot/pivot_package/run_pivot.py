import numpy as np
from pivot_package.pivot_runner import pivot_runner
from pivot_package.vlms import GPT4V
import scipy

# Adjust radius of annotations based on size of the image
radius_per_pixel = 0.025


def run_pivot(
    im,
    query,
    n_samples_init,
    n_samples_opt,
    n_iters,
    n_parallel_trials,
    openai_api_key,
    openai_base_url,
):
    if not openai_api_key:
        print('Must provide OpenAI API Key')
        return []
    if im is None:
        print('Must specify image')
        return []
    if not query:
        print('Must specify description')
        return []

    # Set hyper parameters
    img_size = np.min(im.shape[:2])
    style = {
        'num_samples': 12,
        'circle_alpha': 0.6,
        'alpha': 0.8,
        'arrow_alpha': 0.0,
        'radius': int(img_size * radius_per_pixel),
        'thickness': 2,
        'fontsize': int(img_size * radius_per_pixel),
        'rgb_scale': 255,
        'focal_offset': 1,  # camera distance / std of action in z
    }
    action_spec = {
        'loc': [256, 256],
        'scale': [100, 100],
        'min_scale': [0.0, 0.0],
        'min': [0, 0],
        'max': [511,511],
        'action_to_coord': 250,
        'robot': None,
    }

    vlm = GPT4V(openai_api_key=openai_api_key,openai_base_url=openai_base_url)

    return pivot_runner(
        vlm,
        im,
        query,
        style,
        action_spec,
        n_samples_init=n_samples_init,
        n_samples_opt=n_samples_opt,
        n_iters=n_iters,
        n_parallel_trials=n_parallel_trials,
    )
