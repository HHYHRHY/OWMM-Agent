import os
import json
import numpy as np
from typing import List, Tuple
import habitat
from habitat.config.default import get_config
from habitat.config import read_write
from habitat.core.env import Env
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.datasets.rearrange.rearrange_dataset import (
    RearrangeDatasetV0,
    RearrangeEpisode
)
from habitat.datasets.rearrange.navmesh_utils import (
    get_largest_island_index,
)
from habitat.utils.visualizations.utils import observations_to_image


def save_image(image, file_path):
    from PIL import Image
    img = Image.fromarray(image)
    img.save(file_path)

def get_hssd_single_agent_config(cfg_path="single_rearrange/zxz_llm_fetch.yaml"):
    config = get_config(cfg_path)
    # with read_write(config):
        # config.habitat.task.lab_sensors = {}
        # agent_config = get_agent_config(config.habitat.simulator)
        # agent_config.sim_sensors = {
        #     "rgb_sensor": HabitatSimRGBSensorConfig(),
        #     "depth_sensor": HabitatSimDepthSensorConfig(),
        # }
    return config

# Generate a random quaternion (x, y, z, w) restricted to the x-z plane
def random_quaternion_xz_plane():
    # Restrict the orientation to the x-z plane (no y component)
    theta = np.random.uniform(0, 2 * np.pi)  # random angle in the x-z plane
    qx = np.sin(theta / 2)
    qy = 0  # No rotation around the y-axis
    qz = np.cos(theta / 2)
    qw = 0  # Unit quaternion constraint
    
    return [qx, qy, qz, qw]

def calculate_orientation_xz_plane(position, goal_position):
    """
    Calculate the orientation of the agent facing towards the goal position.
    """
    dx = goal_position[0] - position[0]
    dz = goal_position[2] - position[2]
    
    # Calculate the angle between the x-axis and the vector from the agent's position to the goal position
    theta = np.arctan2(dz, dx)
    
    # Convert the angle to a quaternion
    qx = np.sin(theta / 2)
    qy = 0  # No rotation around the y-axis
    qz = np.cos(theta / 2)
    qw = 0  # Unit quaternion constraint
    
    return [qx, qy, qz, qw]

def get_target_objects_info(sim: RearrangeSim) -> Tuple[List[int], List[np.ndarray], List[np.ndarray]]:
    """
    Get the target objects' information: object ids, start positions, and goal positions from the rearrange episode.
    """
    target_idxs = []
    target_handles = []
    start_positions = []
    goal_positions = []
    rom = sim.get_rigid_object_manager()

    for target_handle, trans in sim._targets.items():
        target_idx = sim._scene_obj_ids.index(rom.get_object_by_handle(target_handle).object_id)
        obj_id = rom.get_object_by_handle(target_handle).object_id
        start_pos = sim.get_scene_pos()[sim._scene_obj_ids.index(obj_id)]
        goal_pos = np.array(trans.translation)

        target_idxs.append(target_idx)
        target_handles.append(target_handle)
        start_positions.append(start_pos)
        goal_positions.append(goal_pos)

    return target_idxs, target_handles, start_positions, goal_positions

# def get_detected_objects(self, observations, *args, **kwargs):
#         """ Get the detected objects from the semantic sensor data """
        
#         observation_keys = list(observations.keys())
        
#         # Retrieve the semantic sensor data
#         if self.agent_id is None:
#             target_keys = [key for key in observation_keys if "semantic" in key]
#         else:
#             target_keys = [
#                 key for key in observation_keys 
#                 if f"agent_{self.agent_id}" in key and "semantic" in key
#             ]
        
#         sensor_detected_objects = {}
        
#         for key in target_keys:
#             semantic_sensor_data = observations[key]
            
#             # Count the occurrence of each object ID in the semantic sensor data
#             unique, counts = np.unique(semantic_sensor_data, return_counts=True)
#             objects_count = dict(zip(unique, counts))
            
#             # Filter objects based on the size threshold
#             sensor_detected_objects[key] = [obj_id for obj_id, count in objects_count.items() if count > self.pixel_threshold]
        
#         # Concatenate all detected objects from all sensors
#         detected_objects = np.unique(np.concatenate(list(sensor_detected_objects.values())))
        
#         return detected_objects

def main(args):
    config = get_hssd_single_agent_config(args.config)
    env = Env(config=config)
    dataset = env._dataset

    metadata = []

    for episode in dataset.episodes:
        # reset automatically calls next episode
        env.reset()
        sim: RearrangeSim = env.sim
        # get the largest island index
        largest_island_idx = get_largest_island_index(
            sim.pathfinder, sim, allow_outdoor=True
        )
        graph_positions = []
        graph_orientations = []
        
        # Get scene objects ids
        scene_obj_ids = sim.scene_obj_ids
        
        # Get target objects' information
        object_ids, object_handles, start_positions, goal_positions = get_target_objects_info(sim)
        
        # Sample navigable points near the target objects start positions and goal positions
        # And calculate the orientation of the agent facing towards the position 
        for idx, (start_pos, goal_pos) in enumerate(zip(start_positions, goal_positions)):
            # Sample navigable point near the start and goal positions
            n_trial = 0
            radius = args.dist_to_target
            while n_trial < args.max_trials: 
                start_navigable_point = sim.pathfinder.get_random_navigable_point_near(
                    circle_center=start_pos, radius=radius, island_index=largest_island_idx
                )
                goal_navigable_point = sim.pathfinder.get_random_navigable_point_near(
                    circle_center=goal_pos, radius=radius, island_index=largest_island_idx
                )
                # if start and goal navigable points are not NaN, break the loop
                if not np.isnan(start_navigable_point[0]) and not np.isnan(goal_navigable_point[0]):
                    break
                # else increase the radius limit and try again
                else:
                    radius += 0.5
                    n_trial += 1

            # Calculate the orientation of the agent facing towards the goal position
            start_orientation = calculate_orientation_xz_plane(start_navigable_point, goal_pos)
            goal_orientation = calculate_orientation_xz_plane(goal_navigable_point, start_pos)
            
            graph_positions.extend([start_navigable_point, goal_navigable_point])
            graph_orientations.extend([start_orientation, goal_orientation])
            

        # Sample navigable points if images are less than the maximum number of images
        if len(graph_positions) < args.max_images:
            for _ in range(args.max_images - len(graph_positions)):
                # Sample a random navigable point in the scene
                point = sim.pathfinder.get_random_navigable_point(island_index=largest_island_idx)
                orientation = random_quaternion_xz_plane()
                graph_positions.append(point)
                graph_orientations.append(orientation)

        # Create a folder for this episode
        episode_dir = os.path.join(args.output_dir, f"episode_{episode.episode_id}")
        if not os.path.exists(episode_dir):
            os.makedirs(episode_dir)

        for idx, (position, orientation) in enumerate(zip(graph_positions, graph_orientations)):
            observations = env.sim.get_observations_at(position=position, rotation=orientation)

            obs_file_list = []
            for obs_key in args.obs_keys:
                if obs_key in observations:
                    image = observations[obs_key]
                    image_file_name = f"episode_{episode.episode_id}_{obs_key}_{idx}.png"
                    image_file_path = os.path.join(episode_dir, image_file_name)
                    save_image(image, image_file_path)
                    obs_file_list.append(image_file_name)

            metadata.append({
                "episode_id": episode.episode_id,
                "obs_files": obs_file_list,
                "position": position,
                "rotation": orientation,
                # "detected_objects": observations["objectgoal"],
            })

    # Save metadata to JSON
    metadata_file_path = os.path.join(args.output_dir, "metadata.json")
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    with open(metadata_file_path, "w") as f:
        json.dump(metadata, f, indent=4, cls=NumpyEncoder)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="single_rearrange/zxz_llm_fetch.yaml")
    parser.add_argument("--output_dir", type=str, default="data/sparse_slam/rearrange/hssd")
    parser.add_argument("--obs_keys", nargs="+", default=["head_rgb"])
    parser.add_argument("--dist_to_target", type=float, default=1.0)
    parser.add_argument("--max_trials", type=int, default=3)
    parser.add_argument("--max_images", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--output_dir", type=str, default="data/sparse_slam/rearrange/mp3d")
    args = parser.parse_args()

    # Create output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set seed
    np.random.seed(args.seed)
    
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)