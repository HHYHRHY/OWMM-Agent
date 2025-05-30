import os
import math
import json
import random
import numpy as np
import magnum as mn
import os.path as osp
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict
from typing import(
    Dict, Generator, List, Optional, Sequence, Tuple, Union, Any
)

import habitat_sim
from habitat_sim.nav import NavMeshSettings
from habitat.core.logging import logger
from habitat.utils.common import cull_string_list_by_substrings
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.datasets.rearrange.run_episode_generator import get_config_defaults
from habitat.datasets.rearrange.rearrange_dataset import RearrangeEpisode, RearrangeDatasetV0
from habitat.datasets.rearrange.samplers.receptacle import Receptacle, AABBReceptacle, parse_receptacles_from_user_config
from habitat.datasets.rearrange.navmesh_utils import (
    get_largest_island_index,
    is_accessible,
)

def eucilidean_distance(pos1, pos2):
    return np.linalg.norm(pos1 - pos2)

# function in HabitatSim class for finding geodesic distance
def geodesic_distance(
        sim: RearrangeSim,
        position_a: Union[Sequence[float], np.ndarray],
        position_b: Union[Sequence[float], np.ndarray],
    ) -> float:
        path = habitat_sim.ShortestPath()
        path.requested_end = np.array(position_b, dtype=np.float32).reshape(3,1)
        path.requested_start = np.array(position_a, dtype=np.float32)
        found_path = sim.pathfinder.find_path(path)

        return found_path, path.geodesic_distance

def is_compatible_episode(
    s: Sequence[float],
    t: Sequence[float],
    sim: RearrangeSim,
    near_dist: float,
    far_dist: float,
    min_height_dist: float,
    same_floor: bool = False,
) -> Union[Tuple[bool, float], Tuple[bool, int]]:

    # In mobility task, s and t may not be on the same floor, 
    # we need to check height difference
    height_dist = np.abs(s[1] - t[1])
    found_path, d_separation = geodesic_distance(sim, s, t)
    if not found_path:
        return False, 0, height_dist
    if not near_dist <= d_separation <= far_dist:
        return False, 0, height_dist
    if not same_floor and not height_dist >= min_height_dist:
        return False, 0, height_dist
    elif same_floor and height_dist > 0.5:
        return False, 0, height_dist

    return True, float(d_separation), float(height_dist)

class MP3DGenerator:
    """
    Generator for MP3D scene dataset employed in RearrangeSim.
    """
     
    def __enter__(self) -> "MP3DGenerator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
         if self.sim != None:
            self.sim.close(destroy=True)
            del self.sim
    
    def __init__(
            self,
            cfg: DictConfig,
            num_episodes: int = 1,
            scene_dataset_path: str = "data/scene_datasets/mp3d/",
            num_objects: int = 1,
            num_object_filter_height: int = 0,
            min_height: float = None,
            max_height: float = None,

        ) -> None:
        self.cfg = cfg
        self.num_episodes = num_episodes
        self.dataset_path = scene_dataset_path
        self.num_objects = num_objects
        self.set_handle_list = []

        min_filter_list = [0.2] * num_objects
        max_filter_list = [1.9] * num_objects
        same_floor_list = [False] * num_objects
        max_filter = random.sample(range(num_objects), num_object_filter_height)

        if num_objects > 2:
            same_floor_list[0], same_floor_list[1] = True, True
            min_filter_list[0], min_filter_list[1] = min_height, min_height
        elif num_objects == 2:
            same_floor_list[0] = True
            min_filter_list[0] = min_height

        for i in max_filter:
            max_filter_list[i] = max_height
        self.min_filter_list = min_filter_list
        self.max_filter_list = max_filter_list
        self.same_floor_list = same_floor_list

        self.sim: RearrangeSim = None
        self.failed_episodes = 0
        self.success_episodes = 0
        self.sub_scene_paths = [d for d in os.listdir(scene_dataset_path) if osp.isdir(osp.join(scene_dataset_path, d))]
        assert self.sub_scene_paths and len(self.sub_scene_paths) > 1

        self.multi_floor_scenes = []
        self.initialize_sim(self.dataset_path)

        if self.cfg.object_sets and self.cfg.receptacle_sets:
            object_set, receptacle_set = (
            self.cfg.object_sets[0].included_substrings, 
            self.cfg.receptacle_sets[0].included_object_substrings
            )

            self.object_set = cull_string_list_by_substrings(
            full_list=self.sim.get_object_template_manager().get_template_handles(),
            included_substrings=object_set,
            excluded_substrings=[]
            )
            self.receptacle_set = cull_string_list_by_substrings(
                full_list=self.sim.get_object_template_manager().get_template_handles(),
                included_substrings=receptacle_set,
                excluded_substrings=[]
            )
        else:
            self.object_set = []
            self.receptacle_set = []

        if self.num_objects > len(self.object_set):
            raise ValueError("Object number exceeds object set size.")


    def initialize_sim(self, dataset_path: str = "data/scene_datasets/mp3d/") -> str:
        """
        Initialize the Simulator for MP3D scene dataset.
        """
        
        backend_cfg = habitat_sim.SimulatorConfiguration()
        selected_sub_scene = random.choice(self.sub_scene_paths)
        scene_glb_path = osp.join(dataset_path, selected_sub_scene, f"{selected_sub_scene}.glb")

        if not osp.exists(scene_glb_path):
            return None
        
        print("===================================================================")
        print(f"Loading {scene_glb_path} as scene.")
        print("===================================================================")

        backend_cfg.scene_id = scene_glb_path
        backend_cfg.scene_dataset_config_file = self.cfg.dataset_path
        backend_cfg.gpu_device_id = self.cfg.gpu_device_id
        backend_cfg.enable_physics = True
        backend_cfg.load_semantic_mesh = True

        sensor_specs = []
        
        color_sensor_spec = habitat_sim.CameraSensorSpec()
        color_sensor_spec.uuid = "color_sensor"
        color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        sensor_specs.append(color_sensor_spec)
        
        sem_cfg = habitat_sim.CameraSensorSpec()
        sem_cfg.uuid = "semantic"
        sem_cfg.sensor_type = habitat_sim.SensorType.SEMANTIC
        sensor_specs.append(sem_cfg)

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs

        sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        if self.sim is None:
            self.sim = habitat_sim.Simulator(sim_cfg)
            object_attr_mgr = self.sim.get_object_template_manager()
            for object_path in self.cfg.additional_object_paths:
                object_attr_mgr.load_configs(osp.abspath(object_path))
        else:
            if self.sim.config.sim_cfg.scene_id != scene_glb_path:
                self.sim.close(destroy=True)
            if self.sim.config.sim_cfg.scene_id == scene_glb_path:
                # we need to force a reset, so reload the NONE scene
                proxy_backend_cfg = habitat_sim.SimulatorConfiguration()
                proxy_backend_cfg.scene_id = "NONE"
                proxy_backend_cfg.gpu_device_id = self.cfg.gpu_device_id
                proxy_hab_cfg = habitat_sim.Configuration(
                    proxy_backend_cfg, [agent_cfg]
                )
                self.sim.reconfigure(proxy_hab_cfg)
            self.sim.reconfigure(sim_cfg)

        return scene_glb_path

    def safe_snap_point(self, point, island_idx):
        new_pos = self.sim.pathfinder.snap_point(
            point, island_idx
        )

        num_sample_points = 2000
        max_iter = 10
        offset_distance = 0.5
        distance_per_iter = 0.5
        regen_i = 0

        while np.isnan(new_pos[0]) and regen_i < max_iter:
            new_pos = self.sim.pathfinder.get_random_navigable_point_near(
                point,
                offset_distance + regen_i * distance_per_iter,
                num_sample_points,
                island_index=island_idx,
            )
            regen_i += 1

        return new_pos

    def choose_file(self, source_set: list, target_set: list, unique: bool = True) -> list:
        assert isinstance(source_set, list) and isinstance(target_set, list)
        assert len(source_set)
        if unique:
            unique_files = list(set(source_set) - set(target_set))
            return random.sample(unique_files, 1)[0]
        else:
            return random.choice(source_set)

    def sample_files(self):

        object_files, target_receptacle_files, goal_receptacle_files = [], [], []

        for i in range(num_objects):
            object_file = self.choose_file(self.object_set, object_files)
            object_files.append(object_file)

            goal_receptacle_file = self.choose_file(
                self.receptacle_set, 
                goal_receptacle_files,
                False
            )
            goal_receptacle_files.append(goal_receptacle_file)

            target_receptacle_file = self.choose_file(
                self.receptacle_set, 
                target_receptacle_files,
                False
            )
            target_receptacle_files.append(target_receptacle_file)

        return object_files, target_receptacle_files, goal_receptacle_files
    
    def is_object_accessible(self, receptacle, min_height = None, max_height = None):
        # get all the aabb_receptacles of receptacle
        aabbs = parse_receptacles_from_user_config(
            receptacle.user_attributes,
            parent_object_handle=receptacle.handle,
            parent_template_directory=receptacle.creation_attributes.file_directory,
        )
        is_access = False

        for _ in range(100):
            aabb = random.choice(aabbs)
            rec_up_global = (
                aabb.get_global_transform(self.sim)
                .transform_vector(aabb.up)
                .normalized()
            )  
            # place the object to aabb sample point
            target_obj_pos = (
                aabb.sample_uniform_global(
                    self.sim, 1.0
                )
                + 0.08 * rec_up_global
            )    
            is_access = is_accessible(
                sim=self.sim,
                point=target_obj_pos,
                height=1.0,
                nav_to_min_distance=1.5,
                nav_island=self.largest_island_idx,
                target_object_id=receptacle.object_id
            )

            snap_pos = self.safe_snap_point(target_obj_pos, self.largest_island_idx)
            if not np.isnan(snap_pos[0]):
                height_to_floor = target_obj_pos[1] - snap_pos[1]
                if min_height and height_to_floor < min_height:
                    is_access = False
                if max_height and height_to_floor > max_height:
                    is_access = False

            if is_access:
                break
        
        if not is_access:
            return None, None
        else:
            return target_obj_pos, aabb
    
    def set_receptacle_to_pos(self, receptacle_template, pos, rom, otm):
        assert otm.get_library_has_handle(receptacle_template)
        receptacle = rom.add_object_by_template_handle(
            receptacle_template
        )
        # the height offset = receptacle COM y - receptacle AABB min y
        height_offset = receptacle.com.y - receptacle.collision_shape_aabb.min.y
        # the sample height of the receptacle is of its COM, we should add its own height
        receptacle.translation = np.array([
            pos[0], 
            pos[1] + height_offset, 
            pos[2]
        ])
        receptacle.rotation = mn.Quaternion.rotation(
            mn.Rad(math.pi), mn.Vector3(0, 1, 0)
        )

        return receptacle

    def clear_objects(self, rom):
        obj_handles = rom.get_object_handles()
        new_handles = list(set(obj_handles) - set(self.set_handle_list))
        for obj_handle in new_handles:
            rom.remove_object_by_handle(obj_handle)

    def set_filter_positions(
            self, 
            pf,
            target_receptacle_file, 
            goal_receptacle_file, 
            object_file,
            rigid_objects,
            target_receptacles,
            goal_receptacles,
            targets,
            name_to_receptacle,
            num_max_tries: int = 200,
            closest_dist_limit: float = 3.0,
            furthest_dist_limit: float = 100.0,
            min_height: float = None,
            max_height: float = None,
            same_floor: bool = False,
        ):

        # 1. Get target and goal positions for receptacles
        num_tries = 0
        while num_tries < num_max_tries:
            target_pos = pf.get_random_navigable_point(
                island_index=self.largest_island_idx
            )
            target_pos = self.safe_snap_point(target_pos, self.largest_island_idx)
            if np.isnan(target_pos[0]):
                continue
            goal_pos = pf.get_random_navigable_point(
                island_index=self.largest_island_idx
            )
            goal_pos = self.safe_snap_point(goal_pos, self.largest_island_idx)
            if np.isnan(goal_pos[0]):
                continue

            # check new pos dist from existed receptacles
            if len(target_receptacles):
                is_valid = True
                for _, _, rec_pos in target_receptacles + goal_receptacles:
                    if eucilidean_distance(target_pos, rec_pos) < 1.0 or \
                        eucilidean_distance(goal_pos, rec_pos) < 1.0:
                        is_valid = False
                        break
                
                if not is_valid:
                    num_tries += 1
                    continue

            is_compatible, geo_dist, height_dist = is_compatible_episode(
                s=goal_pos,
                t=target_pos,
                sim=self.sim,
                near_dist=closest_dist_limit,
                far_dist=furthest_dist_limit,
                min_height_dist=2.5,
                same_floor=same_floor,
            )
            if not is_compatible:
                num_tries += 1
                continue
            break

        if num_tries >= num_max_tries:
            return False, [], [], [], {}, {}
        
        rom = self.sim.get_rigid_object_manager()
        otm = self.sim.get_object_template_manager()
        new_handles = []

        # 2. Set the target receptacle to target_pos
        target_recep = self.set_receptacle_to_pos(
            target_receptacle_file, target_pos, rom, otm
        )
        new_handles.append(target_recep.handle)

        # 3. Check if the object is accessible from target receptacle
        assert otm.get_library_has_handle(object_file)
        obj = rom.add_object_by_template_handle(object_file)
        new_handles.append(obj.handle)
        obj_target_pos, target_aabb = self.is_object_accessible(target_recep, min_height, max_height)
        if not obj_target_pos:
            # we need clear all the objects in the scene
            self.clear_objects(rom)
            return False, [], [], [], {}, {}
        
        # 4. Set the object to obj_target_pos
        obj.translation = obj_target_pos
        obj.rotation = mn.Quaternion.rotation(
            mn.Rad(random.uniform(0, math.pi * 2.0)), mn.Vector3(0, 1, 0)
        )
        # set rigid objects
        object_file_name = object_file.split("/")[-1]
        object_matrix = np.array([[obj.transformation[i][j] for j in range(4)] for i in range(4)]).T
        rigid_objects.append((object_file_name, object_matrix))

        # 5. Set the goal receptacle to goal_pos
        goal_recep = self.set_receptacle_to_pos(
            goal_receptacle_file, goal_pos, rom, otm
        )
        new_handles.append(goal_recep.handle)

        # 6. Remove the object and check if it is accessible from goal receptacle
        rom.remove_object_by_id(obj.object_id)
        obj = rom.add_object_by_template_handle(object_file)
        obj_goal_pos, goal_aabb = self.is_object_accessible(goal_recep, min_height, max_height)
        if not obj_goal_pos:
            self.clear_objects(rom)
            return False, [], [], [], {}, {}

        obj.translation = obj_goal_pos
        obj.rotation = mn.Quaternion.rotation(
            mn.Rad(random.uniform(0, math.pi * 2.0)), mn.Vector3(0, 1, 0)
        )
        # set targets
        targets[obj.handle] = np.array([[obj.transformation[i][j] for j in range(4)] for i in range(4)]).T

        # set target/goal receptacles
        new_tar = (
            target_recep.handle,
            np.array([[target_recep.transformation[i][j] for j in range(4)] for i in range(4)]).T,
            [x for x in target_recep.translation]
        )
        target_receptacles.append(new_tar)
        new_goa = (
            goal_recep.handle,
            np.array([[goal_recep.transformation[i][j] for j in range(4)] for i in range(4)]).T,
            [x for x in goal_recep.translation]
        )
        goal_receptacles.append(new_goa)

        # set name_to_receptacle
        name_to_receptacle[obj.handle] = target_aabb.unique_name
        
        self.set_handle_list.extend(new_handles)

        return True, rigid_objects, target_receptacles, goal_receptacles, targets, name_to_receptacle

    def generate_single_episode(
        self,
        episode_id: int,
        num_max_tries: int = 100,
    ) -> Optional[RearrangeEpisode]:
        
        ep_scene_handle = self.initialize_sim(self.dataset_path)
        if not ep_scene_handle:
            return None

        scene_base_dir = osp.dirname(osp.dirname(ep_scene_handle))
        scene_name = ep_scene_handle.split(".")[0].split("/")[-1]
        navmesh_path = osp.join(
            scene_base_dir, scene_name, scene_name + ".navmesh"
        )
        assert osp.exists(navmesh_path), f"Navmesh does not exist at {navmesh_path}"
        
        self.sim.pathfinder.load_nav_mesh(navmesh_path)
        print("====================================================================")
        print(f"Loaded navmesh from {navmesh_path}")
        print("====================================================================")

        semantic_scene = self.sim.semantic_scene
        if len(semantic_scene.levels) < 2:
            return None
        if scene_name not in self.multi_floor_scenes:
            self.multi_floor_scenes.append(scene_name)

        self.largest_island_idx = get_largest_island_index(
            self.sim.pathfinder, self.sim, allow_outdoor=False
        )

        rom = self.sim.get_rigid_object_manager()
        sample_tries = 0
        while sample_tries <= 10:
            (
                object_files,
                target_receptacle_files,
                goal_receptacle_files,
            ) = self.sample_files()
            rigid_objects, target_receptacles, goal_receptacles = [], [], []
            targets, name_to_receptacle = {}, {}
            all_is_set = True

            for i in range(self.num_objects):
                one_is_set = False
                one_object_tries = 0
                while not one_is_set:
                    (
                        one_is_set,
                        rigid_objects,
                        target_receptacles,
                        goal_receptacles,
                        targets,
                        name_to_receptacle
                    ) = self.set_filter_positions(
                        pf=self.sim.pathfinder,
                        target_receptacle_file=target_receptacle_files[i],
                        goal_receptacle_file=goal_receptacle_files[i],
                        object_file=object_files[i],
                        rigid_objects=rigid_objects,
                        target_receptacles=target_receptacles,
                        goal_receptacles=goal_receptacles,
                        targets=targets,
                        name_to_receptacle=name_to_receptacle,
                        min_height=self.min_filter_list[i],
                        max_height=self.max_filter_list[i],
                        same_floor=self.same_floor_list[i],
                    )
                    one_object_tries += 1
                    if one_object_tries >= 10:
                        break
                all_is_set &= one_is_set

            if all_is_set:
                break
            self.set_handle_list = []
            self.clear_objects(rom)
            sample_tries += 1

        if not all_is_set or len(rigid_objects) != self.num_objects:
            return None
        
        object_labels = {}
        for obj_id, obj_name in enumerate(targets):
            object_labels[obj_name] = f"any_targets|{obj_id}"

        return RearrangeEpisode(
            episode_id=episode_id,
            additional_obj_config_paths=self.cfg.additional_object_paths,
            start_position=[0, 0, 0],
            start_rotation=[0, 0, 0, 1],
            scene_id=ep_scene_handle,
            rigid_objs=rigid_objects,
            ao_states={},
            target_receptacles=target_receptacles,
            targets=targets,
            goal_receptacles=goal_receptacles,
            name_to_receptacle=name_to_receptacle,
            markers={},
            info={
                "object_labels": object_labels,
                "dataset": "mp3d",
            },
        )

    def generate_mobility_episodes(self, output_dir):

        logger.info(f"\n\nConfig:\n{self.cfg}\n\n")
        dataset = RearrangeDatasetV0()
        episodes = []
        with tqdm(total=self.num_episodes) as pbar:
            while len(episodes) < self.num_episodes:
                episode = self.generate_single_episode(episode_id=len(episodes))
                if episode:
                    episodes.append(episode)
                    pbar.update(1)
                    self.success_episodes += 1
                else:
                    self.failed_episodes +=1
                print("=================Generation Progress==================")
                print(f"Success: {self.success_episodes} | Failed: {self.failed_episodes} | Total: {self.success_episodes + self.failed_episodes}")
                print("======================================================")

        dataset.episodes += episodes
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "mp3d_rearrange.json.gz")
        
        if output_path is None:
            output_path = "mobility_episodes.json.gz"
        elif osp.isdir(output_path) or output_path.endswith("/"):
            output_path = (
                osp.abspath(output_path) + "/mp3d_episodes.json.gz"
            )
        else:
            if not output_path.endswith(".json.gz"):
                output_path += ".json.gz"

        if (
            not osp.exists(osp.dirname(output_path))
            and len(osp.dirname(output_path)) > 0
        ):
            os.makedirs(osp.dirname(output_path))
        import gzip

        with gzip.open(output_path, "wt") as f:
            f.write(dataset.to_json())
        
        print("======================================================")
        print("Total multi-floor scenes: ", len(self.multi_floor_scenes))
        print("======================================================")
        print(f"Episodes saved to {output_path}")     

if __name__ == "__main__":
    config_path = "habitat-lab/habitat/datasets/rearrange/configs/mp3d.yaml" 
    output_dir = "data/datasets/mp3d_new"
    scene_dataset_path = "data/scene_datasets/mp3d/"     
    num_episodes = 1
    num_objects = 1
    num_object_filter_height = 0
    min_height = 1.7
    max_height = None

    assert num_episodes > 0, "Number of episodes must be greater than 0."
    assert osp.exists(
            config_path
        ), f"Provided config, '{config_path}', does not exist."
        
    cfg = get_config_defaults()
    override_config = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(cfg, override_config)  

    with MP3DGenerator(
        cfg=cfg,
        num_episodes=num_episodes,
        scene_dataset_path=scene_dataset_path,
        num_objects=num_objects,
        num_object_filter_height=num_object_filter_height,
        min_height=min_height,
        max_height=max_height,

    ) as mp_gen:
        mp_gen.generate_mobility_episodes(output_dir)