from functools import partial
import os
import sys
import time
from typing import List
import pathlib
import numpy as np
import pickle
import open3d as o3d
import copy 
import plotly.graph_objects as go
from scipy import stats
from habitat_sim import Simulator
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat_sim.agent import AgentState
from habitat.articulated_agents.mobile_manipulator import MobileManipulator

from habitat_mas.utils.constants import coco_categories, coco_label_mapping
from habitat_mas.scene_graph.scene_graph_base import SceneGraphBase
from habitat_mas.scene_graph.utils import (
    visualize_scene_graph,
    generate_region_adjacency_description,
    generate_region_objects_description,
    generate_agents_description
)
from habitat_mas.perception.grid_map import GridMap
from habitat_mas.perception.nav_mesh import NavMesh
class SceneGraphMP3D(SceneGraphBase):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.sim: RearrangeSim = None
        self.gt_point_cloud = None
        
        self.meters_per_grid = kwargs.get('meters_per_grid', 0.05)
        self.object_grid_scale = kwargs.get('object_grid_scale', 1)
        self.aligned_bbox = kwargs.get('aligned_bbox', True)
        self.enable_region_layer = kwargs.get('enable_region_layer', True)
        # Util habitat-sim v0.3.1, HM3D region annotation missing
        # self.compute_region_bbox = kwargs.get('compute_region_bbox', True)

    def load_gt_scene_graph(self, sim: RearrangeSim):
        # register habitat simulator
        self.sim = sim
        # get boundary of the scene (one-layer) and initialize map
        self.scene_bounds = self.sim.pathfinder.get_bounds()
        self.current_height = sim.get_agent(0).state.position[1]

        # 1. load navmesh from habitat simulator
        navmesh_vertices: List[np.ndarray] = self.sim.pathfinder.build_navmesh_vertices()
        navmesh_indices: List[int]= self.sim.pathfinder.build_navmesh_vertex_indices()
        self.nav_mesh = NavMesh(
            vertices=np.stack(navmesh_vertices, axis=0),
            triangles=np.array(navmesh_indices).reshape(-1, 3),
        )
        
        rom = self.sim.get_rigid_object_manager()
        all_object_handles = rom.get_object_handles()

        semantic_scene = self.sim.semantic_scene
        # 2. load region layer from habitat simulator
        if self.enable_region_layer: # matterport 3D has region annotations 
            for region in semantic_scene.regions:
            # add region node to region layer
                region_id = int(region.id.split("_")[-1]) # counting from 0, -1 for background
                if region_id < 0:
                    continue

                region_bbox = np.stack(
                    [
                        region.aabb.center - region.aabb.sizes / 2,
                        region.aabb.center + region.aabb.sizes / 2,
                    ],
                    axis=0,
                )
                
                # Add region node to region layer 
                region_class_name = None
                region_label = None
                if region.category is not None:
                    region_class_name = region.category.name()
                    region_label = region.category.index()
                
                parent_level = region.level.id
                level_height = region.level.aabb.center[1]

                region_node = self.region_layer.add_region(
                    region_bbox,
                    region_id=region_id,
                    class_name=region_class_name,
                    label=region_label,
                    parent_level=parent_level
                )

                # 3. load object layer from habitat simulator
                for object_handle, entity_name in self.sim._handle_to_goal_name.items():
                    assert object_handle in all_object_handles
                    obj = rom.get_object_by_handle(object_handle)
                    abs_obj_id = obj.semantic_id
                    pos = obj.translation
                    pos = np.array([pos.x, pos.y, pos.z])

                    rot = obj.rotation

                    is_inside = np.all(pos >= region_bbox[0]) and np.all(
                        pos <= region_bbox[1]
                    )
                    if is_inside and abs_obj_id not in self.object_layer.obj_ids:
                        snap_pos = sim.safe_snap_point(pos)
                        if np.isnan(snap_pos[0]):
                            height_to_floor = pos[1] - level_height
                        else:
                            height_to_floor = pos[1] - snap_pos[1]
                        if height_to_floor < 0:
                            height_to_floor = 0
                        object_node = self.object_layer.add_object(
                            center=pos,
                            rotation=rot,
                            id=abs_obj_id,
                            full_name=object_handle,
                            label=entity_name,
                            parent_region=region_node,
                            height=height_to_floor
                        )
                        region_node.add_object(object_node)
                
                target_trans = self.sim._get_target_trans()
                targets= {}
                for target_id, trans in target_trans:
                    targets[target_id] = trans

                if self.sim.ep_info.goal_receptacles and len(self.sim.ep_info.goal_receptacles):
                    for target_id, goal_recep in enumerate(self.sim.ep_info.goal_receptacles):
                        goal_recep_handle = goal_recep[0]
                        assert goal_recep_handle in all_object_handles
                        obj = rom.get_object_by_handle(goal_recep_handle)
                        abs_obj_id = obj.semantic_id
                        
                        target = targets[target_id]
                        target_pos = np.array(target.translation)
                        target_rot = target.rotation

                        is_inside = np.all(target_pos >= region_bbox[0]) and np.all(
                            target_pos <= region_bbox[1]
                        )
                        if is_inside and abs_obj_id not in self.object_layer.obj_ids:
                            snap_pos = sim.safe_snap_point(target_pos)
                            if np.isnan(snap_pos[0]):
                                height_to_floor = target_pos[1] - level_height
                            else:
                                height_to_floor = target_pos[1] - snap_pos[1]
                            object_node = self.object_layer.add_object(
                                center=target_pos,
                                rotation=target_rot,
                                id=abs_obj_id,
                                full_name=goal_recep_handle,
                                label=f"TARGET_any_targets|{target_id}",
                                parent_region=region_node,
                                height=height_to_floor
                            )
                            region_node.add_object(object_node)
                    
            # 4. build region triangle adjacency graph
            # algorithm to build abstract scene graph:
            # 1) segment navmesh with region bbox; 2) propogate non-labeled triangles 3) build adjaceny graph
            self.nav_mesh.segment_by_region_bbox(
                {region.region_id: region.bbox for region in self.regions}
            )
            self.nav_mesh.propagate_region_ids()
            self.region_layer.add_region_adjacency_edges(
                self.nav_mesh.triangle_region_ids, self.nav_mesh.triangle_adjacency_list
            )
            
        # TODO: load agent externally from habitat-lab API
        # 5. add agent layer to scene graph 
        for i, agent_name in enumerate(sim.agents_mgr.agents_order):
            agent: MobileManipulator = sim.agents_mgr[i].articulated_agent
            agent_name = agent_name
            
            # Assuming agent_layer.add_agent is modified to accept more parameters
            self.agent_layer.add_agent(
                agent_id=i, 
                agent_name=agent_name,
                position=agent.base_pos,
                orientation=agent.base_rot,
                # Add more parameters here
                description=sim.agents_mgr[i].cfg['articulated_agent_type']
            )
                    
    def load_gt_geometry(self):
        # load the ply file of the scene
        scene_id = self.sim.config.sim_cfg.scene_id
        ply_file_path = scene_id[:-4] + "_semantic.ply"
        
        # load the ply file with open3d 
        if os.path.exists(ply_file_path):
            self.gt_point_cloud = o3d.io.read_point_cloud(ply_file_path)
        else:
            print(f"PLY file {ply_file_path} not found")


if __name__ == "__main__":
    import habitat_sim
    import os 

    data_dir = "/home/junting/repo/habitat-lab/data"

    # initialize habitat sim
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = f"{data_dir}/scene_datasets/mp3d/1LXtFkjw3qL/1LXtFkjw3qL.glb"
    backend_cfg.scene_dataset_config_file = f"{data_dir}/scene_datasets/mp3d/mp3d.scene_dataset_config.json"

    sem_cfg = habitat_sim.CameraSensorSpec()
    sem_cfg.uuid = "semantic"
    sem_cfg.sensor_type = habitat_sim.SensorType.SEMANTIC

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sem_cfg]

    sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(sim_cfg)

    # initialize scene graph
    sg = SceneGraphMP3D()
    sg.load_gt_scene_graph(sim)
    sg.load_gt_geometry()

    # save the navmesh triangle mesh and region bounding box dict to files
    # o3d.io.write_triangle_mesh("1LXtFkjw3qL.obj", sg.nav_mesh.mesh)
    # pickle.dump(region_bbox_dict, open("1LXtFkjw3qL_region_bbox_dict.pkl", "wb"))

    sim.close()
    
    ############# Visualization ##################
    
    # visualize_scene_graph(
    #     scene_graph=sg,
    #     scene_o3d=sg.gt_point_cloud,
    #     vis_region_bbox=True,
    #     vis_object_bbox=False,
    #     vis_navmesh=True,
    #     navmesh_shift=[0, 0, -8.0],
    #     vis_region_graph=True,
    #     region_graph_shift=[0, 0, 10.0],
    #     mp3d_coord=True
    # )

    ############ Generate scene description ###########

    # Generate scene descriptions
    region_scene_graph_description = generate_region_adjacency_description(sg.region_layer)
    region_description = generate_region_objects_description(sg.region_layer, region_id=0)
    agent_description = generate_agents_description(sg.agent_layer, sg.region_layer, sg.nav_mesh)
        
    print(region_scene_graph_description)
    print("\n")
    print(region_description)
    print("\n")
    print(agent_description)