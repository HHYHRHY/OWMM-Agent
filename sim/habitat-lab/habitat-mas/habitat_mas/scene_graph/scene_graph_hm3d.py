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
from scipy import stats
from habitat_sim import Simulator

from habitat_mas.utils.constants import coco_categories, coco_label_mapping
from habitat_mas.scene_graph.scene_graph_base import SceneGraphBase
from habitat_mas.scene_graph.utils import (
    aggregate_bboxes
)
from habitat_mas.perception.grid_map import GridMap
from habitat_mas.perception.nav_mesh import NavMesh

class SceneGraphHM3D(SceneGraphBase):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.sim: Simulator = None
        
        self.meters_per_grid = kwargs.get('meters_per_grid', 0.05)
        self.object_grid_scale = kwargs.get('object_grid_scale', 1)
        self.aligned_bbox = kwargs.get('aligned_bbox', True)
        self.enable_region_layer = kwargs.get('enable_region_layer', True)
        # Util habitat-sim v0.3.1, HM3D region annotation missing
        self.compute_region_bbox = kwargs.get('compute_region_bbox', True)

    def load_gt_scene_graph(self, sim: Simulator):
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
        
        semantic_scene = self.sim.semantic_scene
        # 2. load region layer from habitat simulator
        if self.enable_region_layer: # matterport 3D has region annotations 
            for region in semantic_scene.regions:
            # add region node to region layer
                region_id = int(region.id.split("_")[-1]) # counting from 0, -1 for background
                if region_id < 0:
                    continue
                # Compute region bbox
                if self.compute_region_bbox:
                    # compute region bbox from objects inside the region
                    region_object_bboxes = []
                    for obj in region.objects:
                        if obj is not None:
                            if obj.aabb is not None:
                                region_object_bboxes.append(
                                    np.stack(
                                        [
                                            obj.aabb.center - obj.aabb.sizes / 2,
                                            obj.aabb.center + obj.aabb.sizes / 2,
                                        ],
                                        axis=0,
                                    )
                                )
                    region_bbox = aggregate_bboxes(region_object_bboxes)
                             
                else:
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
                
                region_node = self.region_layer.add_region(
                    region_bbox,
                    region_id=region_id,
                    class_name=region_class_name,
                    label=region_label,
                )

                # 3. load object layer from habitat simulator
                for obj in region.objects:
                    if obj is not None:
                        object_id = int(obj.id)  
                        if self.aligned_bbox:
                            center = obj.aabb.center
                            rot_quat = np.array([0, 0, 0, 1])  # identity transform
                            size = obj.aabb.sizes
                        else:  # Use obb, NOTE: quaternion is [w,x,y,z] from habitat, need to convert to [x,y,z,w]
                            center = obj.obb.center
                            rot_quat = obj.obb.rotation[1, 2, 3, 0]
                            size = obj.obb.sizes
                            size = obj.aabb.sizes

                        node_size = (
                            self.meters_per_grid / self.object_grid_scale
                        )  # treat object as a point
                        node_bbox = np.stack(
                            [center - node_size / 2, center + node_size / 2], axis=0
                        )
                        object_node = self.object_layer.add_object(
                            center,
                            rot_quat,
                            size,
                            id=object_id,
                            class_name=obj.category.name(),
                            label=obj.category.index(),
                            bbox=node_bbox,
                        )

                        # connect object to region
                        region_node.add_object(object_node)
                


if __name__ == "__main__":
    import habitat_sim
    import os 
    
    data_dir = "/home/junting/repo/habitat-lab/data"
    
    # initialize habitat sim 
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = f"{data_dir}/scene_datasets/hm3d/val/00891-cvZr5TUy5C5/cvZr5TUy5C5.basis.glb"
    backend_cfg.scene_dataset_config_file = f"{data_dir}/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"

    sem_cfg = habitat_sim.CameraSensorSpec()
    sem_cfg.uuid = "semantic"
    sem_cfg.sensor_type = habitat_sim.SensorType.SEMANTIC

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sem_cfg]

    sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(sim_cfg)

    # initialize scene graph
    sg = SceneGraphHM3D()
    sg.load_gt_scene_graph(sim)
    
    # TODO: add visualization tools here: 
    o3d.visualization.draw_geometries([sg.nav_mesh.mesh], mesh_show_back_face=True)
