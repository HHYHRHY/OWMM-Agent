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
from .scene_graph_base import SceneGraphBase
from habitat_mas.perception.grid_map import GridMap


class SceneGraphGibson(SceneGraphBase):

    def __init__(self, sim: Simulator, enable_region_layer=True) -> None:
        super().__init__()
        self.sim = sim
        # self.scene_name = scene_name

        # scene parameters
        # self.floor_heights = [0]
        self.height = sim.get_agent(0).state.position[1]
        self.meters_per_grid = 0.05
        self.object_grid_scale = 1
        self.aligned_bbox = True
        self.enable_region_layer = enable_region_layer
        
        # parse habitat.sim.SemanticScene
        self.load_gt_scene_graph()

    def load_gt_scene_graph(self):
        # 1. get boundary of the scene (one-layer) and initialize map
        self.scene_bounds = self.sim.pathfinder.get_bounds()
        # NOTE: bottom of bounding box could NOT be the floor
        # self.height = self.scene_bounds[0][1] # y-axis points upwards
        # self.height = self.floor_heights[0]  # assume one-layer scene
        self.free_space_grid = self.sim.pathfinder.get_topdown_view(
            self.meters_per_grid, self.height
        )  # binary matrix
        self.region_layer.init_map(
            self.scene_bounds, self.meters_per_grid, self.free_space_grid
        )

        self.dumy_space_grid = self.sim.pathfinder.get_topdown_view(
            self.meters_per_grid / self.object_grid_scale, self.height
        )  # binary matrix
        self.object_layer.init_map(
            self.scene_bounds,
            self.meters_per_grid / self.object_grid_scale,
            self.dumy_space_grid,
        )
        
        semantic_scene = self.sim.semantic_scene
        # 2. load region layer from habitat simulator
        if self.enable_region_layer: # matterport 3D has region annotations 
            for region in semantic_scene.regions:
            # add region node to region layer
                gt_region_id = int(region.id.split("_")[-1])
                sg_region_id = gt_region_id  # counting from 0, -1 for background
                region_bbox = np.stack(
                    [
                        region.aabb.center - region.aabb.sizes / 2,
                        region.aabb.center + region.aabb.sizes / 2,
                    ],
                    axis=0,
                )
                region_node = self.region_layer.add_region(
                    region_bbox,
                    region_id=sg_region_id,
                    class_name=region.category.name(),
                    label=region.category.index(),
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
                    
        else: # Gibson (original ver.) does not have region annotations 
            # 3. load object layer from habitat simulator
            for object_id, obj in enumerate(semantic_scene.objects):
                if obj is not None:
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
        return
