import copy

import numpy as np
import open3d as o3d
from habitat_mas.scene_graph.utils import project_points_to_grid_xz


class ObjectNode:
    def __init__(
        self,
        id,
        center,
        rotation,
        size,
        class_name=None,
        label=None,
        full_name=None,
        description=None,
        parent_region=None,
        height=None,
        vertices=None,
        colors=None,
        normals=None,
    ):

        # required attributes
        self.id = id
        self.center = center  # [x,y,z]
        self.rotation = rotation  # [x,y,z,w]
        self.size = size  # [x,y,z]

        # optinal attributes
        self.class_name = class_name
        self.label = label
        self.full_name = full_name
        self.description = description

        # edge to parent region node
        self.parent_region = parent_region
        self.height = height

        self.vertices = vertices
        self.colors = colors
        self.normals = normals

    def add_point_clouds(self, vertices, colors, normals):

        self.vertices = vertices
        self.colors = colors
        self.normals = normals


class ObjectLayer:
    def __init__(self):

        self.flag_grid_map = False
        self.obj_ids = []
        self.obj_dict = {}

    def __len__(self):
        return len(self.obj_ids)

    def init_map(self, bounds, grid_size, free_space_grid, project_mode="xz"):

        self.bounds = (
            bounds  # the real map area corresponding to free space grid
        )
        self.grid_size = grid_size
        self.segment_grid = np.zeros_like(free_space_grid, dtype=int) - 1
        self.free_space_grid = np.array(free_space_grid).astype(bool)
        self.project_mode = (
            project_mode  # 'xz' means project xz axes in 3D to xy axes in map
        )

        # implement the data structure for closest grid point searching
        # self.grid_kd_tree = cKDTree(self.free_space_grid)
        self.flag_grid_map = True
        return

    def add_object(
        self,
        center,
        rotation,
        size=None,
        id=None,
        class_name=None,
        label=None,
        full_name=None,
        description=None,
        vertices=None,
        colors=None,
        normals=None,
        bbox=None,
        parent_region=None,
        height=None,
    ):
        # add object node
        if id == None or id in self.obj_ids:
            new_id = 0
            if len(self.obj_ids) > 0:
                new_id = max(self.obj_ids) + 1
            print(
                f"Warning: object id {id} already exists in object layer. Assign id {new_id} instead"
            )
            id = new_id

        self.obj_ids.append(id)
        obj_node = ObjectNode(
            id, center, rotation, size, 
            class_name=class_name, 
            label=label, 
            full_name=full_name, 
            description=description, 
            parent_region=parent_region,
            height=height
        )

        if vertices is not None:
            obj_node.add_point_clouds(
                vertices, colors, normals
            )

        self.obj_dict[id] = obj_node

        # add object segment on layer free space grid map
        if self.flag_grid_map and bbox is not None:
            if self.project_mode == "xz":
                obj_grid_map = self.segment_object_on_grid_map_xz(id, bbox)
            else:  # TODO: if there are other datasets ...
                raise NotImplementedError
            obj_node.grid_size = self.grid_size
            obj_node.grid_map = obj_grid_map

        return obj_node

    def segment_object_on_grid_map_xz(self, obj_id, obj_bbox):
        assert (
            self.flag_grid_map
        ), "called 'segment_region_on_grid_map_xz()' before grid map being initialized"
        # get region bbox on 2d grid map
        # object_2d_bbox: np.array: [[x_min, y_min],[x_max, y_max]]
        obj_2d_bbox = project_points_to_grid_xz(
            self.bounds, obj_bbox, self.grid_size
        )

        obj_mask = np.zeros_like(self.free_space_grid).astype(bool)
        # NOTE: (row_idx, col_idx) corresponds to (y, x) in 2d grid map
        obj_mask[
            int(np.ceil(obj_2d_bbox[0][1])) : int(np.floor(obj_2d_bbox[1][1])),
            int(np.ceil(obj_2d_bbox[0][0])) : int(np.floor(obj_2d_bbox[1][0])),
        ] = True
        # color the region on global grid map
        if (self.segment_grid[obj_mask] == -1).all():
            self.segment_grid[obj_mask] = obj_id
        else:
            print(
                f"Warning: object {obj_id} overlap with object {self.segment_grid[obj_mask]}"
            )
            self.segment_grid[obj_mask] = obj_id

        return obj_mask

    def get_objects_by_ids(self, ids):
        return [self.obj_dict[id] for id in ids]

    def get_centers(self, ids):
        centers = np.array([self.obj_dict[id].center for id in ids])
        return centers

    def get_labels(self, ids):
        labels = [self.obj_dict[id].label for id in ids]
        return labels

    def get_class_names(self, ids):
        names = [self.obj_dict[id].class_name for id in ids]
        return names
