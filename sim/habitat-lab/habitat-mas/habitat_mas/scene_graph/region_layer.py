import copy
from typing import List, Dict
import numpy as np

# from utils.open3d_utils import
# local import
from habitat_mas.scene_graph.object_layer import ObjectNode
from habitat_mas.scene_graph.utils import project_points_to_grid_xz
from scipy.spatial import cKDTree
import networkx as nx

class RegionNode:
    def __init__(
        self,
        region_id,
        bbox,
        grid_map=None,
        grid_size=0.1,
        class_name=None,
        label=None,
        parent_level=None,
    ):
        # required field: minimum attributes for a region: id, bounding box
        self.region_id = region_id
        self.bbox: np.ndarray = bbox # (2,3)

        # optional field
        self.grid_map = grid_map
        self.grid_size = grid_size
        self.class_name = class_name
        self.label = label
        self.objects = []
        self.parent_level = parent_level
        return

    def add_object(self, obj: ObjectNode):

        self.objects.append(obj)
        return


class RegionLayer(nx.Graph):
    def __init__(self):
        super().__init__()
        self.flag_grid_map = False
        self.region_ids = []
        self.region_dict: Dict[int, RegionNode] = {}
        return

    # TODO: consider creating a standalone map class?
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

    def add_region(self, bbox, region_id=None, class_name=None, label=None, parent_level=None):

        # add region node
        if region_id == None or region_id in self.region_ids:
            new_id = 0
            if len(self.region_ids) > 0:
                new_id = max(self.region_ids) + 1
            print(
                f"Warning: region id {region_id} already exists in region layer. Assign id {new_id} instead"
            )
            region_id = new_id
        self.region_ids.append(region_id)
        region_node = RegionNode(region_id, bbox, parent_level=parent_level)
        self.region_dict[region_id] = region_node
        self.add_node(region_node)

        # add segment on layer free space grid map
        if self.flag_grid_map:
            if self.project_mode == "xz":
                region_grid_map = self.segment_region_on_grid_map_xz(region_id, bbox)
            else:  # TODO: if there are other datasets ...
                raise NotImplementedError
            region_node.grid_size = self.grid_size
            region_node.grid_map = region_grid_map

        # add semantic info
        region_node.class_name = class_name
        region_node.label = label

        return region_node

    def segment_region_on_grid_map_xz(self, region_id, region_bbox):
        assert (
            self.flag_grid_map
        ), "called 'segment_region_on_grid_map_xz()' before grid map being initialized"
        # get region bbox on 2d grid map
        # region_2d_bbox: np.array: [[x_min, y_min],[x_max, y_max]]
        region_2d_bbox = project_points_to_grid_xz(
            self.bounds, region_bbox, self.grid_size
        )

        region_mask = np.zeros_like(self.free_space_grid).astype(bool)
        # NOTE: (row_idx, col_idx) corresponds to (y, x) in 2d grid map
        region_mask[
            int(np.ceil(region_2d_bbox[0][1])) : int(
                np.floor(region_2d_bbox[1][1])
            ),
            int(np.ceil(region_2d_bbox[0][0])) : int(
                np.floor(region_2d_bbox[1][0])
            ),
        ] = True
        # color the region on global grid map
        region_mask = np.logical_and(region_mask, self.free_space_grid)
        self.segment_grid[region_mask] = region_id

        return region_mask


    def add_region_adjacency_edges(self, triangle_region_ids, adjacency_list)->nx.Graph:
        """
        Build a region adjacency graph from a triangle mesh and triangle region segmentation.
        
        Parameters:
            triangle_region_ids: np.ndarray, a list of region IDs corresponding to each triangle
            adjacency_list: A list of lists, where each list contains the indices of neighboring triangles for each triangle
        """

        region_ids = np.unique(triangle_region_ids)
        # assert len(region_ids>=0) == len(self.region_ids), "Exception in add_region_adjacency_edges: Region number mismatch"

        # Check adjacency between triangles and add edges between corresponding regions
        for i, neighbors in enumerate(adjacency_list):
            current_region_id = triangle_region_ids[i]
            for neighbor in neighbors:
                neighbor_region_id = triangle_region_ids[neighbor]
                if current_region_id != neighbor_region_id and current_region_id >= 0 and neighbor_region_id >= 0:
                    current_region = self.region_dict[current_region_id]
                    neighbor_region = self.region_dict[neighbor_region_id]
                    self.add_edge(current_region, neighbor_region)
