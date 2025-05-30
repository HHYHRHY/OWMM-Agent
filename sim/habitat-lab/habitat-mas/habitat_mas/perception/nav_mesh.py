import numpy as np
import open3d as o3d
from typing import Dict, List
from habitat_mas.perception.mesh_utils import (
    compute_triangle_adjacency, 
    propagate_triangle_region_ids
)

class NavMesh:
    """Wrapper for navmesh stored as triangle meshes and helper function"""
    
    def __init__(self, vertices, triangles, **kwargs):
        self.vertices = vertices
        self.triangles = triangles.reshape(-1, 3)
        
        # build the triangle mesh
        self._build_triangle_mesh(vertices, self.triangles)
        
        # compute triangle adjacency list
        self._triangle_adjacency_list = compute_triangle_adjacency(self.triangles)
        self.mesh.compute_adjacency_list()
        
        # create raycasting scene for localization
        self._raycasting_scene = o3d.t.geometry.RaycastingScene()
        self._mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
        self._raycasting_scene.add_triangles(self._mesh_tensor)
        
        # self._vertex_adjaceny_list = self.mesh.adjacency_list
        
        # by default, compute navmesh sgementation 
        self.slope_threshold = kwargs.get("slope_threshold", 30)
        self.is_flat_ground = self.segment_by_slope(slope_threshold=self.slope_threshold)
        self.triangle_region_ids = None
    
    @property
    def triangle_adjacency_list(self):
        return self._triangle_adjacency_list
    
    @property
    def vertex_adjacency_list(self):
        return self.mesh.adjacency_list
       
    def _build_triangle_mesh(self, vertices, triangles) -> o3d.geometry.TriangleMesh:
        """Build a triangle mesh from vertices and triangles"""
        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
        self.mesh.triangles = o3d.utility.Vector3iVector(triangles)
        
        # Remove duplicated vertices
        self.mesh = self.mesh.remove_duplicated_vertices()

        # Remove degenerated triangles
        self.mesh = self.mesh.remove_degenerate_triangles()

        # Remove duplicated triangles
        self.mesh = self.mesh.remove_duplicated_triangles()
        
        self.mesh.compute_vertex_normals()
        self.mesh.compute_triangle_normals()
        self.mesh.paint_uniform_color([0.5, 0.5, 0.5])
        
        self.vertices = np.asarray(self.mesh.vertices)
        self.triangles = np.asarray(self.mesh.triangles)
    
    def segment_by_slope(self, slope_threshold=30):
        """Segment the navmesh by slope angle"""
        
        assert self.mesh is not None, "Mesh is not initialized"
        
        # Define the up direction (assuming y-up coordinate frame)
        up = np.array([0, 1, 0])
        
        # Get the triangle normals as a NumPy array
        triangle_normals = np.asarray(self.mesh.triangle_normals)
        
        # Create a list to store the triangle colors
        triangle_colors = []
        
        # Compute the angle between the triangle normal and the up direction for all triangles
        angles = np.degrees(np.arccos(np.dot(triangle_normals, up)))

        # Check if the angles are within the flat-ground threshold
        is_flat_ground = angles < slope_threshold

        # Flat-ground surfaces (green color)
        triangle_colors = np.where(is_flat_ground[:, None], [0, 1, 0], [1, 0, 0])
        
        
        # Assign vertex colors by max voting of triangle colors
        vertex_colors = np.zeros_like(np.asarray(self.mesh.vertices))
        for i, triangle in enumerate(self.mesh.triangles):
            for j in range(3):
                vertex_colors[triangle[j]] += triangle_colors[i]
        
        # Normalize the vertex colors
        vertex_colors /= np.linalg.norm(vertex_colors, axis=1)[:, None]
        
        # Set the vertex colors
        self.mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        
        return is_flat_ground
    


    def segment_by_region_bbox(self, 
                               region_bbox_dict: Dict[int, np.ndarray],
                               upper_edge: float = -0.1, 
                               lower_edge: float = 0.1,
                            ):
        """
        Segment the triangle navigation mesh by region bounding box.
        For each region, assign vertices inside the 
        Arguments:
            region_bbox_dict: dict, region id to region bounding box mapping
            upper_edge: float, upper edge of the region bounding box
            lower_edge: float, lower edge of the region bounding box
        
        """
        # calculate triangle centers 
        triangle_centers = np.mean(self.vertices[self.triangles], axis=1)
        # default region id is -1
        self.triangle_region_ids = -np.ones(len(self.triangles), dtype=int)
        
        
        for region_id, region_bbox in region_bbox_dict.items():
            # if triangle center is inside the region bbox, assign the region id
            trimmed_region_bbox = region_bbox.copy()
            trimmed_region_bbox[0, 1] += upper_edge
            trimmed_region_bbox[1, 1] += lower_edge
            
            # check if the triangle center is inside the region bbox
            is_inside = np.logical_and(
                np.all(triangle_centers >= trimmed_region_bbox[0], axis=1),
                np.all(triangle_centers <= trimmed_region_bbox[1], axis=1)
            )
            
            # assign the region id to the triangle
            self.triangle_region_ids[is_inside] = region_id
            

    def propagate_region_ids(self):
        """Propagate region ids to all connected triangles"""
        assert self.triangle_region_ids is not None, "Region ids are not initialized"
        self.triangle_region_ids = propagate_triangle_region_ids(
            self.triangle_region_ids, self.triangle_adjacency_list
        )
        
    def find_triangle_agent_on(self, agent_pos: np.ndarray):
        """Find the triangle the agent is on"""
        # build the ray pointing down from the agent position, -y direction
        ray = o3d.core.Tensor([[*agent_pos, 0, -1, 0]], dtype=o3d.core.Dtype.Float32)
        # ['primitive_uvs', 'primitive_ids', 'geometry_ids', 'primitive_normals', 't_hit']
        ans = self._raycasting_scene.cast_rays(ray)
        triangle_ids = ans['primitive_ids'].numpy()
        
        # collect all triangles hit by the ray
        triangle_centers = np.mean(self.vertices[self.triangles], axis=1)
        
        # find the triangle below the agent
        triangle_id = np.argmin(np.linalg.norm(triangle_centers - agent_pos, axis=1)[triangle_ids])
        
        return triangle_id