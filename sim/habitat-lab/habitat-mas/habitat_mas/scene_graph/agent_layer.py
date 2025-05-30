import copy
from typing import List, Dict, Tuple
import numpy as np
import open3d as o3d
from habitat_mas.scene_graph.utils import project_points_to_grid_xz
from habitat_mas.perception.nav_mesh import NavMesh

class AgentNode:
    def __init__(
        self,
        agent_id,
        agent_name,
        position,
        orientation,
        # bbox,
        description="",
    ):

        # required attributes
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.position = np.array(position)  # [x,y,z]
        self.orientation = np.array(orientation)  # [x,y,z,w]
        # self.bbox = np.array(bbox)  # [x_min, y_min, z_min, x_max, y_max, z_max]  
        self.description = description
        
    def locate_region_by_position(self, navmesh: NavMesh, visualize=True):
        """Locate the region where the agent is located"""
        assert navmesh.triangle_region_ids is not None, "Exception in locate_region_by_position: Navmesh region ids are not available"
        # project the agent position to the navmesh
        agent_position = np.array(self.position)
        agent_position[1] = navmesh.vertices[:, 1].min()
        triangle_id = navmesh.find_triangle_agent_on(agent_position)
        region_id = navmesh.triangle_region_ids[triangle_id]
        
        if visualize: 
            # visualize the agent position
            agent_pos = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            agent_pos.paint_uniform_color([1, 0, 0])
            agent_pos.translate(agent_position)
            
            # visualize the navmesh with the triangle colored by blue
            triangle_mesh = o3d.geometry.TriangleMesh()
            triangle_mesh.vertices = o3d.utility.Vector3dVector(navmesh.vertices)
            triangle_mesh.triangles = o3d.utility.Vector3iVector(navmesh.triangles)
            triangle_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array([0.5, 0.5, 0.5]) * np.ones_like(navmesh.vertices))
            for i in navmesh.triangles[triangle_id]:
                triangle_mesh.vertex_colors[i] = [0, 0, 1]
             
            o3d.visualization.draw_geometries([navmesh.mesh, agent_pos])
        
        return region_id

class AgentLayer:
    def __init__(self):

        self.flag_grid_map = False
        self.agent_ids: List = []
        self.agent_dict: Dict[str, AgentNode] = {}

    def __len__(self):
        return len(self.agent_ids)

    def add_agent(
        self,
        agent_id,
        agent_name,
        position,
        orientation,
        description="",
    ):

        agent = AgentNode(agent_id, agent_name, position, orientation, description)
        self.agent_ids.append(agent_id)
        self.agent_dict[agent_id]= agent

        return agent
    

    def get_agents_region_ids(self, navmesh: NavMesh)->Dict[int, int]:
        """Get all agents' region ids"""
        agents_region_ids = {}
        for agent_id, agent in self.agent_dict.items():
            region_id = agent.locate_region_by_position(navmesh)
            agents_region_ids[agent_id] = region_id
        return agents_region_ids