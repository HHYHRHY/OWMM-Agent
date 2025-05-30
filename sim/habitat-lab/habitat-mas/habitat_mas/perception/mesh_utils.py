import os
from typing import Optional, Dict, List
import numpy as np
import open3d as o3d 
from collections import deque
import networkx as nx


def compute_triangle_adjacency(triangles):
    """
    Computes the adjacency list for triangles in a mesh based on shared edges.
    
    :param triangles: A list of triangles, where each triangle is represented by a tuple or list of vertex indices.
    :return: A list of lists, where each list contains the indices of neighboring triangles for each triangle.
    """
    # TODO: Improve the algorithm for better performance
    adjacency_list = []
    for i, triangle in enumerate(triangles):
        triangle_neighbors = []
        for j, other_triangle in enumerate(triangles):
            if i != j:
                shared_vertices = set(triangle).intersection(other_triangle)
                if len(shared_vertices) == 2:
                    triangle_neighbors.append(j)
                    
        adjacency_list.append(triangle_neighbors)
    return adjacency_list

def propagate_triangle_region_ids(triangle_region_ids, adjacency_list):
    """
    Propagates region IDs from triangles with known region IDs to neighboring triangles
    that have no region ID (default -1), using a breadth-first search approach.
    
    :param triangle_region_ids: A list of region IDs corresponding to each triangle, with -1 indicating no region ID.
    :param adjacency_list: A list of lists, where each list contains the indices of neighboring triangles for each triangle.
    :return: The updated list of region IDs after propagation.
    """
    
    can_propagate = True
    while can_propagate:
        # For each iteration, we check all triangles can be propagated, and them batch assign region IDs
        propagated_triangle_ids = []
        propagated_region_ids = []
        # For each non-labeled triangle, check if it can be labeled
        for triangle_id, region_id in enumerate(triangle_region_ids):
            if region_id == -1:
                # Get the neighboring triangles regions ids from adjacency_list
                neighbor_region_id_list = [
                    triangle_region_ids[neighbor_id] for neighbor_id in adjacency_list[triangle_id]
                ]
                # if there is a neighbor with a region ID, propagate it
                if len(neighbor_region_id_list) > 0 and max(neighbor_region_id_list) != -1:
                    # random pick a non-negative region id from neighbors
                    neighbor_region_id_list = [x for x in neighbor_region_id_list if x != -1]
                    region_id = neighbor_region_id_list[0]
                    propagated_triangle_ids.append(triangle_id)
                    propagated_region_ids.append(region_id)
            
        # If no triangle can be propagated, stop the loop
        if len(propagated_triangle_ids) == 0:
            can_propagate = False
        else:  
            # Batch assign region IDs
            for triangle_id, region_id in zip(propagated_triangle_ids, propagated_region_ids):
                triangle_region_ids[triangle_id] = region_id
                        
    return triangle_region_ids


def propagate_vertex_region_ids(mesh: o3d.geometry.TriangleMesh, vertex_region_ids: np.ndarray):
    """
    Propagates region IDs from vertices with known region IDs to neighboring vertices
    that have no region ID (default -1), using a breadth-first search approach.
    
    :param mesh: An Open3D TriangleMesh object.
    :param vertex_region_ids: A list of region IDs corresponding to each vertex, with -1 indicating no region ID.
    :return: The updated list of region IDs after propagation.
    """
    # Compute the adjacency list for the vertices in the mesh
    mesh.compute_adjacency_list()
    
    # Initialize a queue for BFS
    queue = deque()
    
    # Enqueue all vertices with known region IDs
    for i, region_id in enumerate(vertex_region_ids):
        if region_id != -1:
            queue.append(i)
    
    # Perform BFS to propagate region IDs
    while queue:
        current_vertex = queue.popleft()
        current_region_id = vertex_region_ids[current_vertex]
        
        for neighbor in mesh.adjacency_list[current_vertex]:
            # If the neighbor vertex has no region ID, assign it the current region ID and enqueue it
            if vertex_region_ids[neighbor] == -1:
                vertex_region_ids[neighbor] = current_region_id
                queue.append(neighbor)
    
    return vertex_region_ids

def build_region_triangle_adjacency_graph(triangle_region_ids, adjacency_list)->nx.Graph:
    """
    Build a region adjacency graph from a triangle mesh and triangle region segmentation.

    Parameters:
        triangle_region_ids: np.ndarray, a list of region IDs corresponding to each triangle
        adjacency_list: A list of lists, where each list contains the indices of neighboring triangles for each triangle
    Returns:
        G: networkx.Graph, a graph where each node represents a region and edges represent adjacency
    """
    # Initialize an empty graph
    graph = nx.Graph()

    # Add nodes to the graph, each representing a region
    for region_id in np.unique(triangle_region_ids):
        if region_id >= 0:
            graph.add_node(region_id)

    # Check adjacency between triangles and add edges between corresponding regions
    for i, neighbors in enumerate(adjacency_list):
        current_region = triangle_region_ids[i]
        for neighbor in neighbors:
            neighbor_region = triangle_region_ids[neighbor]
            if current_region != neighbor_region and current_region >= 0 and neighbor_region >= 0:
                graph.add_edge(current_region, neighbor_region)

    return graph


############# Visualization ###############

def visualize_triangle_region_segmentation(
    mesh: o3d.geometry.TriangleMesh, 
    triangle_region_ids: np.ndarray,
    color_map=Optional[Dict]
):
    """
    Visualizes the segmentation of the mesh into regions based on the region IDs assigned to each triangle.
    
    :param mesh: An Open3D TriangleMesh object.
    :param triangle_region_ids: A list of region IDs corresponding to each triangle.
    """
    # Create a list of colors for each region ID
    region_ids = np.unique(triangle_region_ids)
    if color_map is None:
        # Generate random colors for each region ID
        color_map = {
            region_id: np.random.rand(3) for region_id in region_ids
        }
    triangle_colors = np.array([color_map[region_id] for region_id in triangle_region_ids])
    
    # set -1 region to black
    triangle_colors[triangle_region_ids == -1] = [1, 1, 1]
    
    # Assign vertex colors by max voting of triangle colors
    vertex_colors = np.zeros_like(np.asarray(mesh.vertices))
    for i, triangle in enumerate(mesh.triangles):
        for j in range(3):
            vertex_colors[triangle[j]] += triangle_colors[i]
    
    # Normalize the vertex colors
    vertex_colors /= np.linalg.norm(vertex_colors, axis=1)[:, None]
    
    # Set the vertex colors
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    # Visualize the mesh
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)