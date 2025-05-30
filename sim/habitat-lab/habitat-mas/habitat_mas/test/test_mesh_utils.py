import pytest
import open3d as o3d
import os
import pickle
import numpy as np
from habitat_mas.perception.nav_mesh import NavMesh
from habitat_mas.perception.mesh_utils import (
    propagate_triangle_region_ids,
    propagate_vertex_region_ids,
    visualize_triangle_region_segmentation
)

# Set the root directory for pytest
test_root = os.path.dirname(__file__)
# pytest.rootdir = test_root

def get_navmesh():
    
    navmesh_file_path = os.path.join(test_root, "data/1LXtFkjw3qL.obj")
    mesh = o3d.io.read_triangle_mesh(navmesh_file_path)
    region_bbox_dict_file_path = os.path.join(test_root, "data/1LXtFkjw3qL_region_bbox_dict.pkl")
    region_bbox_dict = pickle.load(open(region_bbox_dict_file_path, "rb"))
    
    navmesh = NavMesh(
        np.array(mesh.vertices), 
        np.array(mesh.triangles), 
        slope_threshold=30
    )
    
    navmesh.segment_by_region_bbox(region_bbox_dict)
    return navmesh
    

def test_propagate_triangle_region_ids():
    visualize=True
    # Parse the .obj file into triangles and triangle_region_ids
    navmesh = get_navmesh()
    triangles = np.array(navmesh.mesh.triangles)
    triangle_region_ids = np.array(navmesh.triangle_region_ids)

    # create a color map for visualization
    region_ids = np.unique(triangle_region_ids)
    color_map = {
        region_id: np.random.rand(3) for region_id in region_ids
    }

    visualize_triangle_region_segmentation(navmesh.mesh, triangle_region_ids, color_map)

    # Call the function with the parsed data
    result = propagate_triangle_region_ids(triangle_region_ids, navmesh.triangle_adjacency_list)

    visualize_triangle_region_segmentation(navmesh.mesh, result, color_map)

    # Assert that the result is as expected
    # Replace this with your actual assertions
    # assert isinstance(result, list)
    # assert all(isinstance(i, int) for i in result)
    # assert all(i >= -2 for i in result)

    # Add more assertions based on your specific requirements and expectations

# def test_propagate_vertex_region_ids():

#     navmesh = get_navmesh()
#     vertex_region_ids = np.array(navmesh.vertex_region_ids)

#     # Call the function with the parsed data
#     result = propagate_vertex_region_ids(navmesh.mesh, vertex_region_ids=None) 

if __name__ == "__main__":
    test_propagate_triangle_region_ids()
    # test_propagate_vertex_region_ids()