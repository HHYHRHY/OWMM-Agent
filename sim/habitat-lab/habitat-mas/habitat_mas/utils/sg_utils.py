# FIXME: This script temporarily import scene graph class from sg_nav package
# which is problematic and dangerous, and needs to be deleted before release

# TODO: Publish semantic_scene_graph from habitat to ROS, and reconstruction GT
# scene graph in sg_nav, after moving agent logics to sg_nav package

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
from scipy.ndimage.morphology import binary_dilation
import quaternion as qt
import ros_numpy 
from sklearn.cluster import DBSCAN
from matplotlib import cm 

from habitat_sim import Simulator
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point, Quaternion
from utils.transformation import points_habitat2world, points_world2habitat
from agents.utils.nms_utils import NMS
from envs.constants import coco_categories, coco_label_mapping


from scene_graph.scene_graph_base import SceneGraphBase
from scene_graph.scene_graph_gt import SceneGraphGTGibson
from scene_graph.object_layer import ObjectLayer, ObjectNode
from scene_graph.region_layer import RegionLayer
from scene_graph.utils import project_points_to_grid_xz


# NOTE: category index in habitat gibson is meaningless 
# def load_scene_priors(scene_prior_file):
#     with open(scene_prior_file, "rb") as f:
#         data = pickle.load(f) 
#     # TODO: load scene_prior_matrix from pickle file 
#     return scene_prior_matrix

class SceneGraphRtabmap(SceneGraphBase):
    # layers #
    object_layer = None
    region_layer = None
    
    # TODO: finetune the DBSCAN parameters
    def __init__(self, rtabmap_pcl, point_features=False, label_mapping=None, 
            scene_bounds=None, grid_map=None, map_resolution=0.05, dbscan_eps=1.0, 
            dbscan_min_samples=5, dbscan_num_processes=4, min_points_filter=5,
            dbscan_verbose=False, dbscan_vis=False, label_scale=2, 
            nms=True, nms_th=0.4):

        # 1. get boundary of the scene (one-layer) and initialize map
        self.scene_bounds = scene_bounds
        self.grid_map = grid_map
        self.map_resolution = map_resolution
        self.point_features = point_features
        self.object_layer = ObjectLayer()
        self.region_layer = RegionLayer()
        
        if self.grid_map is not None:
            self.region_layer.init_map(
                self.scene_bounds, self.map_resolution, self.grid_map
            )

        # 2. use DBSCAN with label as fourth dimension to cluster instance points
        points = ros_numpy.point_cloud2.pointcloud2_to_array(rtabmap_pcl)
        points = ros_numpy.point_cloud2.split_rgb_field(points)
        xyz = np.vstack((points["x"], points["y"], points["z"])).T
        # rgb = np.vstack((points["r"], points["g"], points["b"])).T
        # use g channel to store label 
        g = points["g"].T
        num_class = len(coco_categories)
        # cvrt from 0 for background to -1 for background
        class_label = np.round(g * float(num_class + 1) / 255.0).astype(int) - 1
        # filter out background points 
        objects_mask = (class_label >= 0)
        if not np.any(objects_mask): # no object points in semantic mesh 
            # stop initialization with empty scene graph
            return 
        objects_xyz = xyz[objects_mask]
        objects_label = class_label[objects_mask]
        sem_points = np.concatenate(
            (objects_xyz, label_scale * objects_label.reshape(-1, 1)), axis=1)

        # cluster semantic point clouds to object clusters 
        db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, 
                    n_jobs=dbscan_num_processes).fit(sem_points)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        inst_labels = db.labels_
        object_ids = set(inst_labels)
        if dbscan_verbose:
            num_clusters = len(object_ids) - (1 if -1 in inst_labels else 0)
            num_noise = (inst_labels == -1).sum()
            print(f"DBSCAN on semantic point clouds, num_clusters ({num_clusters}), num_noise ({num_noise})")
        
        if dbscan_vis:
            pass
        
        # 3. non-maximum suppression: filter out noisy detection result 

        if nms:
            valid_object_ids = []
            valid_object_score_bboxes = [] # [p, x, y, z, l,w,h]
            for obj_id in object_ids:
                if obj_id == -1: # outliers
                    continue
                obj_xyz = objects_xyz[inst_labels == obj_id]
                if obj_xyz.shape[0] > min_points_filter:
                    label_modes, _ = stats.mode(objects_label[inst_labels == obj_id], nan_policy="omit")
                    # select mode label as object label
                    obj_label = label_modes[0]
                    if obj_label < len(label_mapping):
                        obj_cls_name = label_mapping[obj_label]
                        center = np.mean(obj_xyz, axis=0)
                        size = np.max(obj_xyz, axis=0) - np.min(obj_xyz, axis=0)
                        score_bbox = np.array([obj_xyz.shape[0], # num of points  
                                            center[0], center[1], center[2],
                                            size[0], size[1], size[2],
                                            ])
                        valid_object_ids.append(obj_id)
                        valid_object_score_bboxes.append(score_bbox)
            
            object_ids = valid_object_ids
            # there could be no valid objects founded 
            if len(valid_object_ids) > 0:
                valid_object_score_bboxes = np.stack(valid_object_score_bboxes, axis=0)
                selected_indices, _ = NMS(valid_object_score_bboxes, nms_th)
                object_ids = [valid_object_ids[idx] for idx in selected_indices]
                
        # 4. create object nodes in scene graph 
        
        for obj_id in object_ids:
            
            if obj_id == -1: # outliers
                continue
            
            obj_xyz = objects_xyz[inst_labels == obj_id]
            if obj_xyz.shape[0] > min_points_filter:
                label_modes, _ = stats.mode(objects_label[inst_labels == obj_id], nan_policy="omit")
                # select mode label as object label
                obj_label = label_modes[0]
                obj_cls_name = ""
                if obj_label >= 0:
                    obj_cls_name = label_mapping[obj_label]
                # else:
                #     obj_cls_name = "background"
                    # use axis-aligned bounding box for now 
                    center = np.mean(obj_xyz, axis=0)
                    rot_quat = np.array([0, 0, 0, 1])  # identity transform
                    size = np.max(obj_xyz, axis=0) - np.min(obj_xyz, axis=0)
                    if not self.point_features:
                        object_vertices = None
                    else:
                        object_vertices = obj_xyz
                        
                    object_node = self.object_layer.add_object(
                        center,
                        rot_quat,
                        size,
                        id=obj_id,
                        class_name=obj_cls_name,
                        label=obj_label,
                        vertices=object_vertices   
                    )

                # no region prediction module implemented 
                # connect object to region
                # region_node.add_object(object_node)

        return



if __name__ == "__main__":
    # import importlib.util
    # sys.path.append(UTILS_DIR.parent.parent)
    # NOTE: run this demo with python -m agents.utils.sg_utils
    from utils.simulator import init_sim
    import rosbag
    import matplotlib.pyplot as plt
    import open3d as o3d 
    
    TEST_SCENEGRAPH_SIMGT = True
    TEST_SCENEGRAPH_RTABMAP = False
    
    if TEST_SCENEGRAPH_SIMGT:
        test_scene = "/media/junting/SSD_data/habitat_data/scene_datasets/gibson/Darden.glb"
        sim = sim, action_names = init_sim(test_scene, init_pos=[1.0, 0.0, 1.0])
        # FIXME: get initial pos and initial rotation of the agent to vis correctly 
        # NOTE: temporarily not test grammar 
        gt_scenegraph = SceneGraphGTGibson(sim)
        bag = rosbag.Bag(
            "/media/junting/SSD_data/struct_nav/rosbag_record/sgnav_sem_gt_Darden_1.bag",
            "r",
        )

        # read first message and construct grid map
        topic, grid_map_msg, t = next(
            bag.read_messages(topics=["/rtabmap/grid_map"])
        )
        grid_map = GridMap.from_msg(grid_map_msg)
        bag.close()

        # demo: return partial scene graph masked by grid_map
        partial_scenegraph = gt_scenegraph.get_partial_scene_graph(grid_map)

        # visualize rtabmap gridmap and habitat
        obj_centers = np.stack(
            [obj_node.center for obj_id, obj_node in 
             partial_scenegraph.object_layer.obj_dict.items()], axis=0
        )
        obj_centers_2d = project_points_to_grid_xz(
            gt_scenegraph.scene_bounds, obj_centers, gt_scenegraph.meters_per_grid
        )
        fig = plt.figure(figsize=(16, 8))
        fig.add_subplot(121)
        plt.imshow(gt_scenegraph.free_space_grid)
        plt.scatter(
            x=obj_centers_2d[:, 0], y=obj_centers_2d[:, 1], c="r", marker="o"
        )
        fig.add_subplot(122)
        vis_grid = grid_map.grid.copy()
        vis_grid[vis_grid == 100] = 1
        plt.imshow(vis_grid, origin="lower")
        plt.show()
    
    if TEST_SCENEGRAPH_RTABMAP:
        
        cm_viridis = cm.get_cmap('viridis')
        obj_color_max = 50
        bag = rosbag.Bag("/home/junting/Downloads/dataset/rtabmap_sem_pcl/2022-07-26-10-46-44.bag","r")
        cloud_msgs = [cloud_msg for topic, cloud_msg, t in
            bag.read_messages(topics=["/rtabsem/cloud_map"])]
        bag.close()
        # initialize open3d visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        ctr = vis.get_view_control()
        ctr.set_zoom(4)
        o3d_pcl = o3d.geometry.PointCloud()
        vis.add_geometry(o3d_pcl)
        
        bboxes = []
        for cloud_msg in cloud_msgs:
        # for cloud_msg in [cloud_msgs[-3]]:
            
            sg = SceneGraphRtabmap(cloud_msg, point_features=True, 
                                   label_mapping=coco_label_mapping)

            print({obj_id:(sg.object_layer.obj_dict[obj_id].class_name, 
                           len(sg.object_layer.obj_dict[obj_id].vertices))
                   for obj_id in sg.object_layer.obj_ids 
                   if sg.object_layer.obj_dict[obj_id].class_name != "background"
                   and len(sg.object_layer.obj_dict[obj_id].vertices) > 10})

            for bbox in bboxes:
                vis.remove_geometry(bbox)
            
            bboxes = []
            points = []
            colors = []
            for obj_id in sg.object_layer.obj_ids:
                
                obj_node = sg.object_layer.obj_dict[obj_id]
                if (obj_node.class_name != "background" and 
                    len(sg.object_layer.obj_dict[obj_id].vertices)):
                    obj_color = np.array(cm_viridis((obj_id % obj_color_max)/obj_color_max))
                    
                    points.append(obj_node.vertices)
                    colors.append(np.repeat(obj_color[:3].reshape(1,3), 
                                            repeats=obj_node.vertices.shape[0], axis=0))
                    min_bound = obj_node.center - obj_node.size / 2.0
                    max_bound = obj_node.center + obj_node.size / 2.0
                    o3d_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
                    # o3d_bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
                    #     o3d.utility.Vector3dVector(obj_node.vertices)
                    # )
                    o3d_bbox.color = (0,1,0)
                    bboxes.append(o3d_bbox)
            
            if len(points) > 0:
                points = np.concatenate(points, axis=0)     
                colors = np.concatenate(colors, axis=0)
                o3d_pcl.points = o3d.utility.Vector3dVector(points)
                o3d_pcl.colors = o3d.utility.Vector3dVector(colors)
            else: 
                o3d_pcl.points = o3d.utility.Vector3dVector([])
                o3d_pcl.colors = o3d.utility.Vector3dVector([])
            
            for bbox in bboxes:
                vis.add_geometry(bbox)
            
            vis.update_geometry(o3d_pcl)
            # ctr.set_zoom(1)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.3)
        vis.destroy_window()
            
            
            
        