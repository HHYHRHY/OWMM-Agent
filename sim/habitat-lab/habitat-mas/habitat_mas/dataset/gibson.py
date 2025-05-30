import copy
import math
import os

import magnum as mn
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from habitat_sim import PathFinder, Simulator
from habitat_sim.utils.common import quat_from_two_vectors, quat_to_coeffs
from habitat.utils.visualizations import maps

obj_nav_class_dict = {
    3: "chair",
    5: "table",
    6: "picture",
    7: "cabinet",
    8: "cushion",
    10: "sofa",
    11: "bed",
    13: "chest_of_drawers",
    14: "plant",
    15: "sink",
    18: "toilet",
    19: "stool",
    20: "towel",
    22: "tv_monitor",
    23: "shower",
    25: "bathtub",
    26: "counter",
    27: "fireplace",
    33: "gym_equipment",
    34: "seating",
    38: "clothes",
}

mp3d_obj_nav_class_list = [
    3,
    5,
    6,
    7,
    8,
    10,
    11,
    13,
    14,
    15,
    18,
    19,
    20,
    22,
    23,
    25,
    26,
    27,
    33,
    34,
    38,
]

hm3d_obj_nav_class_list = [3, 10, 11, 14, 18, 22]


# display a topdown map with matplotlib
def display_map(topdown_map, key_points=None, block=False):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    plt.show(block=block)


# display the path on the 2D topdown map
# @path_points: list of (3,) positions in habitat coords frame
def display_path(
    sim: Simulator, path_points: list, meters_per_pixel=0.025, plt_block=False
):

    scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
    height = scene_bb.y().min
    top_down_map = maps.get_topdown_map(
        sim.pathfinder, height, meters_per_pixel=meters_per_pixel
    )
    recolor_map = np.array(
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
    )
    top_down_map = recolor_map[top_down_map]
    grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
    # convert world trajectory points to maps module grid points
    trajectory = [
        maps.to_grid(
            path_point[2],
            path_point[0],
            grid_dimensions,
            pathfinder=sim.pathfinder,
        )
        for path_point in path_points
    ]
    grid_tangent = mn.Vector2(
        trajectory[1][1] - trajectory[0][1],
        trajectory[1][0] - trajectory[0][0],
    )
    path_initial_tangent = grid_tangent / grid_tangent.length()
    initial_angle = math.atan2(
        path_initial_tangent[0], path_initial_tangent[1]
    )
    # draw the agent and trajectory on the map
    maps.draw_path(top_down_map, trajectory)
    maps.draw_agent(
        top_down_map, trajectory[0], initial_angle, agent_radius_px=8
    )
    print("\nDisplay the map with agent and path overlay:")
    display_map(top_down_map, block=plt_block)


def get_glb_path(scan_dir, scan_name):
    return os.path.join(scan_dir, scan_name, f"{scan_name}.glb")


def get_ply_path(scan_dir, scan_name):
    return os.path.join(scan_dir, scan_name, f"{scan_name}_semantic.ply")


def coo_habitat2mp3d(o3d_cloud):
    """Convert rtab coordinates to mp3d coordinates."""
    # NOTE: habitat use -Y gravity; mp3d use -Z gravity
    quat = quat_from_two_vectors(np.array([0, -1, 0]), np.array([0, 0, -1]))
    o3d_quat = np.roll(quat_to_coeffs(quat), 1)
    r_mat = o3d_cloud.get_rotation_matrix_from_quaternion(o3d_quat)
    o3d_cloud_r = copy.deepcopy(o3d_cloud)
    o3d_cloud_r.rotate(r_mat, center=(0, 0, 0))
    return o3d_cloud_r


# O object_index region_index category_index px py pz  a0x a0y a0z  a1x a1y a1z  r0 r1 r2 0 0 0 0 0 0 0 0
# adapted from https://github.com/facebookresearch/habitat-sim/blob/f8fd41a56b9ad2cbc4781d2a16291ca83ce91964/src/esp/scene/Mp3dSemanticScene.cpp?_pjax=%23js-repo-pjax-container%2C%20div%5Bitemtype%3D%22http%3A%2F%2Fschema.org%2FSoftwareSourceCode%22%5D%20main%2C%20%5Bdata-pjax-container%5D#L139
def getOBB(object_dict):
    center = np.array(
        [
            float(object_dict["px"]),
            float(object_dict["py"]),
            float(object_dict["pz"]),
        ]
    )
    axis0 = np.array(
        [
            float(object_dict["a0x"]),
            float(object_dict["a0y"]),
            float(object_dict["a0z"]),
        ]
    )
    axis1 = np.array(
        [
            float(object_dict["a1x"]),
            float(object_dict["a1y"]),
            float(object_dict["a1z"]),
        ]
    )
    radii = np.array(
        [
            float(object_dict["r0"]),
            float(object_dict["r1"]),
            float(object_dict["r2"]),
        ]
    )

    # Don't need to apply rotation here, it'll already be added in by getVec3f
    boxRotation = np.zeros((3, 3))
    boxRotation[:, 0] = axis0
    boxRotation[:, 1] = axis1
    boxRotation[:, 2] = np.cross(boxRotation[:, 0], boxRotation[:, 1])

    # Don't apply the world rotation here, that'll get added by boxRotation

    return center, 2 * radii, boxRotation


def read_house_file(house_file):
    house_format = """
        H name label num_images num_panoramas num_vertices num_surfaces num_segments num_objects num_categories num_regions num_portals num_levels  0 0 0 0 0  xlo ylo zlo xhi yhi zhi  0 0 0 0 0
        L level_index num_regions label  px py pz  xlo ylo zlo xhi yhi zhi  0 0 0 0 0
        R region_index level_index 0 0 label  px py pz  xlo ylo zlo xhi yhi zhi  height  0 0 0 0
        P portal_index region0_index region1_index label  xlo ylo zlo xhi yhi zhi  0 0 0 0
        S surface_index region_index 0 label px py pz  nx ny nz  xlo ylo zlo xhi yhi zhi  0 0 0 0 0
        V vertex_index surface_index label  px py pz  nx ny nz  0 0 0
        P name  panorama_index region_index 0  px py pz  0 0 0 0 0
        I image_index panorama_index  name camera_index yaw_index e00 e01 e02 e03 e10 e11 e12 e13 e20 e21 e22 e23 e30 e31 e32 e33  i00 i01 i02  i10 i11 i12 i20 i21 i22  width height  px py pz  0 0 0 0 0
        C category_index category_mapping_index category_mapping_name mpcat40_index mpcat40_name 0 0 0 0 0
        O object_index region_index category_index px py pz  a0x a0y a0z  a1x a1y a1z  r0 r1 r2 0 0 0 0 0 0 0 0 
        E segment_index object_index id area px py pz xlo ylo zlo xhi yhi zhi  0 0 0 0 0
    """
    ############ parse house file format ########################3
    parse_dict = {}
    for line in house_format.splitlines():
        line = line.strip().split()
        if len(line) > 0:
            parse_dict[line[0]] = {}
            for shift in range(1, len(line)):
                if line[shift] != "0":  # drop zero column
                    parse_dict[line[0]][line[shift]] = shift

    ########### parse house file with format dictionary ##############
    house_dict = {
        "H": [],  # general house info
        "L": [],  # level
        "R": [],  # region
        "P": [],  # portal
        "S": [],  # triangle mesh surface
        "V": [],  # vertex
        "P": [],  # panorama image
        "I": [],  # image
        "C": [],  # category
        "O": [],  # object
        "E": [],  # segment
    }

    with open(house_file, "r") as f:
        next(f)  # skip file encoding header
        for line in f:
            line = line.split()
            if len(line) > 0:
                item_fmt = parse_dict[line[0]]
                item_dict = {
                    item_property: line[item_fmt[item_property]]
                    for item_property in item_fmt.keys()
                }
                house_dict[line[0]].append(item_dict)
                # if line[0] == 'R':
                #     region_fmt = parse_dict['R']
                #     region_dict = {item: line[region_fmt[item]] for item in region_fmt.keys()}
                #     house_dict['R'].append(region_dict)
                # elif line[0] == 'O':
                #     object_fmt = parse_dict['O']
                #     object_dict = {item: line[object_fmt[item]] for item in object_fmt.keys()}
                #     house_dict['O'].append(object_dict)

    return house_dict, parse_dict


def get_floor_heights(scans_dir, seed=0, sample_n=10000):
    scene_names = os.listdir(scans_dir)
    error_logging = {}
    floor_heights = {}
    for scene in scene_names:
        try:
            navmesh_file = os.path.join(scans_dir, scene, f"{scene}.navmesh")
            house_file = os.path.join(scans_dir, scene, f"{scene}.house")
            house_dict, _ = read_house_file(house_file)
            num_levels = int(house_dict["H"][0]["num_levels"])

            pathfinder = PathFinder()
            pathfinder.load_nav_mesh(navmesh_file)
            assert pathfinder.is_loaded
            pathfinder.seed(seed)
            np.random.seed(seed)

            nav_points = []
            for _ in range(sample_n):
                nav_point = pathfinder.get_random_navigable_point()
                nav_points.append(nav_point)
            nav_points = np.stack(nav_points)
            heights = nav_points[:, 1]  # y-axis upright

            # sort by number of points of that height
            heights, counts = np.unique(heights, return_counts=True)
            sort_idx = np.argsort(-counts)  # descending
            counts = counts[sort_idx]
            heights = heights[sort_idx]

            floor_heights[scene] = sorted(heights[:num_levels].tolist())

            # pathfinder.__del__()
        except Exception as e:
            error_logging[scene] = e

    print("\n\n", "-" * 30)
    for k, v in error_logging.items():
        print(f"Scene {k} error: {v}\n")

    return floor_heights
