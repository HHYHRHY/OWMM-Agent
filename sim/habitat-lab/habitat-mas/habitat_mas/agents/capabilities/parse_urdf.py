import os
from typing import Dict, List, Tuple
import numpy as np
import magnum as mn
from openai import OpenAI
from urchin import URDF, Link, Joint
import networkx as nx
from networkx.readwrite.text import generate_network_text
import habitat_sim
from omegaconf import DictConfig
from habitat.articulated_agents.mobile_manipulator import (
    ArticulatedAgentCameraParams,
    MobileManipulatorParams
)
from habitat.articulated_agents.robots.fetch_robot import FetchRobot
from habitat.articulated_agents.robots.spot_robot import SpotRobot
from habitat.articulated_agents.robots.stretch_robot import StretchRobot
from habitat.articulated_agents.robots.dji_drone import DJIDrone

from habitat_mas.agents.llm_api_keys import get_openai_client
from habitat_mas.agents.sim_utils import make_cfg, default_sim_settings
from habitat_mas.agents.robots.defaults import (
    get_robot_link_id2name_map,
    fetch_camera_params,
    spot_camera_params,
    stretch_camera_params,
    dji_camera_params
)

default_tasks = """
Cross-floor object rearrangement: Robot need to cross floor to pick and place objects from one floor to another;
Single-floor object perception: Robots need to perceive the object placed in abnormal positions;
Single-floor object rearrangement: Robots need to pick and place objects in abnormal positions.
"""
# default_tasks = """
# • Task 1: Cross-floor object navigation: In the multi-floor task requires the collaboration of robots with different base types to navigate to multiple objects in the scene. The wheeled robot operates on a single floor, while the legged robot can navigate between floors, emphasizing the need for coordinated planning with awareness of mobility capabilities of different robots.
# • Task 2: Cooperative perception for manipulation: In single-floor scenario, the robots need to acquire good RGBD perception of an object to enable complex manipulation. The target objects are visible only from viewpoints (eg. camera height will affect the viewpoint) of certain robots. It is necessary to reason about robot sensor type and viewpoint to succeed in this task.
# • Task 3: Collaborative home rearrangement. This single-floor task involves rearranging objects placed in varying positions including the ground, high shelves, or a bed center far from navigable area, requiring the robots with different arm workspace to understand their availablility for different rearrangement targets.
# """

def generate_robot_link_id2name():
    # Define the settings for the simulator
    produce_debug_video = False
    observations = []
    cfg_settings = default_sim_settings.copy()
    cfg_settings["scene"] = "NONE"
    cfg_settings["enable_physics"] = True

    # Create the simulator configuration
    hab_cfg = make_cfg(cfg_settings)
    # Define the robots to initialize
    robot_configs = [
        {"path": "data/robots/hab_fetch/robots/hab_fetch.urdf", "class": FetchRobot},
        {"path": "data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf", "class": SpotRobot},
        {"path": "data/robots/hab_stretch/urdf/hab_stretch.urdf", "class": StretchRobot},
        # NOTE: DJI Drone already has a camera in the URDF
        {"path": "data/robots/dji_drone/urdf/dji_m100_sensors_scaled.urdf", "class": DJIDrone},
    ]
    
    robot_link_id2name_map = {}

    for robot_config in robot_configs:
        with habitat_sim.Simulator(hab_cfg) as sim:
            obj_template_mgr = sim.get_object_template_manager()
            rigid_obj_mgr = sim.get_rigid_object_manager()

            # setup the camera for debug video (looking at 0,0,0)
            sim.agents[0].scene_node.translation = [0.0, -1.0, 2.0]

            # add a ground plane
            cube_handle = obj_template_mgr.get_template_handles("cubeSolid")[0]
            cube_template_cpy = obj_template_mgr.get_template_by_handle(cube_handle)
            cube_template_cpy.scale = np.array([5.0, 0.2, 5.0])
            obj_template_mgr.register_template(cube_template_cpy)
            ground_plane = rigid_obj_mgr.add_object_by_template_handle(cube_handle)
            ground_plane.translation = [0.0, -0.2, 0.0]
            ground_plane.motion_type = habitat_sim.physics.MotionType.STATIC

            # compute a navmesh on the ground plane
            navmesh_settings = habitat_sim.NavMeshSettings()
            navmesh_settings.set_defaults()
            navmesh_settings.include_static_objects = True
            sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
            sim.navmesh_visualization = True

            # add the robot to the world via the wrapper
            robot_path = robot_config["path"]
            robot_class = robot_config["class"]
            agent_config = DictConfig({"articulated_agent_urdf": robot_path})
            robot = robot_class(agent_config, sim, fixed_base=True)

            # reconfigure and update the robot
            robot.reconfigure()
            robot.update()

            # map link ids to names and print the map
            map_link_id_to_name = {}
            for link_id in robot.sim_obj.get_link_ids():
                map_link_id_to_name[link_id] = robot.sim_obj.get_link_name(link_id)
            
            robot_link_id2name_map[robot_class.__name__] = map_link_id_to_name
            
    return robot_link_id2name_map

################# URDF Processing #################

def parse_urdf(file_path):
    """Function to read and parse URDF file"""
    robot = URDF.load(file_path, lazy_load_meshes=True)
    return robot

def add_cameras_to_urdf(urdf: URDF, camera_params: Dict[str, ArticulatedAgentCameraParams], 
                        robot_name:str, skip_link=["third"], link_suffix="_camera"):
    """
    Add camera to the URDF
    
    Parameters:
        urdf (URDF): The URDF object to which the camera is to be added.
        camera_params: A dictionary containing the camera parameters.
        robot_name: The name of the robot. [FetchRobot, SpotRobot, StretchRobot]
        skip_link: A list of links to skip when calculating the height of the camera. Default is ["third"].
    """
    robot_link_id2name_map = get_robot_link_id2name_map(robot_name)
    
    for camera_name, camera_param in camera_params.items():
        if camera_name in skip_link:
            continue
        camera_link = Link(
            name=f"{camera_name}{link_suffix}",
            inertial=None,
            visuals=[],
            collisions=[],
        )
        urdf._links.append(camera_link)
        
        if camera_param.attached_link_id == -1:
            camera_attached_link = urdf.base_link.name
        else:
            camera_attached_link = robot_link_id2name_map[camera_param.attached_link_id]
        
        # Origin must be specified as a 4x4 homogenous transformation matrix
        # Convert camera_param.cam_offset_pos to 4x4 matrix
        origin = np.eye(4, dtype=np.float64)
        origin[:3, 3] = np.array(camera_param.cam_offset_pos)
        # TODO: orientation not used in following applications, skip for now
        # orientation = camera_param.cam_orientation
        relative_transform: mn.Matrix4 = camera_param.relative_transform
        
        # transform the origin matrix with relative_transform
        habitat2urdf_transform = mn.Matrix4.rotation_x(mn.Deg(90))
        origin[:3, 3] = habitat2urdf_transform.transform_point(
            relative_transform.transform_point(origin[:3, 3])
        )

        # orientation = relative_transform.transform_vector(orientation)
        
        camera_joint = Joint(
            name=f"{camera_name}_joint",
            parent=camera_attached_link,
            child=f"{camera_name}{link_suffix}",
            joint_type="fixed",
            origin=origin,
            # axis=orientation
        )
        urdf._joints.append(camera_joint)
        

def save_urdf(urdf: URDF, file_path):
    """Function to save URDF to file"""
    urdf.save(file_path)

def save_urdf_with_cameras(urdf_file_path: str, camera_params: Dict[str, ArticulatedAgentCameraParams], file_path, robot_name):
    """Function to save URDF with cameras to file"""
    urdf = parse_urdf(urdf_file_path)
    add_cameras_to_urdf(urdf, camera_params, robot_name)
    save_urdf(urdf, file_path)


################# LLM URDF Understanding #################

def generate_tree_text_with_edge_types(graph, root):
    """
    Generate a tree structure text including node names and edge types.
    
    Parameters:
    graph (nx.DiGraph): The directed graph representing the tree structure.
    root (node): The root node of the tree.
    
    Returns:
    str: A string representing the tree structure with node names and edge types.
    """
    def dfs(node, depth):
        """
        Perform a depth-first search to generate the tree structure text.
        
        Parameters:
        node (node): The current node in the DFS traversal.
        depth (int): The current depth in the tree.
        
        Returns:
        str: A string representing the subtree rooted at the current node.
        """
        indent = "\t" * depth
        lines = [f"{indent}Link({node})"]
        
        for child in graph.successors(node):
            edge_type = graph[node][child].get('type', 'unknown')
            lines.append(f"{indent}├── Joint of type({edge_type})──Link({child})")
            lines.extend(dfs(child, depth + 1))
        
        return lines
    
    # Generate the tree structure text starting from the root
    tree_structure = dfs(root, 0)
    
    return "\n".join(tree_structure)

# Function to generate physics capabilities summary using OpenAI API
def query_llm_with_urdf(urdf: URDF, model_name="gpt-4o", task_text=default_tasks,
                        robot_id=None, robot_type=None, manipulation=None, perception=None):
    # Extract relevant information from URDF
    if robot_type is None:
        robot_name = urdf.name
    else:
        robot_name = robot_type

    urdf_text = ""
    
    # convert urdf to networkx DiGraph
    urdf_nx_graph = nx.DiGraph()
    
    # Add nodes and edges from urdf._G to the graph
    for link in urdf.links:
        # only keep the name and mass of the node
        # urdf_nx_graph.add_node(node.name, **node.__dict__)
        urdf_nx_graph.add_node(link.name)
    
    for joint in urdf.joints:
        # urdf_nx_graph.add_edge(edge.parent, edge.child, **edge.__dict__)
        urdf_nx_graph.add_edge(joint.parent, joint.child, type=joint.joint_type) 
    
    # for line in generate_network_text(urdf_nx_graph):
    #     urdf_text += line + "\n"
    for node in urdf_nx_graph.nodes:
        if node == urdf.base_link.name:
            root_node = node
    urdf_text = generate_tree_text_with_edge_types(urdf_nx_graph, root_node)
    # TODO: modify the prompt so the LLM attends to the cameras in urdf
    # TODO: prompt with fk function call
#     system_prompt = """
# You are a robot urdf structure reasoner. You will be given a robot's urdf tree-structured text, and you need to provide a summary of the robot's physics capabilities.
# Please pay attention to the task and summarize the mobility, perception, and manipulation capabilities of the robot that are relevant to the task.
# For the robot's mobility capability, consider the robot's base type (e.g., flying, walking, wheeled), and evaluate the robot's ability to move between floors, factoring in whether it can fly, walk up stairs, or is limited to ground-level movement on wheels.
# For the robot's perception capability, pay attention to different robots' camera height in perception part, and compare to inference their perception.
# For the robot's manipulation capability, consider different robots' arm workspace in manipulation part, pay attention to their radius and the center point of the manipulation space, and compare to inference the height and distance the robot can manipulate.
# The response should be a JSON object with each capability as a dictionary, containing a summary field:
# {
#     "mobility": {"summary": ...},
#     "perception": {"summary": ...},
#     "manipulation": {"summary": ...}
# }
# """
#     prompt = f"""
# The robot's name is {robot_name}. Here is the tree structure of the robot's URDF:
# {urdf_text}
# The robot task includes: {task_text}
# Pay attention to the the radius from the center point, (including the height and horizonal distance from the center)
# Notably, The coordinates are in [x, z, y] format, where the second coordinate represents height, and the first and third coordinates represent horizontal positions.
# The robots' manipulation includes: {manipulation}
# The robots' perception includes: {perception}
# Please provide a summary of the robot's physics capabilities based on corresponding information and task.
# """
    system_prompt = """
You are a robot urdf structure reasoner. You will be given a robot's urdf tree-structured text, and you need to provide a summary of the robot's physics capabilities in brief and short sentences.
Please pay attention to the task and summarize the mobility, perception, and manipulation capabilities of the robot that are relevant to the task.
For the mobility capability, consider the robot's base type (e.g., flying, walking, wheeled), and evaluate the robot's ability to move between floors.
For the perception capability, consider the robot's sensors, pay attention to different cameras' positions.
For the manipulation capability, consider the robot's end effectors and the joints, pay attention to robots' arm workspace.
The response should be a JSON object with each capability as a dictionary, containing a summary field:
{
    "mobility": {"summary": ...},
    "perception": {"summary": ...},
    "manipulation": {"summary": ...}
}
"""
    
    prompt = f"""
The robot's name is {robot_name}. Here is the tree structure of the robot's URDF:
{urdf_text}
The robot task includes: {task_text}
Please provide a brief summary of the robot's physics capabilities in short sentences based on this information and task.
"""
    
    # print(prompt)
    
    # Call OpenAI API to generate the summary
    client = get_openai_client()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        response_format={ "type": "json_object"}
    )
    summary = response.choices[0].message.content
    return summary


def generate_physics_summary(urdf_file_path:str, save_path=None):
    urdf = parse_urdf(urdf_file_path)
    summary = query_llm_with_urdf(urdf)
    print("Physics Capabilities Summary:")
    print(summary)
    if save_path:
        with open(save_path, "w") as f:
            f.write(summary)


def generate_urdf_with_cameras():
    
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(cur_dir, "../../../../data")

    robot_urdfs = {
        "FetchRobot": os.path.join(data_dir, "robots/hab_fetch/robots/hab_fetch.urdf"),
        "SpotRobot": os.path.join(data_dir, "robots/hab_spot_arm/urdf/hab_spot_arm.urdf"),
        "StretchRobot": os.path.join(data_dir, "robots/hab_stretch/urdf/hab_stretch.urdf"),
        "DJIDrone": os.path.join(data_dir, "robots/dji_drone/urdf/dji_m100_sensors_scaled.urdf"),
    }
    
    robot_camera_params = {
        "FetchRobot": fetch_camera_params,
        "SpotRobot": spot_camera_params,
        "StretchRobot": stretch_camera_params,
        "DJIDrone": dji_camera_params
    }

    for robot_name, urdf_file_path in robot_urdfs.items():
        urdf_file_name = os.path.basename(urdf_file_path).split(".")[0]
        # robot_class:ArticulatedAgentBase = eval(robot_name)
        camera_params = robot_camera_params[robot_name]
        for camera_setup, camera_param in camera_params.items():
            print(f"Generating urdf for camera setup {camera_setup} of {robot_name}")
            save_path = os.path.join(cur_dir, f"../../data/robot_urdf/{urdf_file_name}_{camera_setup}.urdf")
            save_urdf_with_cameras(urdf_file_path, camera_param, save_path, robot_name)


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(cur_dir, "../../../../data")

    # Data generation: Get robot link to name mapping from classes
    # robot_link_id2name_map = generate_robot_link_id2name()
    # print("\n----------------------------------------------------------------------------\n")
    # print(robot_link_id2name_map)
    
    # Data generation: Add cameras to urdf file and save to new file
    # generate_urdf_with_cameras()

    # urdf_file_path = os.path.join(data_dir, "robots/hab_spot_arm/urdf/hab_spot_arm.urdf")
    # save_path = os.path.join(cur_dir, "../../data/robot_resume/hab_spot_arm.txt")
    # urdf_file_path = os.path.join(cur_dir, "../../data/robot_urdf/hab_spot_arm_with_cameras.urdf")
    # save_path = os.path.join(cur_dir, "../../data/robot_resume/hab_spot_arm_with_cameras.txt")
    # generate_physics_summary(urdf_file_path, save_path)
