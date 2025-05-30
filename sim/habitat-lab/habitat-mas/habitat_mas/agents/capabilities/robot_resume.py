from typing import Dict, List, Tuple
import json
from json import encoder
import numpy as np
import os
from habitat.articulated_agents.robots.dji_drone import DJIDrone
from habitat.articulated_agents.robots.fetch_robot import FetchRobot
from habitat.articulated_agents.robots.stretch_robot import StretchRobot
from habitat.articulated_agents.robots.spot_robot import SpotRobot

from habitat_mas.agents.capabilities.parse_urdf import (
    parse_urdf,
    query_llm_with_urdf
)
from habitat_mas.agents.robots.defaults import (
    robot_urdf_paths,
    robot_arm_workspaces,
    robot_perception,
    dji_camera_params,
    fetch_camera_params,
    stretch_camera_params,
    spot_camera_params
)
from habitat_mas.agents.capabilities.perception import get_cameras_height_and_type
from habitat_mas.agents.capabilities.manipulation import get_all_robot_arm_workspace

# Set the float precision to 2 decimal places
encoder.FLOAT_REPR = lambda o: format(o, '.2f')

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)        


class RobotResume:
    """
    This class is used to store the resume of a robot. It contains the capabilities of the robot.
    - robot_id: the id of the robot
    - robot_type: the type of the robot
    - mobility: the mobility capabilities of the robot, including its motion type(Fixed, Legged, Wheeled, Tracked, Aerial, etc.).
    - perception: the perception capabilities of the robot, including its sensor type(Lidar, Camera, Depth Camera, etc.), and its perceivable area. 
    - manipulation: the manipulation capabilities of the robot, including its end-effector type(Gripper, Suction, etc.), and its manipulation workspace.
    """
    def __init__(self, robot_id: str, robot_type: str, mobility: Dict, perception: Dict, manipulation: Dict):
        self.robot_id = robot_id
        
        self.robot_id = robot_id
        self.robot_type = robot_type
        self.mobility = mobility
        self.perception = perception
        self.manipulation = manipulation
        
    @classmethod
    def from_urdf(cls, robot_id: str, urdf_path: str, robot_type="Unknown") -> 'RobotResume':
        """
        Generate the resume of a robot from its URDF file. The resume includes the capabilities of the robot.
        NOTE: Save the robot resume to file to accelerate batch experiments.

        Args:
            robot_id: the id of the robot
            articulated_agent: the ArticulatedAgentBase object
            urdf_path: the path to the URDF file of the robot
        """
        urdf = parse_urdf(urdf_path)
        summary = query_llm_with_urdf(urdf=urdf,
                                      robot_id=robot_id, robot_type=robot_type,
                                      manipulation=robot_arm_workspaces, perception=robot_perception)
        # convert the summary from json string to dictionary
        # {
        #     "mobility": {"summary": ...},
        #     "perception": {"summary": ...},
        #     "manipulation": {"summary": ...}
        # }
        summary_dict = json.loads(summary)
        mobility = summary_dict["mobility"]
        perception = summary_dict["perception"]
        manipulation = summary_dict["manipulation"]
        
        return cls(robot_id, robot_type, mobility, perception, manipulation)
            
    @classmethod
    def load_from_json(cls, file_path: str) -> 'RobotResume':
        """
        Load the robot resume from a json file. Each property is saved as a key-value pair.
        """
        json_dict: Dict = json.load(open(file_path, "r"))
        return cls(
            robot_id=json_dict.get("robot_id", ""),
            robot_type=json_dict.get("robot_type", "Unknown"),
            mobility=json_dict.get("mobility", {}),
            perception=json_dict.get("perception", {}),
            manipulation=json_dict.get("manipulation", {})
        )
    
    def save_to_json(self, file_path):
        """
        Save the robot resume to a json file. Each property is saved as a key-value pair.
        """
        
        json.dump(
            {
                "robot_id": self.robot_id,
                "robot_type": self.robot_type,
                "mobility": self.mobility,
                "perception": self.perception,
                "manipulation": self.manipulation
            },
            open(file_path, "w"),
            indent=2,
            cls=NumpyEncoder
        )
        
    def __str__(self):
        """
        Convert the  robot resume to a json string, with each property as a key-value pair.
        """
        return json.dumps({
            "robot_id": self.robot_id,
            "robot_type": self.robot_type,
            "mobility": self.mobility,
            "perception": self.perception,
            "manipulation": self.manipulation
        }, indent=2)
        
    def to_string(self):
        """
        Convert the robot resume to a string, with each property as a key-value pair.
        """
        return str(self)
        
def generate_full_robot_resumes():
    """
    Generate the full robot resumes for all the robots in the habitat lab.
    There are 3 steps to generate full robot resumes:
    - Query LLM with URDF to get the robot capabilities summary
    - Get the camera height and type
    - Get the arm workspace
    """
    
    robot_camera_params = {
        "FetchRobot": fetch_camera_params,
        "SpotRobot": spot_camera_params,
        "StretchRobot": stretch_camera_params,
        "DJIDrone": dji_camera_params
    }    
    
    cur_dir = os.path.dirname(__file__)
    habitat_mas_data_dir = os.path.join(cur_dir, "../../data")

    
    for robot_name, camera_params in robot_camera_params.items():
        org_urdf_file_path = robot_urdf_paths[robot_name]
        org_urdf_file_name = os.path.basename(org_urdf_file_path).split(".")[0]        
        
        camera_params = robot_camera_params[robot_name]
        for camera_setup, camera_param in camera_params.items():
            
            urdf_path = os.path.join(habitat_mas_data_dir, "robot_urdf", f"{org_urdf_file_name}_{camera_setup}.urdf")
            robot_resume = RobotResume.from_urdf(f"{robot_name}_{camera_setup}", urdf_path, robot_name)
            camera_links = [f"{camera_name}_camera" for camera_name in camera_param.keys()]
            camera_info = get_cameras_height_and_type(urdf_path, camera_links, robot_name)
            robot_resume.perception["cameras_info"] = camera_info
            print(f"Camera info for {robot_name}_{camera_setup}: {camera_info}")
            
            # Skip the arm workspace for the DJI drone
            robot_arm_workspace = {}
            if robot_name != "DJIDrone":
                robot_arm_workspace = robot_arm_workspaces[robot_name]
            print(f"Robot arm workspace for {robot_name}: {robot_arm_workspace}")
            robot_resume.manipulation["arm_workspace"] = robot_arm_workspace
            
            robot_resume.save_to_json(os.path.join(habitat_mas_data_dir, "robot_resume", f"{robot_name}_{camera_setup}.json"))
            
            
if __name__ == "__main__":
    generate_full_robot_resumes()