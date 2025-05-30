import os
from typing import List, Dict, Any
import numpy as np
import json
from gym import spaces
from dataclasses import dataclass, field

from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat_mas.dataset.defaults import habitat_mas_data_dir
from habitat_mas.scene_graph.scene_graph_hssd import SceneGraphHSSD
from habitat_mas.scene_graph.scene_graph_mp3d import SceneGraphMP3D
from habitat_mas.scene_graph.utils import (
    generate_objects_description, 
    generate_agents_description, 
    generate_mp3d_objects_description, 
    generate_mp3d_agents_description,
    generate_region_adjacency_description
)    

@registry.register_sensor
class HSSDSceneDescriptionSensor(Sensor):
    """Sensor to generate text descriptions of the scene from the environment simulation."""
    # TODO: consider if all scene description sensors should have the same uuid, since only one can be used in a dataset
    cls_uuid: str = "scene_description"
    
    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)
        
    def _get_uuid(self, *args, **kwargs):
        return HSSDSceneDescriptionSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TEXT

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Text(max_length=10000)
    
    def get_observation(self, *args, **kwargs):
        """Generate text descriptions of the scene."""
        
        # Initialize scene graph
        sg = SceneGraphHSSD()
        sg.load_gt_scene_graph(self._sim)
        
        # Generate scene descriptions
        objects_description = generate_objects_description(self._sim, sg.object_layer)
        agent_description = generate_agents_description(sg.agent_layer, sg.region_layer, sg.nav_mesh)
        
        scene_description = {
            "objects_description": objects_description,
            "agent_description": agent_description
        }
        
        # convert dict to json string
        # scene_description_str = json.dumps(scene_description)
        scene_description_str = 'Here are the descriptions of the scene: \n'
        scene_description_str += objects_description + '\n'
        scene_description_str += agent_description
        return scene_description_str

class MP3DSceneDescriptionSensor(Sensor):
    """Sensor to generate text descriptions of the scene from the environment simulation."""
    cls_uuid: str = "scene_description"
    
    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)
        
    def _get_uuid(self, *args, **kwargs):
        return MP3DSceneDescriptionSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TEXT

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Text(max_length=10000)
    
    def get_observation(self, *args, **kwargs):
        """Generate text descriptions of the scene."""
        
        # Initialize scene graph
        sg = SceneGraphMP3D()
        sg.load_gt_scene_graph(self._sim)
        
        # Generate scene descriptions
        regions_description = generate_region_adjacency_description(sg.region_layer) 
        objects_description = generate_mp3d_objects_description(sg.object_layer)
        agent_description = generate_mp3d_agents_description(sg.agent_layer, sg.region_layer)
        
        scene_description = {
            "regions_description": regions_description,
            "objects_description": objects_description,
            "agent_description": agent_description
        }
        
        # convert dict to json string
        # scene_description_str = json.dumps(scene_description)
        scene_description_str = 'Here are the descriptions of the scene: \n'
        scene_description_str += objects_description + '\n'
        scene_description_str += agent_description + '\n'
        scene_description_str += regions_description
        return scene_description_str   
    
@registry.register_sensor
class RobotResumeSensor(Sensor):
    """Sensor to load and retrieve robot resumes from JSON files."""
    cls_uuid: str = "robot_resume"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        self.robot_resume_dir = os.path.join(habitat_mas_data_dir, 
                                             config.robot_resume_dir)
        self.robot_resumes = self.load_robot_resumes()
        super().__init__(**kwargs)

    def _get_uuid(self, *args, **kwargs):
        return RobotResumeSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TEXT

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Text(max_length=10000)
    
    def get_observation(self, robot_configs, *args, **kwargs):
        """
        Retrieve the resumes for all agents as a single string.
        """
            
        robot_resumes = {}
        for agent_config in robot_configs:
            # agent_handle = agent_config["agent_type"]
            agent_handle = f"agent_{agent_config['agent_idx']}"
            agent_type = agent_config["agent_type"]
            robot_resume_file = os.path.join(self.robot_resume_dir, f"{agent_type}.json")
            if os.path.exists(robot_resume_file):
                with open(robot_resume_file, "r") as f:
                    robot_resume = json.load(f, parse_float=lambda x: round(float(x), 2))
                    robot_resumes[agent_handle] = robot_resume
        
        # convert dict to json string
        robot_resumes_str = json.dumps(robot_resumes)
        
        return robot_resumes_str

    def load_robot_resumes(self) -> Dict[str, Dict]:
        """
        Load robot resumes from JSON files located in the robot_resume_dir directory.
        The method reads the agent handles from the environment config and loads their corresponding resumes.
        """
        robot_resumes = {}
        for file in os.listdir(self.robot_resume_dir):
            if file.endswith(".json"):
                agent_handle = file.split(".")[0]
                robot_resume_file = os.path.join(self.robot_resume_dir, file)
                with open(robot_resume_file, "r") as f:
                    robot_resume = json.load(f)
                    robot_resumes[agent_handle] = robot_resume

        return robot_resumes

@registry.register_sensor
class PddlTextGoalSensor(Sensor):
    
    max_length = 10000
    
    def __init__(self, sim, config, *args, task, **kwargs):
        self._task = task
        self._sim = sim
        # By default use verbose string representation
        self.text_type = config.get("text_type", False)
        self.task_description = config.get("task_description", "")
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return "pddl_text_goal"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TEXT

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Text(max_length=self.max_length)

    def _predicate_to_text(self, predicate):
        predicate_name = predicate.name
        args = predicate._arg_values

        if predicate_name == "in":
            obj = args[0].name
            receptacle = args[1].name
            return f"The object '{obj}' is inside the '{receptacle}'."

        elif predicate_name == "holding":
            obj = args[0].name
            robot_id = args[1].name
            return f"The robot '{robot_id}' is holding the object '{obj}'."

        elif predicate_name == "not_holding":
            robot_id = args[0].name
            return f"The robot '{robot_id}' is not holding any object."

        elif predicate_name == "robot_at":
            location = args[0].name
            robot_id = args[1].name
            return f"The robot '{robot_id}' is located at '{location}'."

        elif predicate_name == "at":
            obj = args[0].name
            location = args[1].name
            return f"The object '{obj}' is placed at '{location}'."

        elif predicate_name == "detected_object":
            obj = args[0].name
            robot_id = args[1].name
            return f"The robot '{robot_id}' has detected the object '{obj}'."

        elif predicate_name == "is_detected":
            obj = args[0].name
            return f"The object '{obj}' has been detected by any robot."

        elif predicate_name == "any_at":
            obj = args[0].name
            return f"Any robot is currently at the location of '{obj}'."

        else:
            return str(predicate)

    def _get_description(self, goal) -> str:
        description = f"Goal of this episode is the logical operation {goal._expr_type.value} of the following conditions:\n"
        for i, predicate in enumerate(goal.sub_exprs):
            description += f"{i}. {self._predicate_to_text(predicate)}\n"
        return description

    def _convert_goal_to_text(self, goal):
        if self.text_type == "compact_str":
            goal_str = goal.compact_str
        elif self.text_type == "verbose_str":
            goal_str = goal.verbose_str
        elif self.text_type == "description":
            goal_str = self._get_description(goal)
        else:
            raise ValueError(f"Unknown text type {self.text_type}")
        description = f"""
                        {self.task_description}
                        {goal_str}"""
        return description

    def get_observation(self, observations, episode, *args, **kwargs):
        goal = self._task.pddl_problem.goal
        goal_description = self._convert_goal_to_text(goal)
        # return goal_description
        return np.array(
            list(goal_description.ljust(self.max_length)[:self.max_length].encode('utf-8'))
            , dtype=np.uint8)

def get_text_sensors(sim, *args, **kwargs):
    
    # sensor_keys = kwargs.get("sensor_keys", None)
    # sensor_keys = kwargs.get("sensor_keys", {})
    lab_sensors_config = kwargs.get("lab_sensors_config", None)
    
    
    if lab_sensors_config is None:
        lab_sensors_config = {}
    
        @dataclass
        class HSSDSceneDescriptionSensorConfig:
            type: str = "HSSDSceneDescriptionSensor"

        @dataclass
        class MP3DSceneDescriptionSensorConfig:
            type: str = "MP3DSceneDescriptionSensor"

        @dataclass
        class RobotResumeSensorConfig:
            type: str = "RobotResumeSensor"
            robot_resume_dir: str = "robot_resume"

        lab_sensors_config["robot_resume"] = RobotResumeSensorConfig()

        if "dataset" in sim.ep_info.info and sim.ep_info.info["dataset"] == "mp3d":
            lab_sensors_config["scene_description"] = MP3DSceneDescriptionSensorConfig()
            scene_description_sensor = MP3DSceneDescriptionSensor(
                sim, lab_sensors_config["scene_description"], *args, **kwargs
            )
        else:    
            lab_sensors_config["scene_description"] = HSSDSceneDescriptionSensorConfig()
            scene_description_sensor = HSSDSceneDescriptionSensor(
                sim, lab_sensors_config["scene_description"], *args, **kwargs
            )

        return {
            "scene_description": scene_description_sensor,
            "robot_resume": RobotResumeSensor(
                sim, lab_sensors_config["robot_resume"], *args, **kwargs
            )
        }

def get_text_context(sim, robot_configs: List[Dict], *args, **kwargs):
    
    text_sensors = get_text_sensors(sim, *args, **kwargs)
    text_context = {
        sensor_name: text_sensor.get_observation(
            robot_configs=robot_configs
        ) 
        for sensor_name, text_sensor in text_sensors.items()
    }
    return text_context

if __name__ == "__main__":
    env_config = {
        "agent_handles": ["FetchRobot", "SpotRobot"]
    }
    robot_resume_dir = "../data/robot_resume"
    robot_resume_sensor = RobotResumeSensor(env_config, robot_resume_dir)

    fetch_robot_resume = robot_resume_sensor.get_robot_resume("FetchRobot")
    print(fetch_robot_resume)
    
    spot_robot_resume = robot_resume_sensor.get_robot_resume("SpotRobot")
    print(spot_robot_resume)
    