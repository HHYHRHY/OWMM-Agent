import os
import habitat_sim
from habitat.config.default_structured_configs import ThirdRGBSensorConfig, HeadRGBSensorConfig
from habitat.config.default_structured_configs import SimulatorConfig, AgentConfig
from omegaconf import OmegaConf
from habitat.config.default_structured_configs import TaskConfig, EnvironmentConfig, DatasetConfig, HabitatConfig
from habitat.config.default_structured_configs import (   
    ArmActionConfig,
    BaseVelocityNonCylinderActionConfig,
    OracleNavActionConfig
)
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.core.env import Env

test_root = os.path.dirname(__file__)
data_root = os.path.join(test_root, "../../../data")

def make_sim_cfg(agent_dict):
    # Start the scene config
    sim_cfg = SimulatorConfig(type="RearrangeSim-v0")
    
    # Enable Horizon Based Ambient Occlusion (HBAO) to approximate shadows.
    sim_cfg.habitat_sim_v0.enable_hbao = True
    sim_cfg.habitat_sim_v0.enable_physics = True

    
    # Set up an example scene
    sim_cfg.scene = os.path.join(data_root, "hab3_bench_assets/hab3-hssd/scenes/103997919_171031233.scene_instance.json")
    sim_cfg.scene_dataset = os.path.join(data_root, "hab3_bench_assets/hab3-hssd/hab3-hssd.scene_dataset_config.json")
    sim_cfg.additional_object_paths = [os.path.join(data_root, 'objects/ycb/configs/')]
    
    
    cfg = OmegaConf.create(sim_cfg)

    # Set the scene agents
    cfg.agents = agent_dict
    cfg.agents_order = list(cfg.agents.keys())
    return cfg

def make_hab_cfg(agent_dict, action_dict):
    sim_cfg = make_sim_cfg(agent_dict)
    task_cfg = TaskConfig(type="RearrangeEmptyTask-v0")
    task_cfg.actions = action_dict
    env_cfg = EnvironmentConfig()
    dataset_cfg = DatasetConfig(type="RearrangeDataset-v0", data_path="data/datasets/hssd_height.json.gz")
    hab_cfg = HabitatConfig()
    hab_cfg.environment = env_cfg
    hab_cfg.task = task_cfg
    hab_cfg.dataset = dataset_cfg
    hab_cfg.simulator = sim_cfg
    hab_cfg.simulator.seed = hab_cfg.seed

    return hab_cfg

def init_rearrange_sim(agent_dict):
    # Start the scene config
    sim_cfg = make_sim_cfg(agent_dict)    
    cfg = OmegaConf.create(sim_cfg)
    
    # Create the scene
    sim = RearrangeSim(cfg)

    # This is needed to initialize the agents
    sim.agents_mgr.on_new_scene()

    # For this tutorial, we will also add an extra camera that will be used for third person recording.
    camera_sensor_spec = habitat_sim.CameraSensorSpec()
    camera_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    camera_sensor_spec.uuid = "scene_camera_rgb"

    # TODO: this is a bit dirty but I think its nice as it shows how to modify a camera sensor...
    sim.add_sensor(camera_sensor_spec, 0)

    return sim

def init_rearrange_env(agent_dict, action_dict):
    hab_cfg = make_hab_cfg(agent_dict, action_dict)
    res_cfg = OmegaConf.create(hab_cfg)
    return Env(res_cfg)


def get_fetch_hssd_env():
    # Define the agent configuration
    fetch_agent_config = AgentConfig()
    fetch_agent_config.articulated_agent_urdf = os.path.join(data_root, "robots/hab_fetch/robots/hab_fetch.urdf")
    fetch_agent_config.articulated_agent_type = "FetchRobot"
    fetch_agent_config.ik_arm_urdf = os.path.join(data_root,"robots/hab_fetch/robots/fetch_onlyarm.urdf")
    # Define sensors that will be attached to this agent, here a third_rgb sensor and a head_rgb.
    # We will later talk about why we are giving the sensors these names
    fetch_agent_config.sim_sensors = {
        "third_rgb": ThirdRGBSensorConfig(),
        "head_rgb": HeadRGBSensorConfig(),
    }
    agent_dict = {"main_agent": fetch_agent_config}
    action_dict = {
        "arm_ee_action": ArmActionConfig(
            arm_controller="ArmEEAction",
            grip_controller="MagicGraspAction",
            # grip_controller="SuctionGraspAction",
            # render_ee_target=True
        ),
        "base_velocity_non_cylinder_action": BaseVelocityNonCylinderActionConfig(),
        "oracle_coord_action": OracleNavActionConfig(type="OracleNavCoordinateAction", 
            spawn_max_dist_to_obj=1.0,
            motion_control="base_velocity_non_cylinder",
            navmesh_offset=[[0.0, 0.0], [0.25, 0.0], [-0.25, 0.0]]
        )
    }
    
    return init_rearrange_env(agent_dict, action_dict)
    
    
def get_spot_hssd_env():
    # Define the agent configuration
    spot_agent_config = AgentConfig()
    spot_agent_config.articulated_agent_urdf = os.path.join(data_root, "robots/hab_spot_arm/urdf/hab_spot_arm.urdf")
    spot_agent_config.articulated_agent_type = "SpotRobot"
    spot_agent_config.ik_arm_urdf = os.path.join(data_root, "robots/hab_spot_arm/urdf/hab_spot_onlyarm_dae.urdf")

    # Define sensors that will be attached to this agent, here a third_rgb sensor and a head_rgb.
    # We will later talk about why we are giving the sensors these names
    spot_agent_config.sim_sensors = {
        "third_rgb": ThirdRGBSensorConfig(),
        "head_rgb": HeadRGBSensorConfig(),
    }
    agent_dict = {"main_agent": spot_agent_config}
    action_dict = {
        "arm_ee_action": ArmActionConfig(
            arm_controller="ArmEEAction",
            grip_controller="MagicGraspAction",
            # grip_controller="SuctionGraspAction",
            # render_ee_target=True
        ),
        "base_velocity_non_cylinder_action": BaseVelocityNonCylinderActionConfig(),
        "oracle_coord_action": OracleNavActionConfig(type="OracleNavCoordinateAction", 
            spawn_max_dist_to_obj=1.0,
            motion_control="base_velocity_non_cylinder",
            navmesh_offset=[[0.0, 0.0], [0.25, 0.0], [-0.25, 0.0]]
        )
    }
    
    return init_rearrange_env(agent_dict, action_dict)