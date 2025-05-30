'''
This file is used to define the rearrange actions with pixel on rgb observations as input.
'''
from habitat.tasks.rearrange.actions.actions import (
    ArticulatedAgentAction, 
    ArmEEAction
)
from habitat.tasks.rearrange.actions.oracle_arm_action import OraclePickAction
from habitat.articulated_agents.robots import (
    StretchRobot,
)
from habitat.tasks.rearrange.social_nav.oracle_social_nav_actions import OracleNavCoordAction
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.core.registry import registry
from gym import spaces
import numpy as np
import cv2
def _2d_to_3d_single_point(self, depth_obs, depth_rot,depth_trans,pixel_x, pixel_y):
        # depth_camera = self._sim._sensors[depth_name]._sensor_object.render_camera

        # hfov = float(self._sim._sensors[depth_name]._sensor_object.hfov) * np.pi / 180.
        # W, H = depth_camera.viewport[0], depth_camera.viewport[1]
        W = 512
        H = 512
        hfov = 1.5707963267948966
        # Intrinsic matrix K
        K = np.array([
            [1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., 1 / np.tan(hfov / 2.), 0., 0.],
            [0., 0., 1, 0],
            [0., 0., 0, 1]
        ])
        
        # Normalize pixel coordinates
        xs = (2.0 * pixel_x / (W - 1)) - 1.0  # normalized x [-1, 1]
        ys = 1.0 - (2.0 * pixel_y / (H - 1))  # normalized y [1, -1]

        # Depth value at the pixel
        depth = depth_obs[0,pixel_y,pixel_x,0]

        # Create the homogeneous coordinates for the pixel in camera space
        xys = np.array([xs * depth, ys * depth, -depth, 1.0]).reshape(4, 1)
        
        # Apply the inverse of the intrinsic matrix to get camera space coordinates
        xy_c = np.matmul(np.linalg.inv(K), xys)

        # Get the rotation and translation of the camera
        depth_rotation = np.array(depth_rot)
        depth_translation = np.array(depth_trans)

        # Get camera-to-world transformation
        T_world_camera = np.eye(4)
        T_world_camera[0:3, 0:3] = depth_rotation
        T_world_camera[0:3, 3] = depth_translation

        # Apply transformation to get world coordinates
        T_camera_world = np.linalg.inv(T_world_camera)
        points_world = np.matmul(T_camera_world, xy_c)

        # Get non-homogeneous points in world space
        points_world = points_world[:3, :] / points_world[3, :]
        return points_world.flatten()

def unproject_pixel_to_point(sim: RearrangeSim, sensor_name: str, depth_map: np.ndarray, pixel: tuple) -> np.ndarray:
    """
    Unprojects a pixel from the depth map to a 3D point in space.

    :param sim: RearrangeSim instance
    :param sensor_name: Name of the sensor
    :param depth_map: Depth map from the sensor
    :param pixel: (x, y) pixel coordinates
    :return: 3D point in space
    """
    depth_camera = sim._sensors[sensor_name]._sensor_object.render_camera

    hfov = float(sim._sensors[sensor_name]._sensor_object.hfov) * np.pi / 180.
    W, H = depth_camera.viewport[0], depth_camera.viewport[1]
    K = np.array([
        [1 / np.tan(hfov / 2.), 0., 0., 0.],
        [0., 1 / np.tan(hfov / 2.), 0., 0.],
        [0., 0., 1, 0],
        [0., 0., 0, 1]
    ])
    x, y = pixel
    x = int(x)
    y = int(y)
    depth = depth_map[y, x]
    xs = (2.0 * x / (W - 1)) - 1.0  # normalized x [-1, 1]
    ys = 1.0 - (2.0 * y / (H - 1))  # normalized y [1, -1]
    xys = np.array([xs * depth, ys * depth, -depth, 1])
    xy_c = np.matmul(np.linalg.inv(K), xys)

    depth_rotation = np.array(depth_camera.camera_matrix.rotation())
    depth_translation = np.array(depth_camera.camera_matrix.translation)
    # print("depth_rotation:",depth_rotation)
    # print("depth_translation:",depth_translation)
    T_world_camera = np.eye(4)
    T_world_camera[0:3, 0:3] = depth_rotation
    T_world_camera[0:3, 3] = depth_translation

    T_camera_world = np.linalg.inv(T_world_camera)
    point_world = np.matmul(T_camera_world, xy_c)
    # print("point_world:",point_world)
    return point_world[:3] / point_world[3]

@registry.register_task_action
class PixelArmAction(OraclePickAction):
    """
    Pick/Place action for the articulated_agent given the (x, y) pixel on the RGB image.
    Uses inverse kinematics (requires pybullet) to apply end-effector position control for the articulated_agent's arm.
    """

    def __init__(self, *args, sim: RearrangeSim, task, **kwargs):
        OraclePickAction.__init__(self, *args, sim=sim,task = task, **kwargs)
        self._task = task
        self._prev_ep_id = None
        self._hand_pose_iter = 0
        self.is_reset = False
        
        # camera parameters
        self.camera_type = kwargs.get("camera_type", "head")
        self.depth_camera_name = self._action_arg_prefix + f"{self.camera_type}_depth"

    def reset(self, *args, **kwargs):
        super().reset()

    @property
    def action_space(self):
        return spaces.Box(
            shape=(3,),
            low=0,
            high=10000,
            dtype=np.float32,
        )

    def step(self, arm_action, **kwargs):
        pixel_coord = arm_action[:2]
        action_type = arm_action[2]
        # if no action is specified, return the current end-effector position
        if action_type == 0:
            self.is_reset = False
            return self.ee_target            
        depth_obs = self._sim._prev_sim_obs[self.depth_camera_name].squeeze()
        target_coord = unproject_pixel_to_point(self._sim, self.depth_camera_name, depth_obs, pixel_coord)
        cur_ee_pos = self.cur_articulated_agent.ee_transform().translation
        if not self.is_reset:
            self.ee_target = self._ik_helper.calc_fk(self.cur_articulated_agent.arm_joint_pos)
            self.is_reset = True
        translation = target_coord - cur_ee_pos
        translation_base = self.cur_articulated_agent.base_transformation.inverted().transform_vector(translation)
        should_rest = False
        translation_base = np.clip(translation_base, -1, 1)
        self._ee_ctrl_lim = 0.03
        if isinstance(self.cur_articulated_agent, StretchRobot):
            self._ee_ctrl_lim = 0.06
        translation_base *= self._ee_ctrl_lim
        self.set_desired_ee_pos(translation_base)
        if self._render_ee_target:
            global_pos = self.cur_articulated_agent.base_transformation.transform_point(
                self.ee_target
            )
            self._sim.viz_ids["ee_target"] = self._sim.visualize_position(
                global_pos, self._sim.viz_ids["ee_target"]
            )

        return self.ee_target

@registry.register_task_action
class PixelNavAction(OracleNavCoordAction):
    """
    Navigate action for the articulated_agent given the (x, y) pixel on the RGB image.
    Uses inverse kinematics (requires pybullet) to apply end-effector position control for the articulated_agent's arm.
    """
    
    def __init__(self, *args, task, **kwargs):
        super().__init__(*args, task=task, **kwargs)
        
        # camera parameters
        self.camera_type = kwargs.get("camera_type", "head")
        self.depth_camera_name = self._action_arg_prefix + f"{self.camera_type}_depth"
    
    @property
    def action_space(self):
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "pixel_nav_action": spaces.Box(
                    shape=(2,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                )
            }
        )
    
    def step(self, pixel_nav_action, **kwargs):
        
        # if all pixels are zero, return
        if np.all(pixel_nav_action == 0):
            return
        print("flag!",flush = True)
        depth_obs = self._sim._prev_sim_obs[self.depth_camera_name].squeeze()
        # print("pixel_nav_action:",pixel_nav_action)
        # target_coord = unproject_pixel_to_point(self._sim, self.depth_camera_name, depth_obs, pixel_nav_action)
        target_coord = _2d_to_3d_single(self._sim, self.depth_camera_name, depth_obs, pixel_nav_action[0],pixel_nav_action[1])
        kwargs[self._action_arg_prefix + "oracle_nav_coord_action"] = target_coord
        return super().step(**kwargs)
