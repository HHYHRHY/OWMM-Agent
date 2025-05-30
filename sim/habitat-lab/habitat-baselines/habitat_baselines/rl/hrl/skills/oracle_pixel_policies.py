""" 
Comment by Junting: 
The logic behind the design of use OraclePixelXXXPolicy as a dummy policy to pass the arguments to PixelXXXAction classes:
Due to the multi-processing nature of habitat framework, the process/thread running policies/ agents are different from the process/thread running the simulator.
The thread running the simulator has the full access to GT information of agents and actions. 
Thus we put all the data processing logics into the PixelXXXAction classes, which have full access to simulator GT information.
"""
from dataclasses import dataclass
from typing import Tuple, List, Dict
import numpy as np
import torch
import magnum as mn
from habitat_baselines.rl.hrl.skills.nn_skill import NnSkillPolicy
from habitat_baselines.rl.hrl.skills.oracle_arm_policies import (
    OraclePickPolicy, 
    OraclePlacePolicy
)
from habitat.tasks.rearrange.rearrange_sensors import (
    IsHoldingSensor,
    ObjectToGoalDistanceSensor,
)
from habitat_baselines.rl.hrl.skills.oracle_nav import OracleNavPolicy
from habitat_baselines.rl.hrl.utils import find_action_range
from habitat_baselines.rl.ppo.policy import PolicyActionData

import numpy as np

RELEASE_ID = 0
PICK_ID = 1
PLACE_ID = 2
class OraclePixelPickPolicy(OraclePickPolicy):
    """
    Skill to generate a picking motion based on a pixel position.
    Moves the arm to the 3D position corresponding to the pixel on the RGB image.
    """

    @dataclass
    class OraclePixelPickActionArgs:
        """
        :property position: The (x, y) pixel position on the RGB image
        :property grab_release: Whether we want to grab (1) or drop an object (0)
        """
        position: List  # [x, y]
        grab_release: int

    def _parse_skill_arg(self, skill_name: str, skill_arg):
        """
        Parses the pixel position we should be picking at.
        :param skill_arg: a dictionary specifying the 'position' as [x, y]
        """
        if isinstance(skill_arg, dict) and "position" in skill_arg:
            position = skill_arg["position"]
        else:
            raise ValueError(f"Skill argument must include 'position', got {skill_arg}")

        if not isinstance(position, list) or len(position) != 2:
            raise ValueError(f"'position' must be a list of [x, y], got {position}")

        return OraclePixelPickPolicy.OraclePixelPickActionArgs(position=position, grab_release=PICK_ID)

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        full_action = torch.zeros(
            (masks.shape[0], self._full_ac_size), device=masks.device
        )
        positions = torch.FloatTensor(
            [self._cur_skill_args[i].position for i in cur_batch_idx]
        )
        # print("positions:",positions)
        full_action[:, self._pick_srt_idx:self._pick_srt_idx + 2] = positions
        # print("full_action1:",full_action)
        full_action[:, self._pick_end_idx-2] = torch.FloatTensor([PICK_ID] * masks.shape[0])
        # print("full_action2:",full_action)
        full_action[:, self._pick_end_idx-1] = torch.FloatTensor([PICK_ID] * masks.shape[0])
        full_action[:, self._grip_ac_idx] = torch.FloatTensor([PICK_ID] * masks.shape[0])
        return PolicyActionData(
            actions=full_action, rnn_hidden_states=rnn_hidden_states
        )
        
class OraclePixelPlacePolicy(OraclePlacePolicy):
    """
    Skill to generate a placing motion based on a pixel position.
    Moves the arm to the 3D position corresponding to the pixel on the RGB image.
    """

    @dataclass
    class OraclePixelPlaceActionArgs:
        """
        :property position: The (x, y) pixel position on the RGB image
        :property grab_release: Whether we want to grab (1) or drop an object (0)
        """
        position: list  # [x, y]
        grab_release: int

    def _parse_skill_arg(self, skill_name: str, skill_arg):
        """
        Parses the pixel position we should be placing at.
        :param skill_arg: a dictionary specifying the 'position' as [x, y]
        """
        if isinstance(skill_arg, dict) and "position" in skill_arg:
            position = skill_arg["position"]
        else:
            raise ValueError(f"Skill argument must include 'position', got {skill_arg}")

        if not isinstance(position, list) or len(position) != 2:
            raise ValueError(f"'position' must be a list of [x, y], got {position}")

        return OraclePixelPlacePolicy.OraclePixelPlaceActionArgs(position=position, grab_release=RELEASE_ID)
    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        batch_idx,
    ) -> torch.BoolTensor:
        rel_resting_pos = torch.linalg.vector_norm(
            observations[ObjectToGoalDistanceSensor.cls_uuid], dim=-1
        )
        
        # TODO: need to change the done condition
        return torch.tensor([rel_resting_pos < 0.5])
    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        full_action = torch.zeros(
            (masks.shape[0], self._full_ac_size), device=masks.device
        )
        positions = torch.FloatTensor(
            [self._cur_skill_args[i].position for i in cur_batch_idx]
        )
        
        # full_action[:, self._place_srt_idx:self._place_srt_idx + 2] = positions
        # full_action[:, self._place_end_idx-1] = -1.0
        # full_action[:, self._place_end_idx-2] = torch.FloatTensor([PLACE_ID] * masks.shape[0])

        # full_action[:, self._grip_ac_idx] = torch.FloatTensor([PLACE_ID] * masks.shape[0])
        # if self._is_skill_done(observations, rnn_hidden_states, prev_actions, masks, cur_batch_idx):
        #     full_action[0][self._place_end_idx-1] = -1.0
        full_action[:, self._place_srt_idx:self._place_srt_idx + 2] = positions
        # print("full_action1:",full_action)
        full_action[:, self._place_end_idx-2] = torch.FloatTensor([PLACE_ID] * masks.shape[0])
        # print("full_action2:",full_action)
        full_action[:, self._place_end_idx-1] = torch.FloatTensor([PLACE_ID] * masks.shape[0])
        
        # full_action[:, self._grip_ac_idx] = torch.FloatTensor([PLACE_ID] * masks.shape[0])
        if self._is_skill_done(observations, rnn_hidden_states, prev_actions, masks, cur_batch_idx):
            # full_action[:, self._grip_ac_idx] = torch.FloatTensor([PLACE_ID] * masks.shape[0])
            full_action[0][self._place_end_idx-1] = -1.0
            # full_action[:, self._place_end_idx-2] = -1.0

        return PolicyActionData(
            actions=full_action, rnn_hidden_states=rnn_hidden_states
        )
        
        
class OraclePixelNavPolicy(OracleNavPolicy):
    """
    Skill to pass navigation pixel target to PixelNavAction. 
    Move to the 3D position corresponding to the pixel on the RGB image.
    """
    @dataclass
    class OraclePixelNavActionArgs:
        """
        :property target_position: (2, ) The target position in pixel coordinates
        """
        target_position: List[float]
        camera_name: str = "head_rgb"
        

    def __init__(
        self,
        wrap_policy,
        config,
        action_space,
        filtered_obs_space,
        filtered_action_space,
        batch_size,
        pddl_domain_path,
        pddl_task_path,
        task_config,
    ):
        NnSkillPolicy.__init__(
            self,
            wrap_policy,
            config,
            action_space,
            filtered_obs_space,
            filtered_action_space,
            batch_size,
        )

        self._oracle_nav_ac_idx, _ = find_action_range(
            action_space, "oracle_nav_coord_action"
        )

    @property
    def required_obs_keys(self):
        return super().required_obs_keys + list(
            self._filtered_obs_space.spaces.keys()
        )

    def _parse_skill_arg(self, skill_name: str, skill_arg):
        # TODO: we should also put the camera name in the skill argument
        # if skill arg is a dictionary
        if isinstance(skill_arg, dict):
            target_position = skill_arg["target_position"]
            
        else:
            raise ValueError(
                f"Unexpected number of skill arguments in {skill_arg}"
            )

        return OraclePixelNavPolicy.OraclePixelNavActionArgs(
            target_position,
            camera_name="head_rgb"
        )

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        full_action = torch.zeros(
            (masks.shape[0], self._full_ac_size), device=masks.device
        )
        # TODO: parameterize the depth camera name
        target_positions_3d = [
            self._2d_to_3d_single(
                self._cur_skill_args[i].target_position, 
                observations['camera_info'][self._cur_skill_args[i].camera_name],
                observations['head_depth'],
                i
            )
            for i in cur_batch_idx
        ]
        print("2dpoint:",self._cur_skill_args[0].target_position)
        pixel_x, pixel_y = self._cur_skill_args[0].target_position
        # target_positions_3d = self._2d_to_3d_single_point(observations['depth_obs'].cpu(), observations['depth_rot'].cpu(),observations['depth_trans'].cpu(),pixel_x, pixel_y)
        target_positions_3d = torch.FloatTensor(target_positions_3d).view(masks.shape[0], 3)
        full_action[:, self._oracle_nav_ac_idx:self._oracle_nav_ac_idx + 3] = target_positions_3d
        print("3d_point:",target_positions_3d)
        return PolicyActionData(
            actions=full_action, rnn_hidden_states=rnn_hidden_states
        )
    
    def _2d_to_3d_single(self, target_position, camera_info, depth_obs, batch_idx):
        """
        Backproject the 2D pixel position to 3D world space.
        
        Args:
            target_position: (2, ) The target position in pixel coordinates
            camera_info: The camera information (batched)
            depth_obs: The depth observation (batched)
            batch_idx: The index of the batch
        """
        projection_matrix = camera_info['projection_matrix'][batch_idx].cpu().numpy()
        camera_matrix = camera_info['camera_matrix'][batch_idx].cpu().numpy()
        hfov = camera_info['fov'][batch_idx].item()
        depth_obs = depth_obs[batch_idx, ...].squeeze()

        hfov_rad = float(hfov)
        W, H = projection_matrix.shape[1], projection_matrix.shape[0]
        W = 512
        H = 512
        K = np.array([
            [1 / np.tan(hfov_rad / 2.0), 0.0, 0.0, 0.0],
            [0.0, 1 / np.tan(hfov_rad / 2.0), 0.0, 0.0],
            [0.0, 0.0, 1, 0],
            [0.0, 0.0, 0, 1]
        ])
        # print("W,H,hfov",W,H,hfov)
        x, y = target_position
        x = int(x)
        y = int(y)
        depth_value = depth_obs[y, x].item()

        # Normalize x and y to the range [-1, 1]
        x_normalized = (2.0 * x / W) - 1.0
        y_normalized = 1.0 - (2.0 * y / H)

        # Create the xys array for a single point
        xys = np.array([x_normalized * depth_value, y_normalized * depth_value, -depth_value, 1.0])

        # Transform to camera coordinates
        xy_c = np.matmul(np.linalg.inv(K), xys)

        depth_rotation = np.array(camera_matrix[:3, :3])
        depth_translation = np.array(camera_matrix[:3, 3])

        # Get camera-to-world transformation
        T_world_camera = np.eye(4)
        T_world_camera[0:3, 0:3] = depth_rotation
        T_world_camera[0:3, 3] = depth_translation

        T_camera_world = np.linalg.inv(T_world_camera)
        point_world = np.matmul(T_camera_world, xy_c)

        # Get non-homogeneous point in world space
        point_world = point_world[:3] / point_world[3]
        # print("point_world:",point_world)
        return point_world
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
        depth = depth_obs[0, pixel_x,pixel_y,0]
        
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
