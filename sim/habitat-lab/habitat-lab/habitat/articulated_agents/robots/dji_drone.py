# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
import numpy as np

from habitat.articulated_agents.mobile_manipulator import (
    ArticulatedAgentCameraParams,
    MobileManipulator,
    MobileManipulatorParams,
)
from habitat_sim.physics import MotionType

class DJIDrone(MobileManipulator):
    """DJI Drone with a controllable base."""
    def _get_params(self, height: float=1.5):
        return MobileManipulatorParams(
            arm_joints=[],
            gripper_joints=[],
            wheel_joints=[],
            arm_init_params=np.array([], dtype=np.float32),
            gripper_init_params=np.array([], dtype=np.float32),
            ee_offset=[mn.Vector3(0, 0, 0)],
            ee_links=[0],
            ee_constraint=np.array([[[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]]], dtype=np.float32),
            cameras={                
                # "head": ArticulatedAgentCameraParams(
                #     cam_offset_pos=mn.Vector3(0.0, 0.0, 0.0),
                #     cam_look_at_pos=mn.Vector3(1.0, 0.0, 1.0),
                #     attached_link_id=5, # realsense_depth_optical_frame
                #     relative_transform=mn.Matrix4.rotation_y(mn.Deg(-90))
                #     @ mn.Matrix4.rotation_z(mn.Deg(180)),
                # ),
                "head": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(0.0, 0.0, 0.0),
                    cam_look_at_pos=mn.Vector3(1.0, -1.0, 0.0),
                    attached_link_id=-1,
                ),
                "third": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(-0.5, 1.7, -0.5),
                    cam_look_at_pos=mn.Vector3(1, 0.0, 0.75),
                    attached_link_id=-1,
                ),
                "top": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(0., 7.5, 0),
                    cam_look_at_pos=mn.Vector3(0, 0.0, 0.0),
                    attached_link_id=-2,
                ),
            },
            gripper_closed_state=np.array([], dtype=np.float32),
            gripper_open_state=np.array([], dtype=np.float32),
            gripper_state_eps=0.0,
            arm_mtr_pos_gain=0.0,
            arm_mtr_vel_gain=0.0,
            arm_mtr_max_impulse=0.0,
            wheel_mtr_pos_gain=0.0,
            wheel_mtr_vel_gain=0.0,
            wheel_mtr_max_impulse=0.0,
            base_offset=mn.Vector3(0, -height, 0),
            base_link_names={"m100_base_link"},
        )
    
    @property
    def base_transformation(self):
        add_rot = mn.Matrix4.rotation(
            mn.Rad(-np.pi / 2), mn.Vector3(1.0, 0, 0)
        )
        return self.sim_obj.transformation @ add_rot

    def __init__(
        self, agent_cfg, sim, limit_robo_joints=True, fixed_base=False
    ):
        drone_height = 1.5
        if sim.scene_type == "mp3d":
            drone_height = 0.8
        super().__init__(
            self._get_params(drone_height),
            agent_cfg,
            sim,
            limit_robo_joints,
            fixed_base,
        )
    
    def reconfigure(self) -> None:
        super().reconfigure()
        # Disable Drone's dynamics and collision by default 
        self.sim_obj.motion_type = MotionType.KINEMATIC
        # self.sim_obj.is_collidable = False # exception
        
        
    