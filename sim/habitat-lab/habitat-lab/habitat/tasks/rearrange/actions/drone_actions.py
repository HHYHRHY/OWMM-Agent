#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import os
from typing import Optional, cast

import magnum as mn
import numpy as np
from gym import spaces

import habitat_sim
from habitat.articulated_agents.robots.spot_robot import SpotRobot
from habitat.core.embodied_task import SimulatorTaskAction
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.rearrange.actions.articulated_agent_action import (
    ArticulatedAgentAction,
)

# flake8: noqa
# These actions need to be imported since there is a Python evaluation
# statement which dynamically creates the desired grip controller.
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.utils import rearrange_collision, rearrange_logger
from habitat_sim.physics import MotionType


@registry.register_task_action
class DroneBaseVelAction(ArticulatedAgentAction):
    """
    The articulated agent base motion in 3D space.
    """

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim: RearrangeSim = sim
        self.base_vel_ctrl = habitat_sim.physics.VelocityControl()
        self.base_vel_ctrl.controlling_lin_vel = True
        self.base_vel_ctrl.lin_vel_is_local = True
        self.base_vel_ctrl.controlling_ang_vel = True
        self.base_vel_ctrl.ang_vel_is_local = True
        self._allow_dyn_slide = self._config.get("allow_dyn_slide", True)
        self._lin_speed = self._config.lin_speed
        self._ang_speed = self._config.ang_speed
        self._allow_back = self._config.allow_back
            
    @property
    def action_space(self):
        lin_lim = 20
        ang_lim = 20
        return spaces.Dict([
            # linear velocity in x, y, z
            (self._action_arg_prefix + "lin_vel", 
             spaces.Box(shape=(3,), low=-lin_lim, high=lin_lim, dtype=np.float32)),
            # angular velocity in x, y, z
            (self._action_arg_prefix + "ang_vel", 
             spaces.Box(shape=(1,), low=-ang_lim, high=ang_lim, dtype=np.float32)),
        ])
        
    def _capture_articulated_agent_state(self):
        return {
            "forces": self.cur_articulated_agent.sim_obj.joint_forces,
            "vel": self.cur_articulated_agent.sim_obj.joint_velocities,
            "pos": self.cur_articulated_agent.sim_obj.joint_positions,
        }
        
    def _set_articulated_agent_state(self, set_dat):
        self.cur_articulated_agent.sim_obj.joint_positions = set_dat["forces"]
        self.cur_articulated_agent.sim_obj.joint_velocities = set_dat["vel"]
        self.cur_articulated_agent.sim_obj.joint_forces = set_dat["pos"]

    def step_filter(self, start_pos: mn.Vector3, end_pos: mn.Vector3) -> mn.Vector3:
        r"""Computes a valid navigable end point given a target translation on the NavMesh.
        Uses the configured sliding flag.

        :param start_pos: The valid initial position of a translation.
        :param end_pos: The target end position of a translation.
        """
        # try a step in the kinematic world and return the resulting state, also setting the proxy object to this state, syncrhonizing with agent
        start_orientation = self._sim.get_agent_state(self._agent_index).rotation
        self._sim.set_agent_state(
            end_pos, start_orientation, reset_sensors=False
        )
        if self.cur_articulated_agent.sim_obj.motion_type == MotionType.DYNAMIC and self.cur_articulated_agent.sim_obj.contact_test():
            self._sim.set_agent_state(
                start_pos, start_orientation, reset_sensors=False
            )
            return start_pos
        return end_pos

    def update_base(self):
        ctrl_freq = self._sim.ctrl_freq

        before_trans_state = self._capture_articulated_agent_state()

        trans = self.cur_articulated_agent.sim_obj.transformation
        rigid_state = habitat_sim.RigidState(
            mn.Quaternion.from_matrix(trans.rotation()), trans.translation
        )

        target_rigid_state = self.base_vel_ctrl.integrate_transform(
            1 / ctrl_freq, rigid_state
        )
        end_pos = self.step_filter(
            rigid_state.translation, target_rigid_state.translation
        )

        # try_step may fail, in which case it simply returns the start argument
        did_try_step_fail = end_pos == rigid_state.translation
        if not did_try_step_fail:
            # If try_step succeeded, it snapped our start position to the navmesh
            # We should apply the base offset
            end_pos -= self.cur_articulated_agent.params.base_offset

        target_trans = mn.Matrix4.from_(
            target_rigid_state.rotation.to_matrix(), end_pos
        )
        self.cur_articulated_agent.sim_obj.transformation = target_trans

        if not self._allow_dyn_slide:
            # Check if in the new articulated_agent state the arm collides with anything.
            # If so we have to revert back to the previous transform
            self._sim.internal_step(-1)
            colls = self._sim.get_collisions()
            did_coll, _ = rearrange_collision(
                colls, self._sim.snapped_obj_id, False
            )
            if did_coll:
                # Don't allow the step, revert back.
                self._set_articulated_agent_state(before_trans_state)
                self.cur_articulated_agent.sim_obj.transformation = trans
        if self.cur_grasp_mgr.snap_idx is not None:
            # Holding onto an object, also kinematically update the object.
            # object.
            self.cur_grasp_mgr.update_object_to_grasp()

    def step(self, *args, **kwargs):
        lin_vel = kwargs[self._action_arg_prefix + "lin_vel"]
        ang_vel = kwargs[self._action_arg_prefix + "ang_vel"]

        lin_vel = np.clip(lin_vel, -1, 1) * self._lin_speed
        ang_vel = np.clip(ang_vel, -1, 1) * self._ang_speed

        self.base_vel_ctrl.linear_velocity = mn.Vector3(
            lin_vel[0], lin_vel[1], lin_vel[2]
        )
        self.base_vel_ctrl.angular_velocity = mn.Vector3(0, ang_vel, 0)

        if (
            np.any(lin_vel != 0.0)
            or ang_vel != 0.0
        ):
            self.update_base()


