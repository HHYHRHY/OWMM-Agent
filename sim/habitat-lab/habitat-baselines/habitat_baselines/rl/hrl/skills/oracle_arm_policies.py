import os.path as osp
from dataclasses import dataclass

import torch
import numpy as np
from gym import spaces

from habitat.core.spaces import ActionSpace
from habitat.tasks.rearrange.rearrange_sensors import (
    IsHoldingSensor,
    ObjectToGoalDistanceSensor,
)
from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.rl.hrl.skills.nn_skill import NnSkillPolicy
from habitat_baselines.rl.hrl.utils import find_action_range
from habitat_baselines.rl.ppo.policy import PolicyActionData

RELEASE_ID = 0
PICK_ID = 1
PLACE_ID = 2

class OraclePickPolicy(NnSkillPolicy):
    """
    Skill to generate a picking motion. Moves the arm next to an object,
    """
    @dataclass
    class OraclePickActionArgs:
        """
        :property action_idx: The index of the oracle action we want to execute
        :property grab_release: Whether we want to grab (1) or drop an object (0)
        """
        action_idx: int
        grab_release: int

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
        super().__init__(
            wrap_policy,
            config,
            action_space,
            filtered_obs_space,
            filtered_action_space,
            batch_size,
        )

        action_name = "arm_pick_action"
        self._pick_srt_idx, self._pick_end_idx = find_action_range(action_space, action_name)

    def set_pddl_problem(self, pddl_prob): 
        super().set_pddl_problem(pddl_prob)
        self._all_entities = self._pddl_problem.get_ordered_entities_list()


    def on_enter(
        self,
        skill_arg,
        batch_idx,
        observations,
        rnn_hidden_states,
        prev_actions,
        skill_name,
    ): 
        self._is_target_obj = None
        self._targ_obj_idx = None
        self._prev_angle = {}

        ret = super().on_enter(
            skill_arg,
            batch_idx,
            observations,
            rnn_hidden_states,
            prev_actions,
            skill_name,
        )
        self._was_running_on_prev_step = False
        return ret

    @classmethod
    def from_config(
        cls, config, observation_space, action_space, batch_size, full_config
    ):         
        filtered_action_space = ActionSpace(
            {config.action_name: action_space[config.action_name]}
        )
        baselines_logger.debug(
            f"Loaded action space {filtered_action_space} for skill {config.skill_name}"
        )
        return cls(
            None,
            config,
            action_space,
            observation_space,
            filtered_action_space,
            batch_size,
            full_config.habitat.task.pddl_domain_def,
            osp.join(
                full_config.habitat.task.task_spec_base_path,
                full_config.habitat.task.task_spec + ".yaml",
            ),
            full_config.habitat.task,
        )
        
    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        batch_idx,
    ) -> torch.BoolTensor:
        is_holding = observations[IsHoldingSensor.cls_uuid].view(-1)
        return is_holding.type(torch.bool)

    def _parse_skill_arg(self, skill_name: str, skill_arg): 
        """
        Parses the object or container we should be picking or placing to.
        Uses the same parameters as oracle_nav.
        :param skill_arg: a pddl predicate specifying which object the pick action should target
        """
        if isinstance(skill_arg, dict):
            search_target = skill_arg["target_obj"]
        elif len(skill_arg) == 2:
            search_target, _ = skill_arg
        elif len(skill_arg) == 3:
            _, search_target, _ = skill_arg
        else:
            raise ValueError(
                f"Unexpected number of skill arguments in {skill_arg}"
            )

        target = self._pddl_problem.get_entity(search_target)
        if target is None:
            raise ValueError(
                f"Cannot find matching entity for {search_target}"
            )
        match_i = self._all_entities.index(target)

        return OraclePickPolicy.OraclePickActionArgs(match_i, PICK_ID)

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
        action_idxs = torch.FloatTensor(
            [self._cur_skill_args[i].action_idx for i in cur_batch_idx]
        )
        
        full_action[0][self._pick_srt_idx:self._pick_end_idx-1] = torch.tensor([action_idxs, PICK_ID])
        full_action[0][self._pick_end_idx-1] = 1.0
        # print("full_action:",full_action)
        return PolicyActionData(
            actions=full_action, rnn_hidden_states=rnn_hidden_states
        )
    
    def _get_coord_for_idx(self, object_target_idx):
        obj_entity = self._entities[object_target_idx]
        obj_pos = self._task.pddl_problem.sim_info.get_entity_pos(
            obj_entity
        )
        return obj_pos
    

class OraclePlacePolicy(OraclePickPolicy):
    """
    Skill to generate a placing motion. Moves the arm next to an object,
    """
    
    @dataclass
    class OraclePlaceActionArgs:
        """
        :property action_idx: The index of the oracle action we want to execute
        :property grab_release: Whether we want to grab (1) or drop an object (0)
        """
        action_idx: int
        grab_release: int

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
        super().__init__(
            wrap_policy,
            config,
            action_space,
            filtered_obs_space,
            filtered_action_space,
            batch_size,
            pddl_domain_path,
            pddl_task_path,
            task_config,
        )

        action_name = "arm_place_action"
        self._place_srt_idx, self._place_end_idx = find_action_range(action_space, action_name)

    def on_enter(
        self,
        skill_arg,
        batch_idx,
        observations,
        rnn_hidden_states,
        prev_actions,
        skill_name,
    ):
        self._is_target_obj = None
        self._targ_obj_idx = None
        self._prev_angle = {}

        ret = NnSkillPolicy.on_enter(
            self,
            skill_arg,
            batch_idx,
            observations,
            rnn_hidden_states,
            prev_actions,
            skill_name,
        )
        self._was_running_on_prev_step = False
        return ret

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
        return torch.tensor([rel_resting_pos < 0.02])

    def _parse_skill_arg(self, skill_name: str, skill_arg):
        if isinstance(skill_arg, dict):
            search_target = skill_arg["target_location"]
        elif len(skill_arg) == 2:
            search_target, _ = skill_arg
        elif len(skill_arg) == 3:
            _, search_target, _ = skill_arg
        else:
            raise ValueError(
                f"Unexpected number of skill arguments in {skill_arg}"
            )

        target = self._pddl_problem.get_entity(search_target)
        if target is None:
            raise ValueError(
                f"Cannot find matching entity for {search_target}"
            )
        match_i = self._all_entities.index(target)

        # Since the recep_idx might be 0, we encode the id by plus 1
        return OraclePlacePolicy.OraclePlaceActionArgs(
            match_i, RELEASE_ID
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
        action_idxs = torch.FloatTensor(
            [self._cur_skill_args[i].action_idx for i in cur_batch_idx]
        )

        full_action[0][self._place_srt_idx:self._place_end_idx-1] = torch.tensor([action_idxs, PLACE_ID])
        if self._is_skill_done(observations, rnn_hidden_states, prev_actions, masks, cur_batch_idx):
            full_action[0][self._place_end_idx-1] = -1.0

        return PolicyActionData(
            actions=full_action, rnn_hidden_states=rnn_hidden_states
        )
