from __future__ import annotations
import os
import datetime
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from hydra.core.hydra_config import HydraConfig

import gym.spaces as spaces
import numpy as np
import torch
from habitat_mas.agents.actions.discussion_actions import *
from habitat_mas.utils import AgentArguments
from habitat_mas.utils.models import OpenAIModel

from habitat_baselines.common.storage import Storage
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.multi_agent.pop_play_wrappers import (
    MultiAgentPolicyActionData,
    MultiPolicy,
    MultiStorage,
    MultiUpdater,
    _merge_list_dict,
)
from habitat_baselines.rl.hrl.hierarchical_policy import HierarchicalPolicy
from habitat_baselines.rl.hrl.hl.llm_policy import LLMHighLevelPolicy
from habitat_baselines.rl.multi_agent.utils import (
    add_agent_names,
    add_agent_prefix,
    update_dict_with_agent_prefix,
)

def dummy_agent_chat(env_text_context):
    return {}

class DummyMultiPolicy(MultiPolicy):
    """
    Wraps a set of LLM policies. Add group discussion stage before individual policy actions.
    """

    def __init__(self, update_obs_with_agent_prefix_fn, **kwargs):
        self._active_policies = []
        if update_obs_with_agent_prefix_fn is None:
            update_obs_with_agent_prefix_fn = update_dict_with_agent_prefix
        self._update_obs_with_agent_prefix_fn = update_obs_with_agent_prefix_fn

        # if True, the episode will terminate when all the agent choose to wait
        self.should_terminate_on_wait = kwargs.get("should_terminate_on_wait", False)

        # config for ablation study
        self.should_group_discussion = kwargs.get("should_group_discussion", True)
        self.should_agent_reflection = kwargs.get("should_agent_reflection", True)
        self.should_robot_resume = kwargs.get("should_robot_resume", True)
        self.should_numerical = kwargs.get("should_numerical", True)

    def set_active(self, active_policies):
        self._active_policies = active_policies

    def on_envs_pause(self, envs_to_pause):
        for policy in self._active_policies:
            policy.on_envs_pause(envs_to_pause)

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        envs_text_context=[{}],
        **kwargs,
    ):
        # debug info
        # TODO: disable saving chat history for batch experiments
        save_chat_history = kwargs.get("save_chat_history", False)
        save_chat_history_dir = kwargs.get("save_chat_history_dir", "./chat_history_output")
        # Create a directory to save chat history
        if save_chat_history:
            # save dir format: chat_history_output/<date>/<config>/<episode_id>
            date_str = datetime.datetime.now().strftime("%Y-%m-%d")
            hydra_cfg = HydraConfig.get()
            config_str = hydra_cfg.job.config_name.split("/")[-1].replace(".yaml", "")
            # episode_save_dir = os.path.join(save_chat_history_dir, date_str, config_str, str(episode_id))
            save_chat_history_dir = os.path.join(save_chat_history_dir, date_str, config_str)

            if not os.path.exists(save_chat_history_dir):
                os.makedirs(save_chat_history_dir)
            
        n_agents = len(self._active_policies)
        split_index_dict = self._build_index_split(
            rnn_hidden_states, prev_actions, kwargs
        )
        agent_rnn_hidden_states = rnn_hidden_states.split(
            split_index_dict["index_len_recurrent_hidden_states"], -1
        )
        agent_prev_actions = prev_actions.split(
            split_index_dict["index_len_prev_actions"], -1
        )
        agent_masks = masks.split([1] * n_agents, -1)
        n_envs = prev_actions.shape[0]
        assert n_envs == 1, "Currently, the stage 2 only supports single environment with multiple agents."

        text_goal = observations["pddl_text_goal"][0].tolist()
        text_goal = "".join([chr(code) for code in text_goal]).strip()
        # text_goal = "Fill in a tank with water. The tank is 80cm high, 50cm wide, and 50cm deep. The tank is empty. agent_0 has a 1L backet and agent_1 has a 2L bucket. Use function call to compute how many time agent_1 and agent_2 needs to fill the tank to full."
        # Stage 1: If all prev_actions are zero, which means it is the first step of the episode, then we need to do group discussion
        envs_chat_history = []
        for i in range(n_envs):
            env_prev_actions = prev_actions[i]
            env_text_context = envs_text_context[i]
            # if no previous actions, then it is the first step of the episode
            chat_result = dummy_agent_chat(env_text_context)
            envs_chat_history.append(chat_result)

        # Stage 2: Individual policy actions
        agent_actions = []
        for agent_i, policy in enumerate(self._active_policies):
            # collect assigned tasks for agent_i across all envs
            agent_i_handle = f"agent_{agent_i}"

            agent_obs = self._update_obs_with_agent_prefix_fn(observations, agent_i)

            # TODO: Currently, the stage 2 only supports single environment with multiple agents.
            # TODO: Please update the code of agent initialization and policy action to support vectorized environment.
            # Initialize action execution agent with new context information
            policy: HierarchicalPolicy
            llm_policy: LLMHighLevelPolicy = policy._high_level_policy
            if not llm_policy.llm_agent.initilized:
                episode_id = envs_text_context[0]["episode_id"]
                episode_save_dir = os.path.join(save_chat_history_dir, str(episode_id))
                logging_path = os.path.join(episode_save_dir, f"{agent_i_handle}_action_history.json")
                
                llm_policy.llm_agent.init_agent(
                    envs_chat_history,
                    enable_logging=save_chat_history,
                    logging_file=logging_path,
                )

            # Run the policy
            agent_actions.append(
                policy.act(
                    agent_obs,
                    agent_rnn_hidden_states[agent_i],
                    agent_prev_actions[agent_i],
                    agent_masks[agent_i],
                    deterministic,
                    envs_text_context=envs_text_context,
                    envs_chat_history=envs_chat_history,
                )
            )

        if self.should_terminate_on_wait:
            should_terminate = True
            for agent_id, agent_action in enumerate(agent_actions):
                wait_id = self._active_policies[agent_id]._name_to_idx['wait']
                if agent_action.skill_id != wait_id:
                    should_terminate = False
                    break
            if should_terminate:
                print("=================Terminate=================")
                print("All agents are waiting for next action.")
                print("===========================================")
                for agent_i, policy in enumerate(self._active_policies):
                    agent_actions[agent_i].actions[i, policy._stop_action_idx] = 1.0

        policy_info = _merge_list_dict(
            [ac.policy_info for ac in agent_actions]
        )
        batch_size = masks.shape[0]
        device = masks.device

        action_dims = split_index_dict["index_len_prev_actions"]

        # We need to split the `take_actions` if they are being assigned from
        # `actions`. This will be the case if `take_actions` hasn't been
        # assigned, like in a monolithic policy where there is no policy
        # hierarchicy.
        if any(ac.take_actions is None for ac in agent_actions):
            length_take_actions = action_dims
        else:
            length_take_actions = None

        def _maybe_cat(get_dat, feature_dims, dtype):
            all_dat = [get_dat(ac) for ac in agent_actions]
            # Replace any None with dummy data.
            all_dat = [
                torch.zeros((batch_size, feature_dims[ind]), device=device, dtype=dtype)
                if dat is None
                else dat
                for ind, dat in enumerate(all_dat)
            ]
            return torch.cat(all_dat, -1)

        rnn_hidden_lengths = [ac.rnn_hidden_states.shape[-1] for ac in agent_actions]
        return MultiAgentPolicyActionData(
            rnn_hidden_states=torch.cat(
                [ac.rnn_hidden_states for ac in agent_actions], -1
            ),
            actions=_maybe_cat(lambda ac: ac.actions, action_dims, prev_actions.dtype),
            values=_maybe_cat(
                lambda ac: ac.values, [1] * len(agent_actions), torch.float32
            ),
            action_log_probs=_maybe_cat(
                lambda ac: ac.action_log_probs,
                [1] * len(agent_actions),
                torch.float32,
            ),
            take_actions=torch.cat(
                [
                    ac.take_actions if ac.take_actions is not None else ac.actions
                    for ac in agent_actions
                ],
                -1,
            ),
            policy_info=policy_info,
            should_inserts=np.concatenate(
                [
                    ac.should_inserts
                    if ac.should_inserts is not None
                    else np.ones(
                        (batch_size, 1), dtype=bool
                    )  # None for monolithic policy, the buffer should be updated
                    for ac in agent_actions
                ],
                -1,
            ),
            length_rnn_hidden_states=rnn_hidden_lengths,
            length_actions=action_dims,
            length_take_actions=length_take_actions,
            num_agents=n_agents,
        )

    def _build_index_split(self, rnn_hidden_states, prev_actions, kwargs):
        """
        Return a dictionary with rnn_hidden_states lengths and action lengths that
        will be used to split these tensors into different agents. If the lengths
        are already in kwargs, we return them as is, if not, we assume agents
        have the same action/hidden dimension, so the tensors will be split equally.
        Therefore, the lists become [dimension_tensor // num_agents] * num_agents
        """
        n_agents = len(self._active_policies)
        index_names = [
            "index_len_recurrent_hidden_states",
            "index_len_prev_actions",
        ]
        split_index_dict = {}
        for name_index in index_names:
            if name_index not in kwargs:
                if name_index == "index_len_recurrent_hidden_states":
                    all_dim = rnn_hidden_states.shape[-1]
                else:
                    all_dim = prev_actions.shape[-1]
                split_indices = int(all_dim / n_agents)
                split_indices = [split_indices] * n_agents
            else:
                split_indices = kwargs[name_index]
            split_index_dict[name_index] = split_indices
        return split_index_dict

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks, **kwargs):
        split_index_dict = self._build_index_split(
            rnn_hidden_states, prev_actions, kwargs
        )
        agent_rnn_hidden_states = torch.split(
            rnn_hidden_states,
            split_index_dict["index_len_recurrent_hidden_states"],
            dim=-1,
        )
        agent_prev_actions = torch.split(
            prev_actions, split_index_dict["index_len_prev_actions"], dim=-1
        )
        agent_masks = torch.split(masks, [1, 1], dim=-1)
        all_value = []
        for agent_i, policy in enumerate(self._active_policies):
            agent_obs = self._update_obs_with_agent_prefix_fn(observations, agent_i)
            all_value.append(
                policy.get_value(
                    agent_obs,
                    agent_rnn_hidden_states[agent_i],
                    agent_prev_actions[agent_i],
                    agent_masks[agent_i],
                )
            )
        return torch.stack(all_value, -1)

    def get_extra(
        self, action_data: MultiAgentPolicyActionData, infos, dones
    ) -> List[Dict[str, float]]:
        all_extra = []
        for policy in self._active_policies:
            all_extra.append(policy.get_extra(action_data, infos, dones))
        # The action_data is shared across all policies, so no need to reutrn multiple times
        inputs = all_extra[0]
        ret: List[Dict] = []
        for env_d in inputs:
            ret.append(env_d)

        return ret

    @property
    def policy_action_space(self):
        # TODO: Hack for discrete HL action spaces.
        all_discrete = np.all(
            [
                isinstance(policy.policy_action_space, spaces.MultiDiscrete)
                for policy in self._active_policies
            ]
        )
        if all_discrete:
            return spaces.MultiDiscrete(
                tuple(
                    [policy.policy_action_space.n for policy in self._active_policies]
                )
            )
        else:
            return spaces.Dict(
                {
                    policy_i: policy.policy_action_space
                    for policy_i, policy in enumerate(self._active_policies)
                }
            )

    @property
    def policy_action_space_shape_lens(self):
        lens = []
        for policy in self._active_policies:
            if isinstance(policy.policy_action_space, spaces.Discrete):
                lens.append(1)
            elif isinstance(policy.policy_action_space, spaces.Box):
                lens.append(policy.policy_action_space.shape[0])
            else:
                raise ValueError(
                    f"Action distribution {policy.policy_action_space}" "not supported."
                )
        return lens

    @classmethod
    def from_config(
        cls,
        config,
        observation_space,
        action_space,
        update_obs_with_agent_prefix_fn: Optional[Callable] = None,
        **kwargs,
    ):
        kwargs["should_terminate_on_wait"] = config.habitat.dataset.should_terminate_on_wait
        kwargs["should_group_discussion"] = config.habitat.dataset.should_group_discussion
        kwargs["should_agent_reflection"] = config.habitat.dataset.should_agent_reflection
        kwargs["should_robot_resume"] = config.habitat.dataset.should_robot_resume
        kwargs["should_numerical"] = config.habitat.dataset.should_numerical

        return cls(update_obs_with_agent_prefix_fn, **kwargs)


class DummyMultiStorage(MultiStorage):
    def __init__(self, update_obs_with_agent_prefix_fn, **kwargs):
        super().__init__(update_obs_with_agent_prefix_fn, **kwargs)

    @classmethod
    def from_config(
        cls,
        config,
        observation_space,
        action_space,
        update_obs_with_agent_prefix_fn: Optional[Callable] = None,
        **kwargs,
    ):
        return cls(update_obs_with_agent_prefix_fn, **kwargs)


class DummyMultiUpdater(MultiUpdater):
    def __init__(self):
        self._active_updaters = []

    @classmethod
    def from_config(cls, config, observation_space, action_space, **kwargs):
        return cls()
