# ruff: noqa
from typing import Any, Dict, List, Tuple

import torch
from habitat.tasks.rearrange.multi_task.pddl_action import PddlAction
from habitat_mas.agents.actions.arm_actions import *
from habitat_mas.agents.actions.base_actions import *
from habitat_mas.agents.crab_agent import CrabAgent

from habitat_mas.agents.dummy_agent import DummyAgentSingle

from habitat_baselines.rl.hrl.hl.high_level_policy import HighLevelPolicy
from habitat_baselines.rl.ppo.policy import PolicyActionData

class DummyPolicy(HighLevelPolicy):
    """
    High-level policy that uses an LLM agent to select skills.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._all_actions = self._setup_actions()
        self._n_actions = len(self._all_actions)
        self._active_envs = torch.zeros(self._num_envs, dtype=torch.bool)

        environment_action_name_set = set(
            [action._name for action in self._all_actions]
        )

        # Initialize the LLM agent
        self.llm_agent = self._init_llm_agent(**kwargs)

    def _init_llm_agent(self, **kwargs):
        return DummyAgentSingle(**kwargs) #now,is true dummy agent!

    def _parse_function_call_args(self, llm_args: Dict) -> str:
        """
        Parse the arguments of a function call from the LLM agent to the policy input argument format.
        """
        return llm_args

    def apply_mask(self, mask):
        """
        Apply the given mask to the agent in parallel envs.

        Args:
            mask: Binary mask of shape (num_envs, ) to be applied to the agents.
        """
        self._active_envs = mask

    def get_next_skill(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        plan_masks,
        deterministic,
        log_info,
        eval_info = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[Any], torch.BoolTensor, PolicyActionData]:
        """
        Get the next skill to execute from the LLM agent.
        """
        full_config = kwargs.get("full_config",None)
        # print(full_config)
        data_path = full_config['habitat']['dataset']['data_path']
        output_path = full_config['habitat_baselines']['image_dir']
        # print("obserations:",observations,flush = True)
        batch_size = masks.shape[0]
        next_skill = torch.zeros(batch_size)
        skill_args_data = [None for _ in range(batch_size)]
        immediate_end = torch.zeros(batch_size, dtype=torch.bool)
        # print("observations:",observations)
        for batch_idx, should_plan in enumerate(plan_masks):
            if should_plan != 1.0:
                continue
            # Query the LLM agent with the current observations
            # to get the next action and arguments
            # print("plan_mask:",plan_masks,flush=True)
            # vlm_output =
            # print("eval_info_dummypolicy",eval_info)
            # llm_output = self.llm_agent.agent_output(observations,eval_info)
            llm_output = self.llm_agent.vlm_eval_response(observations,data_path,output_path)
            if llm_output is None:
                next_skill[batch_idx] = self._skill_name_to_idx["wait"]
                skill_args_data[batch_idx] = ["50"]
                continue
            print("llm_output:",llm_output)
            action_name = llm_output["name"]
            action_args = self._parse_function_call_args(
                llm_output["arguments"])
            
            if action_name in self._skill_name_to_idx:
                next_skill[batch_idx] = self._skill_name_to_idx[action_name]
                skill_args_data[batch_idx] = action_args
            else:
                # If the action is not valid, do nothing
                next_skill[batch_idx] = self._skill_name_to_idx["wait"]
                skill_args_data[batch_idx] = ["50"]
        print(f"return_info:{next_skill};{skill_args_data}",flush =True)

        return (
            next_skill,
            skill_args_data,
            immediate_end,
            PolicyActionData(),
        )
