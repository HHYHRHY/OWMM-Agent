# ruff: noqa
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from habitat.tasks.rearrange.multi_task.pddl_action import PddlAction
from habitat_mas.agents.actions.arm_actions import *
from habitat_mas.agents.actions.base_actions import *
from habitat_mas.agents.crab_agent import CrabAgent

# TODO: replace dummy_agent with llm_agent

from habitat_baselines.rl.hrl.hl.high_level_policy import HighLevelPolicy
from habitat_baselines.rl.ppo.policy import PolicyActionData
from habitat_mas.utils import AgentArguments

ACTION_POOL = [send_request, nav_to_obj, pick, place, reset_arm, wait]


class LLMHighLevelPolicy(HighLevelPolicy):
    """
    High-level policy that uses an LLM agent to select skills.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._all_actions = self._setup_actions()
        self._n_actions = len(self._all_actions)
        self._active_envs = torch.zeros(self._num_envs, dtype=torch.bool)

        # environment_action_name_set = set(
        #     [action._name for action in self._all_actions]
        # )

        # llm_actions = [
        #     action
        #     for action in ACTION_POOL
        #     if action.name in environment_action_name_set
        # ]
        # Initialize the LLM agent
        self.llm_agent = self._init_llm_agent(kwargs["agent_name"], ACTION_POOL)

    def _init_llm_agent(self, agent_name, action_list):
        # Initialize the LLM agent here based on the config
        # This could load a pre-trained model, set up prompts, etc.
        # Return the initialized agent

        return CrabAgent(agent_name, action_list)

    def _parse_function_call_args(self, action_name, action_args: Dict) -> str:
        """
        Parse the arguments of a function call from the LLM agent to the policy input argument format.
        """
        if action_name == "place":
            return [action_args['target_obj'], action_args['target_location'], action_args['robot']]
        elif action_name == "pick":
            return [action_args['target_obj'], action_args['robot']]

        return action_args

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
        **kwargs,
    ) -> Tuple[torch.Tensor, List[Any], torch.BoolTensor, PolicyActionData]:
        """
        Get the next skill to execute from the LLM agent.
        """
        # TODO: use these text context to query the LLM agent with function call
        envs_text_context = kwargs.get("envs_text_context", None)
        agent_arguments: AgentArguments = kwargs.get("agent_arguments", None)
        if envs_text_context is None:
            raise ValueError("Environment text context not provided to the policy.")
        if agent_arguments is None:
            raise ValueError("Agent arguments not provided to the policy.")

        # print("=================env_text_context===================")
        # print(envs_text_context)
        # print("==================================================")

        start_action_prompt = (
            'You are just starting the task to take actions. '
            'Here is the current environment description: """\n{scene_description}\n"""\n\n'
            'Based on the task and environment, generate the most appropriate next action. \n'
            "Make sure that each action strictly adheres to the tool call's parameter list for that specific action. "
            r'Before providing the action, validate that both the action and its parameters conform exactly to the defined structure. '
            'Ensure that all required parameters are included and correctly formatted.'            
        )
        step_action_prompt = (
            'You have completed your previous action. '
            'Based on the task, generate the most appropriate next action. \n'
            "Make sure that each action strictly adheres to the tool call's parameter list for that specific action. "
            r'Before providing the action, validate that both the action and its parameters conform exactly to the defined structure. '
            'Ensure that all required parameters are included and correctly formatted.'
        )

        semantic_observation = envs_text_context[0]["scene_description"]
        # print(semantic_observation)
        if not self.llm_agent.start_act:
            get_next_action_message = start_action_prompt.format(
                scene_description=semantic_observation
            )
            self.llm_agent.start_act = True
        else:
            get_next_action_message = step_action_prompt

        batch_size = masks.shape[0]
        next_skill = torch.zeros(batch_size)
        skill_args_data = [None for _ in range(batch_size)]
        immediate_end = torch.zeros(batch_size, dtype=torch.bool)

        assert self.llm_agent.initialized, "Exception in LLMHighLevelPolicy.get_next_skill(): LLM agent not initialized."

        for batch_idx, should_plan in enumerate(plan_masks):
            if should_plan != 1.0:
                continue

            # Query the LLM agent with the current observations
            # to get the next action and arguments
            llm_output = self.llm_agent.chat(get_next_action_message)
            print("=================llm_output===================")
            print("Agent: ", self.llm_agent.name)
            print(llm_output)
            print("=================total token usage=======================")
            print("Agent: {} {}".format(self.llm_agent.name, self.llm_agent.get_token_usage()))
            print("==============================================")
            if llm_output is None:
                next_skill[batch_idx] = self._skill_name_to_idx["wait"]
                skill_args_data[batch_idx] = ["500"]
                continue

            action_name = llm_output["name"]
            action_args = self._parse_function_call_args(action_name, llm_output["arguments"])

            if action_name in self._skill_name_to_idx:
                next_skill[batch_idx] = self._skill_name_to_idx[action_name]
                skill_args_data[batch_idx] = action_args
            else:
                # If the action is not valid, do nothing
                next_skill[batch_idx] = self._skill_name_to_idx["wait"]
                skill_args_data[batch_idx] = ["500"]

        return (
            next_skill,
            skill_args_data,
            immediate_end,
            PolicyActionData(),
        )
