import json
from .crab_core import Action
from typing import List
import openai


class OpenAIModel:
    def __init__(
        self,
        system_prompt: str,
        action_space: List[Action],
        model="gpt-4o",
        window_size=None,
    ) -> None:
        self.system_message = {
            "role": "system",
            "content": system_prompt,
        }
        self._convert_action_to_schema(action_space)
        self.action_map = {action.name: action for action in action_space}
        self.chat_history = []
        self.window_size = window_size
        self.model = model
        self.client = openai.OpenAI()

    def chat(self, content: str):
        new_message = {"role": "user", "content": content}

        request = [self.system_message]
        # Add chat_history
        if self.window_size is None:
            for message in self.chat_history:
                request = request + message
        elif self.window_size > 0 and len(self.chat_history) > 0:
            for message in self.chat_history[-self.window_size :]:
                request = request + message

        request.append(new_message)
        self.chat_history.append([new_message])

        response = self.client.chat.completions.create(
            messages=request,  # type: ignore
            model=self.model,
            tools=[{"type": "function", "function": action} for action in self.actions],
            tool_choice="required",
        )

        response_message = response.choices[0].message
        self.chat_history[-1].append(response_message)

        tool_calls = response_message.tool_calls
        for tool_call in tool_calls:
            self.chat_history[-1].append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_call.function.name,
                    "content": "",
                }
            )  # extend conversation with function response

        if len(tool_calls) != 1:
            raise RuntimeError("agent output more than one action per step.")
        call = tool_calls[0]
        parameters = json.loads(call.function.arguments)
        return (call.function.name, parameters)
        # function_call_res = []
        # for call in tool_calls:
        #     action_name = call.function.name
        #     action = self.action_map[action_name]
        #     parameters = action.parameters(
        #         json.loads(call.function.arguments)
        #     ).model_dump()
        #     function_call_res.append((action_name, parameters))
        # return function_call_res

    def _convert_action_to_schema(self, action_space):
        self.actions = []
        for action in action_space:
            new_action = action.to_openai_json_schema()
            self.actions.append(new_action)
