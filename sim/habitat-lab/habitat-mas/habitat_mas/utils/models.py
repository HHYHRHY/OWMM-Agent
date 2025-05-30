import json
import os
from ..agents.crab_core import Action
from typing import List
import openai
from .python_interpreter import SubprocessInterpreter
# from openai.types.chat.chat_completion import ChatCompletionMessage
# from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from pydantic import BaseModel

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, BaseModel):
            # Pydantic models are not serializable by json.dump by default
            return dict(obj)
        return super().default(obj)
        
class OpenAIModel:
    def __init__(
        self,
        system_prompt: str,
        action_space: List[Action],
        model="gpt-4o",
        window_size=None,
        discussion_stage=False,
        code_execution=False,
        enable_logging=False,
        logging_file="",
        save_on_each_chat=True,
        agent_name="unknown",
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
        self.planning_stage = discussion_stage
        self.code_execution = code_execution
        if self.code_execution:
            self.interpreter = SubprocessInterpreter()
        self.openai_tools = (
            [{"type": "function", "function": action} for action in self.actions]
            if action_space
            else None
        )
        self.tool_calls_enable = True if action_space else False
        self.token_usage = 0
        
        # Debug logging
        self.enable_logging = enable_logging
        self.logging_file = logging_file
        self.save_on_each_chat = save_on_each_chat
        self.agent_name = agent_name

    
    def __del__(self):
        """Save chat history to a file if logging is enabled"""
        if self.enable_logging:
            self.save_chat_history(self.logging_file)
           
    def save_chat_history(self, file_path: str):

        with open(file_path, "w") as f:
            full_history = [self.system_message] + self.chat_history
            json.dump(full_history, f, indent=2, cls=CustomJSONEncoder)
        
        episode_path = os.path.dirname(file_path)
        token_path = os.path.join(episode_path, "token_usage.json")
        if not os.path.exists(token_path):
            with open(token_path, 'w') as f:
                json.dump({}, f)

        with open(token_path, 'r') as f:
            data = json.load(f)
        data[f"{self.agent_name}"] = self.token_usage

        with open(token_path, 'w') as f:
            json.dump(data, f, indent=4)

    def set_system_message(self, system_message: str):
        self.system_message = {"role": "system", "content": system_message}

    def execute_action(self, action_name: str, parameters: dict) -> str:
        print("Internal action: ", action_name, parameters)
        return str(self.action_map[action_name].run(**parameters))

    def chat(self, content: str, crab_planning=False):
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

        if self.planning_stage:
            while True:
                if self.tool_calls_enable:
                    response = self.client.chat.completions.create(
                        messages=request,  # type: ignore
                        model=self.model,
                        tools=self.openai_tools,
                    )
                else:
                    response = self.client.chat.completions.create(
                        messages=request,  # type: ignore
                        model=self.model,
                    )
                self.token_usage += response.usage.total_tokens

                response_message = response.choices[0].message
                self.chat_history[-1].append(response_message)
                request.append(response_message)

                tool_calls = response_message.tool_calls
                codes = _extract_code(response_message.content)
                if self.tool_calls_enable and tool_calls is not None:
                    for tool_call in tool_calls:
                        tool_call_result = {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_call.function.name,
                            "content": self.execute_action(
                                tool_call.function.name,
                                json.loads(tool_call.function.arguments),
                            ),
                        }
                        self.chat_history[-1].append(tool_call_result)
                        request.append(tool_call_result)

                elif self.code_execution and codes:
                    # execution_results = []
                    # for code_block, code_type in codes:
                    #     execution_results.append(self.interpreter.run(code_block, code_type))
                    
                    # result_content = "\n".join(execution_results)
                    # print("============Codes============")
                    # for idx, code in enumerate(codes):
                    #     print(f"Code block {idx}: {code[0]}")
                    
                    merged_code = "\n".join([code_block for code_block, code_type in codes
                                             if code_type in self.interpreter._CODE_TYPE_MAPPING
                                             ])
                    result_content = self.interpreter.run(merged_code, "python")
                
                    print("============merged code============")
                    print(merged_code)
                    print("============Results============")
                    print(result_content)
                    print("==============================")
                    result_message = {"role": "user", "content": result_content}
                    self.chat_history[-1].append(result_message)
                    request.append(result_message)
                else:
                    if self.save_on_each_chat:
                        self.save_chat_history(self.logging_file)
                    return response_message.content
        elif crab_planning:
            while True:
                response = self.client.chat.completions.create(
                    messages=request,  # type: ignore
                    model=self.model,
                )
                self.token_usage += response.usage.total_tokens

                response_message = response.choices[0].message
                self.chat_history[-1].append(response_message)
                request.append(response_message)

                if self.save_on_each_chat:
                    self.save_chat_history(self.logging_file)
                return response_message.content
        else:
            response = self.client.chat.completions.create(
                messages=request,  # type: ignore
                model=self.model,
                tools=[
                    {"type": "function", "function": action} for action in self.actions
                ],
                tool_choice="required",
            )
            self.token_usage += response.usage.total_tokens

            response_message = response.choices[0].message
            self.chat_history[-1].append(response_message)
            tool_calls = response_message.tool_calls
            for idx, tool_call in enumerate(tool_calls):
                self.chat_history[-1].append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_call.function.name,
                        "content": "Success"
                        if idx == 0
                        else "Didn't execute because only the first tool call is executed",
                    }
                )  # extend conversation with function response

            # if len(tool_calls) != 1:
            #     print(tool_calls)
            #     raise RuntimeError("agent output more than one action per step.")
            call = tool_calls[0]
            parameters = json.loads(call.function.arguments)
            
            if self.save_on_each_chat:
                self.save_chat_history(self.logging_file)
            
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


def _extract_code(content) -> list[tuple[str, str]]:
    codes = []
    texts = []

    lines = content.split("\n")
    idx = 0
    start_idx = 0
    while idx < len(lines):
        while idx < len(lines) and (not lines[idx].lstrip().startswith("```")):
            idx += 1
        text = "\n".join(lines[start_idx:idx]).strip()
        texts.append(text)

        if idx >= len(lines):
            break

        code_type = lines[idx].strip()[3:].strip()
        idx += 1
        start_idx = idx
        while not lines[idx].lstrip().startswith("```"):
            idx += 1
        code = "\n".join(lines[start_idx:idx]).strip()
        codes.append((code, code_type))

        idx += 1
        start_idx = idx

    return codes
