from typing import List, Tuple, Dict, Union, Callable
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools.convert_to_openai import format_tool_to_openai_function
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field

# from habitat_mas.agents.llm_agent_base import LLMAgentBase


class OpenAIAgent:
    def __init__(self, client, model):
        self.client = client
        self.model = model
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "generate_plan",
                    "description": "Generate a plan",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "context": {
                                "type": "string",
                                "description": "The context for generating the plan",
                            },
                        },
                        "required": ["context"],
                    },
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run",
                    "description": "Run the plan",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "plan": {
                                "type": "string",
                                "description": "The plan to run",
                            },
                        },
                        "required": ["plan"],
                    },
                }
            },
        ]

    def generate_plan(self, context):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": context},
        ]
        response = self.chat_completion_request(messages, tool_choice={"type": "function", "function": {"name": "generate_plan"}})
        return response['choices'][0]['finish_reason']

    def run(self, plan):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": plan},
        ]
        response = self.chat_completion_request(messages, tool_choice={"type": "function", "function": {"name": "run"}})
        return response['choices'][0]['finish_reason']

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def chat_completion_request(self, messages, tool_choice=None):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice=tool_choice,
            )
            return response
        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            return e