from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Callable, Tuple, Union

class LLMAgentBase:
    def __init__(self, name: str, description: str, llm: object, tools: List[object], 
                 output_parser: object, discussion_agents: List=None):
        self.name = name
        self.description = description
        self.llm = llm
        self.tools = tools
        self.output_parser = output_parser
        self.discussion_agents = discussion_agents or []
        self.executor = ThreadPoolExecutor()

    def generate_text(self, prompt: str, stop: List[str] = None, 
                      max_tokens: int = 100) -> str:
        """Generate text using the LLM."""
        future = self.executor.submit(self.llm, prompt, stop=stop, max_tokens=max_tokens)
        response = future.result()
        return response.generations[0][0].text

    def function_call(self, function: Callable, **kwargs) -> str:
        """Call a function and return its string output."""
        future = self.executor.submit(function, **kwargs)
        return str(future.result())

    def discuss(self, topic: str, max_iters: int = 3) -> str:
        """Engage in multi-agent discussion on a topic."""
        discussion = []
        for i in range(max_iters):
            futures = []
            for agent in [self] + self.discussion_agents:
                prompt = f"{agent.name}'s thoughts on '{topic}':\n\n"
                prompt += "\n".join(discussion[-3:]) 
                future = self.executor.submit(agent.generate_text, prompt)
                futures.append(future)
            
            for future in as_completed(futures):
                response = future.result()
                discussion.append(f"{agent.name}: {response}")
        
        return "\n".join(discussion)

    def plan(self, intermediate_steps: List[Tuple[Dict, str]], **kwargs) -> Union[List[Dict], Dict]:
        """Given input, decide what to do."""
        # Custom planning logic here
        pass

    def run(self, intermediate_steps: List[Tuple[Dict, str]], **kwargs) -> Dict:
        """Given input, run the agent."""
        # Custom run logic here
        pass