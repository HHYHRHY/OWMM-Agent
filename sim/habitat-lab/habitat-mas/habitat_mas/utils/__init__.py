from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentArguments:
    robot_id: str
    robot_type: str
    task_description: str
    subtask_description: str
    chat_history: list[dict[str, str]]
