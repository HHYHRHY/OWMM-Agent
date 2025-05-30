import os
from typing import List, Dict, Tuple, Union
import numpy as np
from habitat import Env

from habitat.tasks.rearrange.actions.oracle_nav_action import (
    OracleNavCoordinateAction,
    OracleNavAction
)

from ..crab_core import action

# TODO: to connect with the habitat-lab/ habitat-mas agents. 

@action
def get_agents() -> List[str]:
    """
    Retrieve the list of agents currently present in the environment.
    """
    pass

@action
def send_request(request: str, target_agent: str) -> str:
    """
    Send a text request to the fellow agents.
    
    Args:
        request: The text request to send.
        target_agent: The agent to send the request to.
    
    """
    pass

@action
def wait():
    """
    Wait if no immediate action is required from you.
    """
    pass

@action
def nav_to_obj(target_obj: str):
    """
    Navigate to the specified 'any_targets' or 'TARGET_any_target' object. Upon successfully completing this action, you should be positioned at the exact location of the 'any_targets' or 'TARGET_any_target'. 'target_obj' can NOT be a region name!
    
    Args:
        target_obj: The 'any_targets' or 'TARGET_any_targets' object to navigate to. 'target_obj' MUST follow the format 'any_targets|x' or 'TARGET_any_targets|x'. Must NOT be a region name.
    """ 
    pass

# @action
# def nav_to_goal(target_obj: str):
#     """
#     Navigate to the specified any_targets object. This action can only be executed if you are not currently holding or have not previously picked up the any_targets object. Upon successful completion, you should be positioned at the exact location of the any_targets.
    
#     Args:
#         target_obj: The any_targets object to navigate to. IMPORTANT: 'target_obj' must follow the format: 'any_targets|x', which is the unique name of the any_targets object, must NOT be a region name.
#     """
#     pass
