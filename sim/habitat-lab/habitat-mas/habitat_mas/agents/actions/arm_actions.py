import os
from typing import List, Dict, Tuple, Union
import numpy as np
from habitat import Env

from ..crab_core import action


@action
def pick(target_obj: str):
    """
    Pick up the 'any_targets' object. This action can only be executed under the following conditions: (1) You must not already be holding the 'any_targets' object, and (2) You must be exactly at the 'any_targets' position, which means you have to navigate to the 'any_targets' position first. After successfully executing this action, you will be holding the 'any_targets' object while remaining at the 'any_targets' position.
    
    Args:
        target_obj: The 'any_targets' object to pick.
    """
    pass


@action
def place(target_obj: str, target_location: str):
    """
    Place the 'any_targets' object at the 'TARGET_any_targets' position. This action can only be executed under the following conditions: (1) You must be holding the 'any_targets' object, and (2) You must be precisely at the 'TARGET_any_targets' position, which mean you have to navigate to the 'TARGET_any_targets' position first. After successfully executing this action, you will no longer be holding the 'any_targets' object, and it will be positioned at the 'TARGET_any_targets' position.
    
    Args:
        target_obj: The 'any_targets' object you are holding.
        target_location: The 'TARGET_any_targets' position to place the any_targets object.
    """
    pass


@action
def reset_arm():
    """
    Reset your arm to its default position. This action can only be executed if you have an arm and you MUST call this after performing pick-and-place actions.
    """
    pass
