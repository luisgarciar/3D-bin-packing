""" Basic agents for 3D packing problems."""

import gym
from gym.spaces import dict
import numpy as np
from typing import List, Type, Tuple, Dict
from src.packing_kernel import Box, Container


def rnd_agent(observation: Dict) -> Dict:
    """Random agent for the packing environment.

    Args:
        observation (dict): Environment observation.

    Returns:
        action (dict): Action to be taken.
    """
    action = {"position": [0, 0], "box_index": 0}
    return action
