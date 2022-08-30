""" Basic agents for 3D packing problems."""


import gym
from gym.spaces import dict
import numpy as np
from typing import List, Type, Tuple, Dict
from nptyping import NDArray, Int, Shape
from src.packing_engine import Box, Container
from src.packing_env import


def rnd_agent(observation -> Dict):
"""Random agent for the packing environment.

    Args:
        observation (dict): Environment observation.

    Returns:
        action (dict): Action to be taken.
    """
        action =
        action = {'position': [0, 0], 'box_index': 0}
        return action

