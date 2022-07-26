""" Basic agents for 3D packing problems."""

from typing import Dict


def rnd_agent(observation: Dict) -> Dict:
    """Random agent for the packing environment.

    Args:
        observation (dict): Environment observation.

    Returns:
        action (dict): Action to be taken.
    """
    action = {"position": [0, 0], "box_index": 0}
    return action
