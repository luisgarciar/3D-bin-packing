""" Basic agents for 3D packing problems."""

from typing import Dict


def rnd_agent(observation: Dict) -> Dict:
    """Return a simple deterministic action for the packing environment.

    Parameters
    ----------
    observation : Dict
        Current environment observation (unused by this baseline policy).

    Returns
    -------
    Dict
        Action dictionary with the selected `position` and `box_index`.
    """
    action = {"position": [0, 0], "box_index": 0}
    return action
