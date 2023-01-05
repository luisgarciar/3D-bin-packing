from src.utils import boxes_generator
import gym
from gym import make
import warnings
from sb3_contrib.ppo_mask import MultiInputPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
import numpy as np
from numpy.typing import NDArray


def mask_fn(env: gym.Env) -> NDArray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.get_action_mask


# Ignore plotly and gym deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Environment initialization
container_size = [10, 10, 10]
box_sizes2 = [[3, 3, 3], [3, 2, 3], [3, 4, 2], [3, 2, 4], [3, 2, 3]]
env = make(
    "PackingEnv-v0",
    container_size=container_size,
    box_sizes=box_sizes2,
    num_visible_boxes=3,
    render_mode=None,
    options=None,
)
obs = env.reset()

# MaskablePPO initialization
# To configure the Maskable PPO agent, we need to wrap the environment
env = ActionMasker(env, mask_fn)  # Wrap to enable masking
model = MaskablePPO("MultiInputPolicy", env, gamma=0.4, verbose=1)
model.learn(5)
print("done training")
model.save("ppo_mask")
