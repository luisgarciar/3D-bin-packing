import warnings

import gym
from numpy.typing import NDArray
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib.common.maskable.utils import get_action_masks

from src.utils import boxes_generator


def mask_fn(env: gym.Env) -> NDArray:
    return env.get_action_mask


def make_env(container_size, num_boxes, num_visible_boxes=1, seed=0):
    """
    Utility function for initializing bin packing env with action masking
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    env = gym.make(
        "PackingEnv-v0",
        container_size=container_size,
        box_sizes=boxes_generator(container_size, num_boxes, seed),
        num_visible_boxes=num_visible_boxes,
    )
    env = ActionMasker(env, mask_fn)
    return env


if __name__ == "__main__":
    # Ignore plotly and gym deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # Environment initialization
    container_size = [5, 5, 5]
    num_boxes = 10
    num_visible_boxes = 10
    num_env = 4
    env_kwargs = dict(
        container_size=container_size,
        num_boxes=num_boxes,
        num_visible_boxes=num_visible_boxes,
        seed=42,
        render_mode="rgb_array",
    )
    env = make_vec_env(make_env, n_envs=num_env, env_kwargs=env_kwargs)
    print("finished initialization of vectorized environment")
    print("beginning training")

    # MaskablePPO initialization
    model = MaskablePPO("MultiInputPolicy", env, gamma=0.4, verbose=1)
    checkpoint_callback = CheckpointCallback(
        save_freq=50, save_path="../logs/", name_prefix="rl_model"
    )
    model.learn(5000, callback=checkpoint_callback)
    print("done training")
    model.save("ppo_mask")

    obs = env.reset()
    while True:
        # Retrieve current action mask
        action_masks = get_action_masks(env)
        action, _states = model.predict(obs, action_masks=action_masks)
        obs, rewards, dones, info = env.step(action)
        env.render()
