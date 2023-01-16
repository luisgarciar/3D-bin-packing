import warnings

import gym
from gym import make
from numpy.typing import NDArray
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from src.utils import boxes_generator


def mask_fn(env: gym.Env) -> NDArray:
    return env.get_action_mask


def make_env(
    container_size,
    num_boxes,
    num_visible_boxes=1,
    seed=0,
    render_mode="rgb_array",
    random_boxes=False,
    only_terminal_reward=False,
):
    """Utility function for initializing bin packing env with action masking
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    env = gym.make(
        "PackingEnv-v0",
        container_size=container_size,
        box_sizes=boxes_generator(container_size, num_boxes, seed),
        num_visible_boxes=num_visible_boxes,
        render_mode=render_mode,
        random_boxes=random_boxes,
        only_terminal_reward=only_terminal_reward,
    )
    env = ActionMasker(env, mask_fn)
    return env

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    container_size = [10, 10, 10]
    box_sizes2 = [[3, 3, 3], [3, 2, 3], [3, 4, 2], [3, 2, 4], [3, 2, 3]]
    env = make_env(container_size, 5, 1, 0, "rgb_array", False, False)

    model = MaskablePPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=10)
    print("done training")
    model.save("ppo_mask")

    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()

    env.close()
