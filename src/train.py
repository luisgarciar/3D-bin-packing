import glob
import warnings

import gym
from PIL import Image
from numpy.typing import NDArray
from sb3_contrib.ppo_mask import MaskablePPO

from src.utils import boxes_generator


def make_env(
    container_size,
    num_boxes,
    num_visible_boxes=1,
    seed=0,
    render_mode=None,
    random_boxes=False,
    only_terminal_reward=False,
):
    """
    Parameters

    ----------
    container_size: size of the container
    num_boxes: number of boxes to be packed
    num_visible_boxes: number of boxes visible to the agent
    seed: seed for RNG
    render_mode: render mode for the environment
    random_boxes: whether to use random boxes or not
    only_terminal_reward: whether to use only terminal reward or not
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
    return env


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    container_size = [10, 10, 10]
    box_sizes2 = [[3, 3, 3], [3, 2, 3], [3, 4, 2], [3, 2, 4], [3, 2, 3]]

    orig_env = gym.make(
        "PackingEnv-v0",
        container_size=container_size,
        box_sizes=box_sizes2,
        num_visible_boxes=1,
        render_mode="rgb_array",
        random_boxes=False,
        only_terminal_reward=False,
    )
    env = make_env(container_size, 5, 1, 0, "rgb_array", False, False)

    model = MaskablePPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=10)
    print("done training")
    model.save("ppo_mask")

    obs = env.reset()
    frames = []
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        frames.append(env.render(mode="rgb_array"))
    env.close()

    gif = frames[0]
    gif.save(
        "rollout",
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=100,
        loop=0,
    )
