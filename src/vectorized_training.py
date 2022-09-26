import warnings
import gym
from numpy.typing import NDArray
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import os


from src.utils import boxes_generator


def mask_fn(env: gym.Env) -> NDArray:
    return env.get_action_mask


def make_env(
    container_size, num_boxes, num_visible_boxes=1, seed=0, render_mode="human"
):
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
        render_mode=render_mode,
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
    num_env = 2
    env_kwargs = dict(
        container_size=container_size,
        num_boxes=num_boxes,
        num_visible_boxes=num_visible_boxes,
        render_mode="rgb_array",
        seed=42,
    )
    env = make_vec_env(make_env, n_envs=num_env, env_kwargs=env_kwargs)
    print("finished initialization of vectorized environment")
    print("beginning training")

    # MaskablePPO initialization
    model = MaskablePPO("MultiInputPolicy", env, gamma=0.4, verbose=1)
    checkpoint_callback = CheckpointCallback(
        save_freq=10, save_path="../logs/", name_prefix="rl_model"
    )
    model.learn(20, callback=checkpoint_callback)
    print("done training")
    model.save("../models/ppo_mask_cont555_boxes10_vis10_steps_200_num_env_2")
    print("saved model")
    del model

    model = MaskablePPO.load("ppo_mask_cont555_boxes10_vis10_steps_200_numenv_2")
    print("done loading")
    seed = 33
    render_mode = "rgb_array"
    eval_env = gym.make(
        "PackingEnv-v0",
        container_size=container_size,
        box_sizes=boxes_generator(container_size, num_boxes, seed),
        num_visible_boxes=num_visible_boxes,
        render_mode=render_mode,
    )

    log_dir = "../eval/"
    os.makedirs(log_dir, exist_ok=True)
    eval_env = Monitor(eval_env, log_dir)

    print("beginning evaluation")
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=2,
        return_episode_rewards=True,
    )
    print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")
