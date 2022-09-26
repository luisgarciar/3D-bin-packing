import io

import numpy as np
from PIL import Image
from gym import make
from gym.envs.registration import register
from src.utils import boxes_generator


def plotly_fig2array(figure):
    # convert Plotly fig to  an array
    fig_bytes = figure.to_image(format="png")  # , width=600, height=450)
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)


if __name__ == "__main__":
    import warnings
    import os
    import sys

    p_dir = os.path.split(os.getcwd())[0]
    if p_dir not in sys.path:
        sys.path.append(p_dir)

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # Environment initialization
    register(id="PackingEnv", entry_point="src.packing_env:PackingEnv")

    env = make(
        "PackingEnv",
        new_step_api=False,
        container_size=[11, 11, 11],
        box_sizes=boxes_generator([10, 10, 10], num_items=80, seed=5),
        num_visible_boxes=1,
        render_mode="rgb_array",
    )

    # Run the random agent without saving the gif
    obs = env.reset()
    images = []
    for step_num in range(80):
        action_mask = obs["action_mask"]
        action = env.action_space.sample(action_mask)
        obs, reward, done, info = env.step(action)
        print(step_num)
        if done:
            break

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    fig = env.container.plot()
    fig.show()
