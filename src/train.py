from src.utils import boxes_generator
from gym import make
import warnings

from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks


# Ignore plotly and gym deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Environment initialization
env = make(
    "PackingEnv-v0",
    container_size=[10, 10, 10],
    box_sizes=boxes_generator([10, 10, 10], num_boxes=64, seed=42),
    num_visible_boxes=1,
)

obs = env.reset()


# To configure the Maskable PPO agent, we need to wrap the environment
# see


model = MaskablePPO("MultiInputPolicy", env, gamma=0.4, verbose=1)
model.learn(5000)
