from gym.envs.registration import register
from . import packing_kernel, packing_env, utils

register(id="PackingEnv-v0", entry_point="src.packing_env:PackingEnv")
