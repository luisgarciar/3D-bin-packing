from gymnasium.envs.registration import register

register(id="PackingEnv-v0", entry_point="src.packing_env:PackingEnv")
