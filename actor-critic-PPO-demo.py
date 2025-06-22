# This script runs the last trained PPO model and renders the results in a gymnasium environment.
# Currently, it uses the policy from the "PTH/ppo_500" directory

import gymnasium as gym
import time
from stable_baselines3 import PPO
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "envs"))
from UnbalancedDiskPPO import UnbalancedDisk_sincos

# === Define a wrapper to make the environment compatible with SB3 ===
class DiskWrapper(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = UnbalancedDisk_sincos(dt=0.025, umax=3.0)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed=None, options=None):
        obs, _ = self.env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, False, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

# === Load the environment and model ===
env = DiskWrapper()
model = PPO.load("PTH/ppo_500", env=env)

# === Run the model and render the result ===
obs, _ = env.reset()

try:
    for i in range(500):
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        print(f"Step {i}: reward={reward:.3f}, action={action}")
        env.render()
        time.sleep(1/24)
        if done:
            obs, _ = env.reset()
finally:
    env.close()