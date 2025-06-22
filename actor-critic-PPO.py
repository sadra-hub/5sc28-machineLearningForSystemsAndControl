# This script used stable_baselines3 for training a PPO agent to learn how to balance a unbalanced disk. 
# The reward function is located the env/UnbalancedDiskPPO.py file, which is designed to encourage the agent
# to keep the disk upright while minimizing its angular velocity.

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "envs"))
from UnbalancedDiskPPO import UnbalancedDisk_sincos # type: ignore

# Wrap the custom environment for compatibility with SB3
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

# Create and validate environment
env = DiskWrapper()
check_env(env, warn=True)

# Monitor for logging
env = Monitor(env)

# Create PPO model
model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64, learning_rate=3e-4, gamma=0.99)

# Train the model
model.learn(total_timesteps=200_000)

# Save the model
model.save("ppo_unbalanced_disk")
print("\n✅ Model saved as 'ppo_unbalanced_disk.zip'")

# Evaluate and visualize
obs, _ = env.reset()
rewards = []

for _ in range(500):
    action, _states = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    rewards.append(reward)
    env.render()
    if done:
        obs, _ = env.reset()

# Plot the reward trend
plt.plot(np.cumsum(rewards))
plt.title("Cumulative Reward During Evaluation")
plt.xlabel("Step")
plt.ylabel("Cumulative Reward")
plt.grid(True)
plt.show()

# Close environment
env.close()