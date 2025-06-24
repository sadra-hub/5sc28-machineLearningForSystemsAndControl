from stable_baselines3 import A2C
import time
from envs.UnbalancedDiskA2C import UnbalancedDisk
import numpy as np
import imageio
import pygame
import cv2



model = A2C.load("./PTH/a2c_60000")

frames = []

env = UnbalancedDisk()
obs, info = env.reset()

try:
    for _ in range(400):
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        # Convert PyGame surface to NumPy RGB image
        frame = pygame.surfarray.array3d(env.surf)  # shape: (W, H, 3)
        frame = np.transpose(frame, (1, 0, 2))      # shape: (H, W, 3)
        frames.append(frame)

        time.sleep(1 / 20)
        if terminated or truncated:
            obs, info = env.reset()
finally:
    env.close()

    # Get dimensions from first frame
    height, width, _ = frames[0].shape

    # Create OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('unbalanced_disk.mp4', fourcc, 20, (width, height))

    # Write all frames
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
        out.write(frame_bgr)

    out.release()
    print("Video saved as unbalanced_disk.mp4 ✅")
