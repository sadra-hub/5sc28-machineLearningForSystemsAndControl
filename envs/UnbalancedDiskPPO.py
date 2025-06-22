# This is the environment for the Unbalanced Disk problem used for PPO training and rendering.
# I have changed the reward function to be more suitable for PPO training.
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
from os import path

class UnbalancedDisk(gym.Env):
    def __init__(self, umax=3., dt=0.025, render_mode='human'):
        ############# start do not edit  ################
        self.omega0 = 11.339846957335382
        self.delta_th = 0
        self.gamma = 1.3328339309394384
        self.Ku = 28.136158407237073
        self.Fc = 6.062729509386865
        self.coulomb_omega = 0.001
        ############# end do not edit ###################

        self.umax = umax
        self.dt = dt
        self.action_space = spaces.Box(low=-umax, high=umax, shape=tuple())
        low = [-float('inf'), -40]
        high = [float('inf'), 40]
        self.observation_space = spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            shape=(2,)
        )

        self.render_mode = render_mode
        self.viewer = None
        self.u = 0
        self.reset()

    def step(self, action):
        self.u = np.clip(action, -self.umax, self.umax)

        def f(t, y):
            th, omega = y
            dthdt = omega
            friction = self.gamma * omega + self.Fc * np.tanh(omega / self.coulomb_omega)
            domegadt = -self.omega0 ** 2 * np.sin(th + self.delta_th) - friction + self.Ku * self.u
            return np.array([dthdt, domegadt])

        sol = solve_ivp(f, [0, self.dt], [self.th, self.omega])
        self.th, self.omega = sol.y[:, -1]

        # --- reward shaping ---
        theta = (self.th + np.pi) % (2 * np.pi) - np.pi  # normalize to [-π, π]
        omega = self.omega

        r_theta = np.cos(theta - np.pi)         # peak at θ = π
        r_omega = -0.05 * omega**2              # penalize rotation
        reward = r_theta + r_omega              # total reward
        reward = np.clip(reward, -2.0, 2.0)     # bound reward

        done = abs(theta) > np.pi * 1.1         # fail condition
        return self.get_obs(), reward, done, False, {}

    def reset(self, seed=None):
        self.th = np.random.uniform(-np.pi, np.pi)
        self.omega = np.random.normal(loc=0.0, scale=0.2)
        self.u = 0
        return self.get_obs(), {}

    def get_obs(self):
        self.th_noise = self.th + np.random.normal(loc=0, scale=0.001)
        self.omega_noise = self.omega + np.random.normal(loc=0, scale=0.001)

        th_norm = (self.th_noise + np.pi) % (2 * np.pi) - np.pi
        return np.array([th_norm, self.omega_noise], dtype=np.float32)

    def render(self):
        import pygame
        from pygame import gfxdraw
        screen_width, screen_height = 500, 500
        th = self.th
        if self.viewer is None:
            pygame.init()
            pygame.display.init()
            self.viewer = pygame.display.set_mode((screen_width, screen_height))
        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        gfxdraw.filled_circle(self.surf, screen_width // 2, screen_height // 2, int(screen_width / 2 * 0.65 * 1.3), (32, 60, 92))
        gfxdraw.filled_circle(self.surf, screen_width // 2, screen_height // 2, int(screen_width / 2 * 0.06 * 1.3), (132, 132, 126))
        from math import cos, sin
        r = screen_width // 2 * 0.40 * 1.3
        cx = int(screen_width // 2 - sin(th) * r)
        cy = int(screen_height // 2 - cos(th) * r)
        gfxdraw.filled_circle(self.surf, cx, cy, int(screen_width / 2 * 0.22 * 1.3), (155, 140, 108))
        gfxdraw.filled_circle(self.surf, cx, cy, int(screen_width / 2 * 0.22 / 8 * 1.3), (71, 63, 48))

        fname = path.join(path.dirname(__file__), "clockwise.png")
        self.arrow = pygame.image.load(fname)
        u = float(self.u) if not isinstance(self.u, (np.ndarray, list)) else float(self.u[0])
        arrow_size = abs(u / self.umax * screen_height) * 0.25
        Z = (arrow_size, arrow_size)
        arrow_rot = pygame.transform.scale(self.arrow, Z)
        if u < 0:
            arrow_rot = pygame.transform.flip(arrow_rot, True, False)

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.viewer.blit(self.surf, (0, 0))
        self.viewer.blit(arrow_rot, (screen_width // 2 - arrow_size // 2, screen_height // 2 - arrow_size // 2))
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
        return True

    def close(self):
        if self.viewer is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.viewer = None

class UnbalancedDisk_sincos(UnbalancedDisk):
    def __init__(self, umax=3., dt=0.025):
        super().__init__(umax=umax, dt=dt)
        low = [-1, -1, -40.]
        high = [1, 1, 40.]
        self.observation_space = spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            shape=(3,)
        )

    def get_obs(self):
        self.th_noise = self.th + np.random.normal(loc=0, scale=0.001)
        self.omega_noise = self.omega + np.random.normal(loc=0, scale=0.001)
        return np.array([np.sin(self.th_noise), np.cos(self.th_noise), self.omega_noise], dtype=np.float32)