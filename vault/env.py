"""BalanceSafetyEnv — reach-avoid RL environment for the 4-state balance robot.

A Gymnasium env wrapping f_cert for SafetySAC (Fisac et al., reach-avoid RL): the reward IS
the safety margin g(s), and SafetySAC learns V(s) = min-over-trajectory g (the reach-avoid
value) plus a fallback policy pi_safe. The value filter (filter.py) then renders {V >= 0}
forward-invariant for the deployed controller.

  observation = [v, theta, theta_dot, psi_dot, mu]   (mu-aware value)
  action      = normalized [tau_L, tau_R] in [-1, 1]^2
  reward      = certified margin g (f_cert.margin, with a roll-wrench term under disturbance)
  terminated  = g < 0  (left the safe set)

Optional bounded disturbance (pitch/yaw-accel forcing + roll wrench) supports robust training.
For a *learned* adversary (ISAACS-style), train against the higher-fidelity `mujoco_plant`.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import config as C
from . import f_cert as F


class BalanceSafetyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, mu_range=C.MU_RANGE, tau_max=C.TAU_MAX, max_steps=200, disturb=True,
                 ebar_theta=0.1, ebar_psi=C.EBAR_PSI, tau_roll_bar=C.TAU_ROLL_BAR, seed=None):
        super().__init__()
        self.mu_range = mu_range
        self.tau_max = tau_max
        self.max_steps = max_steps
        self.disturb = disturb
        self.ebar_theta = ebar_theta
        self.ebar_psi = ebar_psi
        self.tau_roll_bar = tau_roll_bar
        hi = np.array([2.0, 1.5, 8.0, 3.0, 1.0], np.float32)
        self.observation_space = spaces.Box(-hi, hi, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)

    def _obs(self):
        return np.array([*self.x, self.mu], np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.mu = float(self._rng.uniform(*self.mu_range))
        self.x = np.array([self._rng.uniform(0.0, 1.5), self._rng.uniform(-0.5, 0.5),
                           self._rng.uniform(-2.0, 2.0), self._rng.uniform(-2.0, 2.0)])
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        u = np.clip(np.asarray(action, float), -1.0, 1.0) * self.tau_max
        x_next = F.f_cert_step(self.x, u, self.mu)
        tau_roll = 0.0
        if self.disturb:                                    # bounded forcing on post-step velocities
            x_next = x_next.copy()
            x_next[2] += self._rng.uniform(-self.ebar_theta, self.ebar_theta) * C.DT
            x_next[3] += self._rng.uniform(-self.ebar_psi, self.ebar_psi) * C.DT
            tau_roll = self._rng.uniform(-self.tau_roll_bar, self.tau_roll_bar)
        self.x = x_next
        self.t += 1
        g = float(F.margin(self.x))
        if tau_roll:                                        # roll wrench as effective lateral accel
            a_lat = abs(self.x[0] * self.x[3]) + abs(tau_roll) / (C.MASS * C.COM_H)
            g = float(min(g, 1.0 - a_lat / C.A_TIP))
        terminated = bool(g < 0.0)
        truncated = bool(self.t >= self.max_steps)
        return self._obs(), g, terminated, truncated, {"mu": self.mu, "g": g}
