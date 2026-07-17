"""Vectorized batched BicycleGoal — steps B envs at once with numpy.

A DummyVecEnv loops over envs in Python (throughput fixed by per-env cost);
SubprocVecEnv pays per-step IPC. For a pure-numpy env the win is to batch the
whole thing: state (B, 5), one RK4 over the batch, one margin computation. This
does tens of thousands of env-steps/s on CPU with no GPU and no processes.

Same physics, margins, obs, and contract as :class:`BicycleGoal` (numpy-parity
checked in tests) — this is the fast trainer for the same task, exposed as a
Stable-Baselines3 ``VecEnv`` so ReachAvoidPPO/SafetyPPO consume it directly.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from .bicycle5d import (
  ACCEL_LIM, CAR_L, CAR_W, CLAMP_G, CLAMP_L, DEFAULT_GOAL, DEFAULT_OBSTACLES,
  DEFAULT_START, DELTA_LIM, DSTB_LIM, DT, G_SCALE, L_SCALE, OMEGA_LIM,
  V_MAX, V_MIN, WHEELBASE, _box_sdf, _l_of_dist, BicycleGoal,
)
import gymnasium as gym


class BicycleGoalVec(VecEnv):
  """Batched BicycleGoal. ``num_envs`` bicycles stepped as numpy arrays."""

  def __init__(self, num_envs: int, adversary: bool = False,
               randomize: bool = True,
               obstacles: Sequence[Tuple[float, float, float]] = DEFAULT_OBSTACLES,
               goal: Tuple[float, float, float] = DEFAULT_GOAL,
               start: Tuple[float, float] = DEFAULT_START,
               timeout: int = 300, terminate_on_goal: bool = True,
               spawn: str = "edge", seed: Optional[int] = None):
    self.n = int(num_envs)
    self.adversary = bool(adversary)
    self.randomize = bool(randomize)
    #: "edge" spawns near the start point (the default, easy origin task);
    #: "map" spawns across the whole map so training matches the eval coverage.
    self.spawn = spawn
    self._obs0 = np.asarray(obstacles, dtype=np.float64).reshape(-1, 3)
    self._goal0 = np.asarray(goal, dtype=np.float64)
    self._start = np.asarray(start, dtype=np.float64)
    self.timeout = int(timeout)
    self.terminate_on_goal = bool(terminate_on_goal)
    self.L = WHEELBASE
    self.n_obs = len(self._obs0)
    self.rng = np.random.default_rng(seed)

    obs_space = gym.spaces.Box(-np.inf, np.inf, (4 + 2 + 3 * self.n_obs,),
                               dtype=np.float32)
    if self.adversary:
      hi = np.concatenate([[ACCEL_LIM, OMEGA_LIM], DSTB_LIM]).astype(np.float32)
    else:
      hi = np.array([ACCEL_LIM, OMEGA_LIM], np.float32)
    act_space = gym.spaces.Box(-hi, hi, dtype=np.float32)
    super().__init__(self.n, obs_space, act_space)
    self.ctrl_action_dim = 2

    self.s = np.zeros((self.n, 5))          # [x, y, v, psi, delta]
    self.obst = np.zeros((self.n, self.n_obs, 3))
    self.goal = np.zeros((self.n, 3))
    self.t = np.zeros(self.n, dtype=np.int64)
    self._actions = None

  # --- batched physics ------------------------------------------------------
  def _deriv(self, s, u, d):
    v, psi, delta = s[:, 2], s[:, 3], s[:, 4]
    out = np.empty_like(s)
    out[:, 0] = v * np.cos(psi) + d[:, 0]
    out[:, 1] = v * np.sin(psi) + d[:, 1]
    out[:, 2] = u[:, 0] + d[:, 2]
    out[:, 3] = v * np.tan(delta) / self.L + d[:, 3]
    out[:, 4] = u[:, 1] + d[:, 4]
    return out

  def _dyn_step(self, u, d):
    s, dt = self.s, DT
    k1 = self._deriv(s, u, d)
    k2 = self._deriv(s + k1 * dt / 2, u, d)
    k3 = self._deriv(s + k2 * dt / 2, u, d)
    k4 = self._deriv(s + k3 * dt, u, d)
    s = s + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
    s[:, 2] = np.clip(s[:, 2], V_MIN, V_MAX)
    s[:, 4] = np.clip(s[:, 4], -DELTA_LIM, DELTA_LIM)
    s[:, 3] = (s[:, 3] + np.pi) % (2 * np.pi) - np.pi
    self.s = s

  # --- batched margins ------------------------------------------------------
  def _g(self):
    if self.n_obs == 0:
      return np.full(self.n, CLAMP_G)
    x, y, psi = self.s[:, 0], self.s[:, 1], self.s[:, 3]
    c, sn = np.cos(-psi)[:, None], np.sin(-psi)[:, None]      # (n,1)
    dx = self.obst[:, :, 0] - x[:, None]                      # (n, n_obs)
    dy = self.obst[:, :, 1] - y[:, None]
    px = c * dx - sn * dy
    py = sn * dx + c * dy
    sd = _box_sdf(px, py, CAR_L / 2, CAR_W / 2) - self.obst[:, :, 2]
    return np.clip(sd.min(axis=1) / G_SCALE, -CLAMP_G, CLAMP_G)

  def _l(self):
    d = np.hypot(self.s[:, 0] - self.goal[:, 0], self.s[:, 1] - self.goal[:, 1])
    return np.clip(_l_of_dist(d, self.goal[:, 2]), -CLAMP_L, CLAMP_L)

  def _obs(self):
    x, y, v, psi, delta = (self.s[:, i] for i in range(5))
    c, sn = np.cos(-psi), np.sin(-psi)

    def to_car(gx, gy):
      dx, dy = gx - x, gy - y
      return c * dx - sn * dy, sn * dx + c * dy

    gx, gy = to_car(self.goal[:, 0], self.goal[:, 1])
    parts = [v, np.sin(psi), np.cos(psi), delta, gx, gy]
    for k in range(self.n_obs):
      rx, ry = to_car(self.obst[:, k, 0], self.obst[:, k, 1])
      parts += [rx, ry, self.obst[:, k, 2]]
    return np.stack(parts, axis=1).astype(np.float32)

  # --- reset (whole batch or a subset) --------------------------------------
  def _reset_idx(self, idx):
    m = len(idx)
    self.obst[idx] = self._obs0[None].repeat(m, 0)
    self.goal[idx] = self._goal0[None].repeat(m, 0)
    if self.randomize and self.n_obs:
      self.obst[idx, :, 0] += self.rng.uniform(-0.30, 0.30, (m, self.n_obs))
      self.obst[idx, :, 1] += self.rng.uniform(-0.35, 0.35, (m, self.n_obs))
      self.obst[idx, :, 2] += self.rng.uniform(-0.05, 0.08, (m, self.n_obs))
    if self.randomize:
      self.goal[idx, 0] += self.rng.uniform(-0.25, 0.25, m)
      self.goal[idx, 1] += self.rng.uniform(-0.50, 0.50, m)
    if self.spawn in ("map", "wide"):
      # Train where you eval. "wide" spawns across the APPROACH region (left
      # side through the obstacle band, FULL y-height) -- this covers the eval
      # coverage fringes (top/bottom cars) without the degenerate near-goal /
      # in-obstacle spawns that "map" (whole map, x up to the goal) injects and
      # that collapse training. "map" is kept for experiments only.
      x_hi = 2.6 if self.spawn == "map" else 1.3
      x = self.rng.uniform(-0.2, x_hi, m)
      y = self.rng.uniform(-1.1, 1.1, m)
      for _ in range(8):                       # reject spawns inside an obstacle
        if self.n_obs:
          dd = (np.hypot(self.obst[idx, :, 0] - x[:, None],
                         self.obst[idx, :, 1] - y[:, None])
                - self.obst[idx, :, 2]).min(axis=1)
          bad = dd < 0.35
        else:
          bad = np.zeros(m, bool)
        if not bad.any():
          break
        x[bad] = self.rng.uniform(-0.2, x_hi, bad.sum())
        y[bad] = self.rng.uniform(-1.1, 1.1, bad.sum())
      self.s[idx, 0], self.s[idx, 1] = x, y
      # heading toward the goal (+ jitter): keeps the goal in-frame from anywhere
      psi = np.arctan2(self.goal[idx, 1] - y, self.goal[idx, 0] - x)
      self.s[idx, 3] = psi + self.rng.uniform(-0.5, 0.5, m)
    else:                                      # "edge": near the start point
      self.s[idx, 0] = self._start[0] + self.rng.uniform(-0.15, 0.15, m)
      self.s[idx, 1] = self._start[1] + self.rng.uniform(-0.30, 0.30, m)
      self.s[idx, 3] = self.rng.uniform(-0.35, 0.35, m)
    self.s[idx, 2] = self.rng.uniform(0.0, 0.4, m)
    self.s[idx, 4] = self.rng.uniform(-0.10, 0.10, m)
    self.t[idx] = 0

  def reset(self):
    self._reset_idx(np.arange(self.n))
    return self._obs()

  # --- VecEnv step ----------------------------------------------------------
  def step_async(self, actions):
    self._actions = np.asarray(actions, dtype=np.float64)

  def step_wait(self):
    a = self._actions
    u = np.clip(a[:, :2], [-ACCEL_LIM, -OMEGA_LIM], [ACCEL_LIM, OMEGA_LIM])
    d = (np.clip(a[:, 2:7], -DSTB_LIM, DSTB_LIM) if self.adversary
         else np.zeros((self.n, 5)))
    self._dyn_step(u, d)
    self.t += 1
    g, l = self._g(), self._l()
    reached = l >= 0.0
    collided = g < 0.0
    terminated = collided | (reached & self.terminate_on_goal)
    truncated = self.t >= self.timeout
    dones = terminated | truncated

    obs = self._obs()
    infos = [{"l_x": float(l[i]), "reached": bool(reached[i]),
              "collided": bool(collided[i])} for i in range(self.n)]

    done_idx = np.nonzero(dones)[0]
    if len(done_idx):
      term_obs = obs[done_idx]                       # SB3 auto-reset convention
      for j, i in enumerate(done_idx):
        infos[i]["terminal_observation"] = term_obs[j]
        infos[i]["TimeLimit.truncated"] = bool(truncated[i] and not terminated[i])
      self._reset_idx(done_idx)
      obs[done_idx] = self._obs()[done_idx]          # return the reset obs
    return obs, g.astype(np.float32), dones, infos

  # --- VecEnv boilerplate ---------------------------------------------------
  def close(self):
    pass

  def get_attr(self, attr_name, indices=None):
    return [getattr(self, attr_name, None)] * self.n

  def set_attr(self, attr_name, value, indices=None):
    setattr(self, attr_name, value)

  def env_method(self, method_name, *args, indices=None, **kwargs):
    return [None] * self.n

  def env_is_wrapped(self, wrapper_class, indices=None):
    return [False] * self.n
