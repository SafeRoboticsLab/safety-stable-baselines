"""GPU-resident batched BicycleGoal — the same task on the tensor path.

:class:`~safety_sb3.testing.bicycle5d_vec.BicycleGoalVec` batches the env with
numpy, which is fast per env-step but pins training to the CPU and to a device
bounce per step. This is the same env written in torch against
:class:`safety_sb3.tensor_env.TensorVecEnv`, so state, margins, obs, replay and
the backup all stay on one device:

    reset()              -> obs                       (n, obs_dim)
    step_tensor(actions) -> (obs, g, dones, timeouts, l_x)

Two reasons this matters for the SAC family specifically, and only one of them
is the device:

1. **No numpy on the hot path.** ``AbstractSAC`` detects ``is_tensor_env`` and
   swaps in ``_collect_rollouts_tensor`` + ``TensorReplayBuffer``.
2. **The update-to-data ratio is a function of ``num_envs``.** SB3 counts
   ``train_freq`` in VECTOR steps, so ``train_freq=(16, "step")`` means 16
   *vector* steps — ``16 * num_envs`` env-steps — per ``train()`` call. At 16
   envs a 2M-step budget buys 125,000 gradient steps; at 256 envs the same
   budget buys 7,812. Running few envs is therefore not "the same run, slower":
   it is a 16x higher UTD, and that (not just the device) is what made the
   16-env CPU SAC cells take an hour.

Physics, margins, obs layout, spawn modes and episode semantics are ported
verbatim from the numpy env and pinned by ``tests/test_bicycle5d_tensor.py``,
which drives both envs from an identical state with an identical action
sequence and compares trajectories. The numpy env stays the reference (it is
what ``bicycle5d_render``/``bicycle5d_demo`` evaluate against) — this is the
trainer, not a replacement.

Precision: ``dtype=th.float32`` by default (what the learners and buffers use).
``dtype=th.float64`` exists so the parity test can separate a real behavioral
difference from float32 round-off; it is not meant for training.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch as th

from ..tensor_env import TensorVecEnv
from .bicycle5d import (
  ACCEL_LIM, CAR_L, CAR_W, CLAMP_G, CLAMP_L, DEFAULT_GOAL, DEFAULT_OBSTACLES,
  DEFAULT_START, DELTA_LIM, DSTB_LIM, DT, GOAL_VALUE, G_SCALE, L_SCALE,
  OMEGA_LIM, V_MAX, V_MIN, WHEELBASE,
)


def _box_sdf(px: th.Tensor, py: th.Tensor, hx: float, hy: float) -> th.Tensor:
  """Torch twin of :func:`safety_sb3.testing.bicycle5d._box_sdf`."""
  qx, qy = px.abs() - hx, py.abs() - hy
  z = th.zeros((), dtype=px.dtype, device=px.device)
  outside = th.hypot(th.maximum(qx, z), th.maximum(qy, z))
  inside = th.minimum(th.maximum(qx, qy), z)
  return outside + inside


def _l_of_dist(d: th.Tensor, goal_r: th.Tensor) -> th.Tensor:
  """Torch twin of :func:`safety_sb3.testing.bicycle5d._l_of_dist` (piecewise:
  steep inside the goal for value range, gentle outside for a far gradient)."""
  inside = GOAL_VALUE * (goal_r - d) / goal_r
  outside = (goal_r - d) / L_SCALE
  return th.where(d <= goal_r, inside, outside)


class BicycleGoalTensorVec(TensorVecEnv):
  """Batched BicycleGoal on device. Same contract as ``BicycleGoalVec``.

  :param adversary: action becomes ``Box(2 + 5)`` — ``[0:2]`` control
      (accel, omega), ``[2:7]`` disturbance — matching the two-player contract
      (``ctrl_action_dim=2``). Use with ``SafetySAC2P`` / ``ReachAvoidSAC2P``.
  :param spawn: ``"edge"`` (near the start point), ``"wide"`` (the approach
      region, full y-height) or ``"map"`` (whole map). Same semantics and same
      ranges as the numpy env.
  :param device: torch device for state, margins and observations.
  :param dtype: state/margin precision. ``float32`` for training; ``float64``
      only to isolate round-off in the parity test.
  """

  def __init__(self, num_envs: int, adversary: bool = False,
               randomize: bool = True,
               obstacles: Sequence[Tuple[float, float, float]] = DEFAULT_OBSTACLES,
               goal: Tuple[float, float, float] = DEFAULT_GOAL,
               start: Tuple[float, float] = DEFAULT_START,
               timeout: int = 300, terminate_on_goal: bool = True,
               spawn: str = "edge", seed: Optional[int] = None,
               device: str = "cuda:0", dtype: th.dtype = th.float32):
    self.n = int(num_envs)
    self.adversary = bool(adversary)
    self.randomize = bool(randomize)
    self.spawn = spawn
    self.dtype = dtype
    self.timeout = int(timeout)
    self.terminate_on_goal = bool(terminate_on_goal)
    self.L = WHEELBASE

    obst0 = np.asarray(obstacles, dtype=np.float64).reshape(-1, 3)
    self.n_obs = len(obst0)
    obs_space = gym.spaces.Box(-np.inf, np.inf, (4 + 2 + 3 * self.n_obs,),
                               dtype=np.float32)
    if self.adversary:
      hi = np.concatenate([[ACCEL_LIM, OMEGA_LIM], DSTB_LIM]).astype(np.float32)
    else:
      hi = np.array([ACCEL_LIM, OMEGA_LIM], np.float32)
    act_space = gym.spaces.Box(-hi, hi, dtype=np.float32)
    super().__init__(self.n, obs_space, act_space, device)
    self.ctrl_action_dim = 2

    tk = dict(dtype=dtype, device=device)
    self._obst0 = th.as_tensor(obst0, **tk)
    self._goal0 = th.as_tensor(np.asarray(goal, dtype=np.float64), **tk)
    self._start = th.as_tensor(np.asarray(start, dtype=np.float64), **tk)
    self._dstb_lim = th.as_tensor(DSTB_LIM.astype(np.float64), **tk)
    self._u_lim = th.as_tensor([ACCEL_LIM, OMEGA_LIM], **tk)

    self.s = th.zeros(self.n, 5, **tk)                 # [x, y, v, psi, delta]
    self.obst = th.zeros(self.n, self.n_obs, 3, **tk)
    self.goal = th.zeros(self.n, 3, **tk)
    self.t = th.zeros(self.n, dtype=th.int64, device=device)

    self.rng = th.Generator(device=device)
    self.rng.manual_seed(0 if seed is None else int(seed))
    # rolling episode outcomes for metrics() (never trained on)
    self._m_reached = self._m_collided = self._m_done = 0

  # --- rng helper -----------------------------------------------------------
  def _u(self, lo: float, hi: float, *shape) -> th.Tensor:
    """``uniform(lo, hi)`` of the given shape, on device, in ``self.dtype``."""
    return th.rand(*shape, generator=self.rng, dtype=self.dtype,
                   device=self.device) * (hi - lo) + lo

  # --- batched physics ------------------------------------------------------
  def _deriv(self, s: th.Tensor, u: th.Tensor, d: th.Tensor) -> th.Tensor:
    v, psi, delta = s[:, 2], s[:, 3], s[:, 4]
    return th.stack([
      v * th.cos(psi) + d[:, 0],
      v * th.sin(psi) + d[:, 1],
      u[:, 0] + d[:, 2],
      v * th.tan(delta) / self.L + d[:, 3],
      u[:, 1] + d[:, 4],
    ], dim=1)

  def _dyn_step(self, u: th.Tensor, d: th.Tensor) -> None:
    s, dt = self.s, DT
    k1 = self._deriv(s, u, d)
    k2 = self._deriv(s + k1 * dt / 2, u, d)
    k3 = self._deriv(s + k2 * dt / 2, u, d)
    k4 = self._deriv(s + k3 * dt, u, d)
    s = s + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
    s[:, 2] = s[:, 2].clamp(V_MIN, V_MAX)
    s[:, 4] = s[:, 4].clamp(-DELTA_LIM, DELTA_LIM)
    s[:, 3] = (s[:, 3] + th.pi) % (2 * th.pi) - th.pi
    self.s = s

  # --- batched margins ------------------------------------------------------
  def _g(self) -> th.Tensor:
    """Signed distance from the car RECTANGLE to the nearest obstacle circle."""
    if self.n_obs == 0:
      return th.full((self.n,), float(CLAMP_G), dtype=self.dtype,
                     device=self.device)
    x, y, psi = self.s[:, 0], self.s[:, 1], self.s[:, 3]
    c, sn = th.cos(-psi)[:, None], th.sin(-psi)[:, None]        # (n, 1)
    dx = self.obst[:, :, 0] - x[:, None]                        # (n, n_obs)
    dy = self.obst[:, :, 1] - y[:, None]
    px = c * dx - sn * dy
    py = sn * dx + c * dy
    sd = _box_sdf(px, py, CAR_L / 2, CAR_W / 2) - self.obst[:, :, 2]
    return (sd.min(dim=1).values / G_SCALE).clamp(-CLAMP_G, CLAMP_G)

  def _l(self) -> th.Tensor:
    d = th.hypot(self.s[:, 0] - self.goal[:, 0], self.s[:, 1] - self.goal[:, 1])
    return _l_of_dist(d, self.goal[:, 2]).clamp(-CLAMP_L, CLAMP_L)

  def _obs(self) -> th.Tensor:
    x, y, v, psi, delta = (self.s[:, i] for i in range(5))
    c, sn = th.cos(-psi), th.sin(-psi)

    def to_car(gx, gy):
      dx, dy = gx - x, gy - y
      return c * dx - sn * dy, sn * dx + c * dy

    gx, gy = to_car(self.goal[:, 0], self.goal[:, 1])
    parts = [v, th.sin(psi), th.cos(psi), delta, gx, gy]
    for k in range(self.n_obs):
      rx, ry = to_car(self.obst[:, k, 0], self.obst[:, k, 1])
      parts += [rx, ry, self.obst[:, k, 2]]
    return th.stack(parts, dim=1)

  # --- reset (whole batch or a masked subset) -------------------------------
  def _reset_mask(self, m: th.Tensor) -> None:
    """Respawn the envs selected by the boolean mask ``m`` (n,)."""
    k = int(m.sum())
    if k == 0:
      return
    obst = self._obst0.unsqueeze(0).repeat(k, 1, 1)
    goal = self._goal0.unsqueeze(0).repeat(k, 1)
    if self.randomize and self.n_obs:
      obst[:, :, 0] += self._u(-0.30, 0.30, k, self.n_obs)
      obst[:, :, 1] += self._u(-0.35, 0.35, k, self.n_obs)
      obst[:, :, 2] += self._u(-0.05, 0.08, k, self.n_obs)
    if self.randomize:
      goal[:, 0] += self._u(-0.25, 0.25, k)
      goal[:, 1] += self._u(-0.50, 0.50, k)

    s = th.zeros(k, 5, dtype=self.dtype, device=self.device)
    if self.spawn == "cover":
      # CERTIFICATE-LEARNING coverage: uniform over the WHOLE reachable state
      # space -- every position, EVERY heading, full speed and steering range --
      # so the learned V̂ is defined everywhere the HJ oracle is (not just the
      # goal-facing approach cone the task spawns sample). Used for E063 twins.
      x = self._u(-0.6, 4.0, k)
      y = self._u(-1.4, 1.4, k)
      for _ in range(8):                   # reject spawns inside an obstacle
        if self.n_obs:
          dd = (th.hypot(obst[:, :, 0] - x[:, None], obst[:, :, 1] - y[:, None])
                - obst[:, :, 2]).min(dim=1).values
          bad = dd < 0.30
        else:
          bad = th.zeros(k, dtype=th.bool, device=self.device)
        if not bool(bad.any()):
          break
        nb = int(bad.sum())
        x[bad] = self._u(-0.6, 4.0, nb)
        y[bad] = self._u(-1.4, 1.4, nb)
      s[:, 0], s[:, 1] = x, y
      s[:, 2] = self._u(0.0, 2.0, k)                  # full speed range
      s[:, 3] = self._u(-3.14159265, 3.14159265, k)   # ALL headings
      s[:, 4] = self._u(-0.35, 0.35, k)               # full steering range
    elif self.spawn in ("map", "wide"):
      # Train where you eval. "wide" covers the APPROACH region (left side
      # through the obstacle band, FULL y-height); "map" runs x up to the goal
      # and injects degenerate near-goal spawns — kept for experiments only.
      x_hi = 2.6 if self.spawn == "map" else 1.3
      x = self._u(-0.2, x_hi, k)
      y = self._u(-1.1, 1.1, k)
      for _ in range(8):                   # reject spawns inside an obstacle
        if self.n_obs:
          dd = (th.hypot(obst[:, :, 0] - x[:, None], obst[:, :, 1] - y[:, None])
                - obst[:, :, 2]).min(dim=1).values
          bad = dd < 0.35
        else:
          bad = th.zeros(k, dtype=th.bool, device=self.device)
        if not bool(bad.any()):
          break
        nb = int(bad.sum())
        x[bad] = self._u(-0.2, x_hi, nb)
        y[bad] = self._u(-1.1, 1.1, nb)
      s[:, 0], s[:, 1] = x, y
      # heading toward the goal (+ jitter): keeps the goal in-frame from anywhere
      s[:, 3] = th.atan2(goal[:, 1] - y, goal[:, 0] - x) + self._u(-0.5, 0.5, k)
    else:                                  # "edge": near the start point
      s[:, 0] = self._start[0] + self._u(-0.15, 0.15, k)
      s[:, 1] = self._start[1] + self._u(-0.30, 0.30, k)
      s[:, 3] = self._u(-0.35, 0.35, k)
    if self.spawn != "cover":              # cover set v/delta above (full range)
      s[:, 2] = self._u(0.0, 0.4, k)
      s[:, 4] = self._u(-0.10, 0.10, k)
    self.obst[m], self.goal[m], self.s[m], self.t[m] = obst, goal, s, 0

  def reset(self) -> th.Tensor:
    self._reset_mask(th.ones(self.n, dtype=th.bool, device=self.device))
    return self._obs()

  # --- the tensor step ------------------------------------------------------
  def step_tensor(self, actions: th.Tensor):
    a = actions.to(dtype=self.dtype)
    u = th.clamp(a[:, :2], -self._u_lim, self._u_lim)
    d = (th.clamp(a[:, 2:7], -self._dstb_lim, self._dstb_lim) if self.adversary
         else th.zeros(self.n, 5, dtype=self.dtype, device=self.device))
    self._dyn_step(u, d)
    self.t += 1
    g, l = self._g(), self._l()
    reached = l >= 0.0
    collided = g < 0.0
    terminated = collided | (reached & self.terminate_on_goal)
    truncated = self.t >= self.timeout
    dones = terminated | truncated
    timeouts = truncated & ~terminated

    n_done = int(dones.sum())
    if n_done:
      self._m_done += n_done
      self._m_reached += int((reached & dones).sum())
      self._m_collided += int((collided & dones).sum())
      self._reset_mask(dones)              # auto-reset: obs below is post-reset
    return self._obs(), g, dones, timeouts, l

  # --- diagnostics ----------------------------------------------------------
  def metrics(self) -> dict[str, float]:
    """Episode outcome rates since the last call (logged, never trained on)."""
    if self._m_done == 0:
      return {}
    out = {"reach_rate": self._m_reached / self._m_done,
           "collide_rate": self._m_collided / self._m_done,
           "episodes": float(self._m_done)}
    self._m_reached = self._m_collided = self._m_done = 0
    return out
