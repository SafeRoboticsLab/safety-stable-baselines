"""GPU-resident vectorized-env contract for massive-parallel safety training.

Standard SB3 ``VecEnv`` is numpy end-to-end: with thousands of GPU-simulated
envs (mjlab / Isaac) every step pays a device->host->device bounce for obs,
rewards and infos, and the rollout buffer re-uploads at train time. The tensor
path keeps EVERYTHING on device:

    env.step_tensor(actions) -> (obs, reward_g, dones, timeouts, l_x)

all ``torch.Tensor`` on ``env.device``. The Safety* algorithms detect
``is_tensor_env`` and switch to a torch-native ``collect_rollouts`` +
:mod:`safety_sb3.tensor_buffers` (identical backup math, no numpy anywhere on
the hot path).

Reward semantics are unchanged: the reward IS the safety margin ``g(s)``;
``l_x`` is the target margin for reach-avoid (zeros for avoid-only envs).
Reward normalization does not exist on this path by design (it would corrupt
the Safety Bellman backup); observation normalization is provided by
:class:`TensorVecNormalize` (an rsl_rl-style running normalizer on device).

Subclass :class:`TensorVecEnv` for your simulator bridge; it also subclasses
SB3's ``VecEnv`` so BaseAlgorithm accepts it (the numpy step API raises).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


class TensorVecEnv(VecEnv):
  """Torch-native VecEnv. Implement ``reset`` and ``step_tensor``."""

  is_tensor_env = True
  render_mode = None

  def __init__(self, num_envs: int, observation_space: spaces.Space,
               action_space: spaces.Space, device: str = "cuda:0"):
    super().__init__(num_envs, observation_space, action_space)
    self.device = device

  # --- the tensor API ------------------------------------------------------
  def reset(self) -> th.Tensor:  # type: ignore[override]
    raise NotImplementedError

  def step_tensor(
    self, actions: th.Tensor
  ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """-> (obs, reward_g, dones, timeouts, l_x); all on ``self.device``.

    ``dones`` = terminated | truncated (env auto-resets internally);
    ``timeouts`` = truncated & ~terminated; ``l_x`` = target margin (zeros
    for avoid-only envs)."""
    raise NotImplementedError

  def metrics(self) -> dict[str, float]:
    """Optional per-rollout scalars (curriculum levels, task metrics) that the
    algorithm forwards into its logger. Never train on these."""
    return {}

  # --- numpy VecEnv API: not supported on the tensor path ------------------
  def step_async(self, actions):
    raise RuntimeError(
      "TensorVecEnv has no numpy step API — pair it with a safety_sb3 "
      "algorithm (SafetyPPO/ReachAvoidPPO detect is_tensor_env)."
    )

  def step_wait(self):
    raise RuntimeError("TensorVecEnv: use step_tensor().")

  def close(self) -> None:
    pass

  def _indices(self, indices):
    if indices is None:
      return range(self.num_envs)
    return [indices] if isinstance(indices, int) else indices

  def get_attr(self, attr_name, indices=None):
    return [getattr(self, attr_name, None) for _ in self._indices(indices)]

  def set_attr(self, attr_name, value, indices=None):
    setattr(self, attr_name, value)

  def env_method(self, method_name, *args, indices=None, **kwargs):
    return [None for _ in self._indices(indices)]

  def env_is_wrapped(self, wrapper_class, indices=None):
    return [False for _ in self._indices(indices)]


class TensorVecNormalize(TensorVecEnv):
  """Running observation normalizer on device (rsl_rl-parity).

  Wraps a :class:`TensorVecEnv`; normalizes observations with running
  mean/var updated during training (freeze with ``training=False``). There is
  deliberately NO reward normalization on this path — the reward is the
  physical margin g(s).
  """

  def __init__(self, venv: TensorVecEnv, epsilon: float = 1e-8,
               clip_obs: float = 100.0):
    super().__init__(venv.num_envs, venv.observation_space, venv.action_space,
                     venv.device)
    self.venv = venv
    self.training = True
    self.epsilon = float(epsilon)
    self.clip_obs = float(clip_obs)
    dim = int(np.prod(venv.observation_space.shape))
    dev = venv.device
    self.obs_mean = th.zeros(dim, device=dev)
    self.obs_var = th.ones(dim, device=dev)
    self.count = th.tensor(1e-4, device=dev)

  # --- normalization -------------------------------------------------------
  def _update(self, obs: th.Tensor) -> None:
    batch_mean = obs.mean(dim=0)
    batch_var = obs.var(dim=0, unbiased=False)
    batch_count = obs.shape[0]
    delta = batch_mean - self.obs_mean
    tot = self.count + batch_count
    self.obs_mean += delta * batch_count / tot
    m_a = self.obs_var * self.count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + delta.square() * self.count * batch_count / tot
    self.obs_var = m2 / tot
    self.count = tot

  def normalize_obs(self, obs: th.Tensor) -> th.Tensor:
    return th.clamp(
      (obs - self.obs_mean) / th.sqrt(self.obs_var + self.epsilon),
      -self.clip_obs, self.clip_obs,
    )

  def normalize_obs_np(self, obs: np.ndarray) -> np.ndarray:
    """Numpy convenience for eval/video harnesses."""
    t = th.as_tensor(obs, dtype=th.float32, device=self.device)
    return self.normalize_obs(t).cpu().numpy()

  # --- env API --------------------------------------------------------------
  def reset(self) -> th.Tensor:
    obs = self.venv.reset()
    if self.training:
      self._update(obs)
    return self.normalize_obs(obs)

  def step_tensor(self, actions: th.Tensor):
    obs, r, dones, timeouts, l_x = self.venv.step_tensor(actions)
    if self.training:
      self._update(obs)
    return self.normalize_obs(obs), r, dones, timeouts, l_x

  def metrics(self) -> dict[str, float]:
    return self.venv.metrics()

  def close(self) -> None:
    self.venv.close()

  # --- persistence ----------------------------------------------------------
  def save(self, path: str) -> None:
    th.save({"obs_mean": self.obs_mean, "obs_var": self.obs_var,
             "count": self.count}, path)

  @classmethod
  def load(cls, path: str, venv: TensorVecEnv,
           device: Optional[str] = None) -> "TensorVecNormalize":
    obj = cls(venv)
    state = th.load(path, map_location=device or venv.device, weights_only=True)
    obj.obs_mean = state["obs_mean"].to(venv.device)
    obj.obs_var = state["obs_var"].to(venv.device)
    obj.count = state["count"].to(venv.device)
    return obj
