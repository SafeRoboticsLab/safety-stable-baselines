"""GPU-resident rollout buffers with the safety backups (torch twins of
:mod:`safety_sb3.safety_buffers` — identical math, no numpy on the hot path).

Both call the same operators as the numpy buffers, from
:mod:`safety_sb3.backups`; ``tests/test_backups.py`` asserts exact parity.
``get()`` yields standard ``RolloutBufferSamples`` whose fields are
device tensors, so stock ``PPO.train()`` consumes them unchanged. ``values`` /
``returns`` are exposed as numpy properties (PPO's explained-variance logging
touches them once per update).
"""

from __future__ import annotations

from typing import Generator, Optional

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBufferSamples

from . import backups


class TensorSafetyRolloutBuffer:
  """Torch rollout buffer with the Safety Bellman backup."""

  is_tensor_buffer = True

  def __init__(self, buffer_size: int, observation_space: spaces.Space,
               action_space: spaces.Space, device: str = "cuda:0",
               gae_lambda: float = 0.95, gamma: float = 0.99,
               n_envs: int = 1, **_ignored):
    self.buffer_size = int(buffer_size)
    self.n_envs = int(n_envs)
    self.device = device
    self.gamma = float(gamma)
    self.gae_lambda = float(gae_lambda)
    self.obs_dim = int(np.prod(observation_space.shape))
    self.act_dim = int(np.prod(action_space.shape))
    self.reset()

  def reset(self) -> None:
    T, N, dev = self.buffer_size, self.n_envs, self.device
    self.observations = th.zeros(T, N, self.obs_dim, device=dev)
    self.actions = th.zeros(T, N, self.act_dim, device=dev)
    self.rewards = th.zeros(T, N, device=dev)
    self.episode_starts = th.zeros(T, N, device=dev)
    self._values = th.zeros(T, N, device=dev)
    self.log_probs = th.zeros(T, N, device=dev)
    self.advantages = th.zeros(T, N, device=dev)
    self._returns = th.zeros(T, N, device=dev)
    self.pos = 0
    self.full = False

  # numpy views for PPO.train()'s explained-variance logging.
  @property
  def values(self) -> np.ndarray:
    return self._values.detach().cpu().numpy()

  @property
  def returns(self) -> np.ndarray:
    return self._returns.detach().cpu().numpy()

  def add(self, obs: th.Tensor, actions: th.Tensor, rewards: th.Tensor,
          episode_starts: th.Tensor, values: th.Tensor,
          log_probs: th.Tensor) -> None:
    p = self.pos
    self.observations[p] = obs.reshape(self.n_envs, self.obs_dim)
    self.actions[p] = actions.reshape(self.n_envs, self.act_dim)
    self.rewards[p] = rewards.reshape(self.n_envs)
    self.episode_starts[p] = episode_starts.reshape(self.n_envs).float()
    self._values[p] = values.reshape(self.n_envs)
    self.log_probs[p] = log_probs.reshape(self.n_envs)
    self.pos += 1
    if self.pos == self.buffer_size:
      self.full = True

  # --- backup ---------------------------------------------------------------
  def _target(self, step: int, v_next: th.Tensor,
              not_done: th.Tensor) -> th.Tensor:
    return backups.avoid_target(self.rewards[step], v_next, not_done,
                                self.gamma)

  @th.no_grad()
  def compute_returns_and_advantage(self, last_values: th.Tensor,
                                    dones: th.Tensor) -> None:
    last_values = last_values.reshape(self.n_envs).to(self.device)
    dones = dones.reshape(self.n_envs).float().to(self.device)
    last_gae = th.zeros(self.n_envs, device=self.device)
    for step in reversed(range(self.buffer_size)):
      if step == self.buffer_size - 1:
        next_non_terminal = 1.0 - dones
        v_next = last_values
      else:
        next_non_terminal = 1.0 - self.episode_starts[step + 1]
        v_next = self._values[step + 1]
      target = self._target(step, v_next, next_non_terminal)
      delta = target - self._values[step]
      last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
      self.advantages[step] = last_gae
    self._returns = self.advantages + self._values

  # --- sampling ---------------------------------------------------------------
  def get(self, batch_size: Optional[int] = None
          ) -> Generator[RolloutBufferSamples, None, None]:
    assert self.full, "buffer not full"
    total = self.buffer_size * self.n_envs
    obs = self.observations.reshape(total, self.obs_dim)
    act = self.actions.reshape(total, self.act_dim)
    val = self._values.reshape(total)
    logp = self.log_probs.reshape(total)
    adv = self.advantages.reshape(total)
    ret = self._returns.reshape(total)
    idx = th.randperm(total, device=self.device)
    if batch_size is None:
      batch_size = total
    for start in range(0, total, batch_size):
      b = idx[start:start + batch_size]
      yield RolloutBufferSamples(
        observations=obs[b], actions=act[b], old_values=val[b],
        old_log_prob=logp[b], advantages=adv[b], returns=ret[b])


class TensorReachAvoidRolloutBuffer(TensorSafetyRolloutBuffer):
  """Torch rollout buffer with the reach-avoid backup (adds ``l_x``).

  Torch twin of :class:`safety_sb3.safety_buffers.ReachAvoidRolloutBuffer`;
  see it for the operator, the anchor rationale, and ``terminal_type``.
  """

  def __init__(self, *args, terminal_type: str = "all", **kwargs):
    self.terminal_type = backups.check_terminal_type(terminal_type)
    super().__init__(*args, **kwargs)

  def reset(self) -> None:
    super().reset()
    self.l_x = th.zeros(self.buffer_size, self.n_envs, device=self.device)

  def _target(self, step: int, v_next: th.Tensor,
              not_done: th.Tensor) -> th.Tensor:
    return backups.reach_avoid_target(self.rewards[step], self.l_x[step],
                                      v_next, not_done, self.gamma,
                                      self.terminal_type)
