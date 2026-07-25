"""Off-policy replay buffer for the reach-avoid Mode — player-agnostic.

Like the rollout buffers, replay buffers carry the **M** of MAP and nothing
else: both the single-player and two-player reach-avoid SAC learners store into
the same class (the two-player game concatenates ``[a_ctrl, a_dstb]`` into the
ordinary action field, so no extra column is needed).

``ReachAvoidReplayBuffer`` extends SB3's :class:`ReplayBuffer` with the **target
margin** ``l(s)`` that the reach-avoid backup needs::

    V(s) = (1-gamma)*min(l, g) + gamma*min(g, max(l, V(s')))

The safety margin ``g(s)`` keeps riding on the standard ``reward`` field; the
env supplies ``l(s)`` per step via ``info["l_x"]``, and the buffer captures it
itself in :meth:`ReachAvoidReplayBuffer.add`. The avoid and cumulative modes need
no extra column and use SB3's stock ``ReplayBuffer``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer


class ReachAvoidReplayBufferSamples(NamedTuple):
  observations: th.Tensor
  actions: th.Tensor
  next_observations: th.Tensor
  dones: th.Tensor
  rewards: th.Tensor  # safety margin g(s)
  l_x: th.Tensor  # target margin l(s)


class ReachAvoidReplayBuffer(ReplayBuffer):
  """SB3 replay buffer that also stores the per-step target margin ``l(s)``.

  ``l(s)`` is read from ``info["l_x"]`` on each ``add``. Assumes the default
  SAC buffer layout (``optimize_memory_usage=False``), which stores
  ``next_observations`` explicitly.
  """

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    if self.optimize_memory_usage:
      raise ValueError(
        "ReachAvoidReplayBuffer requires optimize_memory_usage=False "
        "(it stores next_observations explicitly)."
      )
    self.l_x = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

  def add(self, obs, next_obs, action, reward, done, infos) -> None:
    # Must capture l_x at the CURRENT pos before super().add() advances it.
    for env_i, info in enumerate(infos):
      self.l_x[self.pos, env_i] = float(info.get("l_x", 0.0))
    super().add(obs, next_obs, action, reward, done, infos)

  def sample(self, batch_size: int, env=None) -> ReachAvoidReplayBufferSamples:
    upper = self.buffer_size if self.full else self.pos
    batch_inds = np.random.randint(0, upper, size=batch_size)
    env_indices = np.random.randint(0, self.n_envs, size=batch_size)

    next_obs = self._normalize_obs(
      self.next_observations[batch_inds, env_indices, :], env
    )
    return ReachAvoidReplayBufferSamples(
      observations=self.to_torch(
        self._normalize_obs(self.observations[batch_inds, env_indices, :], env)
      ),
      actions=self.to_torch(self.actions[batch_inds, env_indices, :]),
      next_observations=self.to_torch(next_obs),
      dones=self.to_torch(
        (
          self.dones[batch_inds, env_indices]
          * (1 - self.timeouts[batch_inds, env_indices])
        ).reshape(-1, 1)
      ),
      rewards=self.to_torch(
        self._normalize_reward(
          self.rewards[batch_inds, env_indices].reshape(-1, 1), env
        )
      ),
      l_x=self.to_torch(self.l_x[batch_inds, env_indices].reshape(-1, 1)),
    )
