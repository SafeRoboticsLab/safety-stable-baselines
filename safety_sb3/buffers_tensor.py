"""GPU-resident rollout buffers — torch twins of :mod:`safety_sb3.buffers_rollout`.

Identical math, no numpy on the hot path. One class per Mode, player-agnostic
(see the numpy module for why buffers carry no ``1P``/``2P``). Both families
call the same operators from :mod:`safety_sb3.backups`; ``tests/test_backups.py``
asserts exact parity.

``get()`` yields standard ``RolloutBufferSamples`` whose fields are device
tensors, so stock ``PPO.train()`` consumes them unchanged. ``values`` /
``returns`` are exposed as numpy properties (PPO's explained-variance logging
touches them once per update).

:meth:`TensorSafetyRolloutBuffer.record_extras` is the tensor twin of the numpy
buffers' ``infos`` capture: on this path the env returns ``l_x`` directly from
``step_tensor``, so the buffer receives the tensor rather than a list of dicts.
"""

from __future__ import annotations

from typing import Generator, NamedTuple, Optional

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBufferSamples

from . import backups


class MaskedRolloutBufferSamples(NamedTuple):
  """``RolloutBufferSamples`` + a per-sample ``policy_mask`` (RAS handover training).

  The six standard PPO fields (order preserved) plus ``policy_mask`` -- a 0/1
  float, 1 where the REACH policy acted (trainable), 0 where a frozen hand-off
  skill acted (excluded from the policy gradient, kept for value/GAE)."""
  observations: th.Tensor
  actions: th.Tensor
  old_values: th.Tensor
  old_log_prob: th.Tensor
  advantages: th.Tensor
  returns: th.Tensor
  policy_mask: th.Tensor


class TensorSafetyRolloutBuffer:
  """Torch rollout buffer with the AVOID (safety) Bellman backup."""

  is_tensor_buffer = True

  def __init__(self, buffer_size: int, observation_space: spaces.Space,
               action_space: spaces.Space, device: str = "cuda:0",
               gae_lambda: float = 0.95, gamma: float = 0.99,
               n_envs: int = 1, mode: str | None = None, **_ignored):
    self._MODE = backups.check_mode(self._MODE if mode is None else mode)
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

  def record_extras(self, l_x: th.Tensor) -> None:
    """Capture per-step extras into the slot ``add()`` will fill.

    Tensor twin of the numpy buffers' ``record_extras(infos)``: the GPU-resident
    env hands back ``l_x`` from ``step_tensor`` directly, so there are no infos
    to parse. No-op here — the avoid operator needs only ``g``.
    """

  # --- backup ---------------------------------------------------------------
  #: backup this buffer computes; the ``mode=`` ctor kwarg overrides it
  _MODE = backups.AVOID

  def _target(self, step: int, v_next: th.Tensor,
              not_done: th.Tensor) -> th.Tensor:
    """Dispatch on ``self._MODE`` — twin of the numpy buffer's ``_target``."""
    l_x = getattr(self, "l_x", None)
    return backups.target(
      self._MODE, self.rewards[step], v_next, not_done, self.gamma,
      l=None if l_x is None else l_x[step],
      terminal_type=getattr(self, "terminal_type", "all"))

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

  Torch twin of :class:`safety_sb3.buffers_rollout.ReachAvoidRolloutBuffer`; see
  it for the operator, the anchor rationale, and ``terminal_type``.
  """

  _MODE = backups.REACH_AVOID

  def __init__(self, *args, terminal_type: str = "all", **kwargs):
    self.terminal_type = backups.check_terminal_type(terminal_type)
    super().__init__(*args, **kwargs)

  def reset(self) -> None:
    super().reset()
    self.l_x = th.zeros(self.buffer_size, self.n_envs, device=self.device)

  def record_extras(self, l_x: th.Tensor) -> None:
    """Keep this step's target margin ``l(s)`` (straight off ``step_tensor``)."""
    self.l_x[self.pos] = l_x.reshape(self.n_envs)


class TensorReachAvoidMaskedRolloutBuffer(TensorReachAvoidRolloutBuffer):
  """Reach-avoid buffer + a per-step ``policy_mask`` (RAS phase-2 handover training).

  Identical to :class:`TensorReachAvoidRolloutBuffer` for value/returns/GAE --
  ``compute_returns_and_advantage`` is INHERITED UNCHANGED, so lander-driven
  steps DO contribute to the value backup (that is the point: the reach-avoid
  value learns to reach a state the frozen lander actually lands from). The mask
  it additionally stores is consumed ONLY by :class:`ReachAvoidMaskedPPO1P`'s
  ``train()`` to zero those steps out of the POLICY gradient. With an all-ones
  mask this buffer is behaviourally identical to its parent.
  """

  def reset(self) -> None:
    super().reset()
    # 1.0 = reach-controlled (trainable), 0.0 = hand-off skill drove this step.
    self.policy_mask = th.ones(self.buffer_size, self.n_envs, device=self.device)

  def record_policy_mask(self, mask: th.Tensor) -> None:
    """Keep this step's reach-active mask (straight off ``env._policy_mask``)."""
    self.policy_mask[self.pos] = mask.reshape(self.n_envs).float()

  def get(self, batch_size: Optional[int] = None
          ) -> Generator[MaskedRolloutBufferSamples, None, None]:
    assert self.full, "buffer not full"
    total = self.buffer_size * self.n_envs
    obs = self.observations.reshape(total, self.obs_dim)
    act = self.actions.reshape(total, self.act_dim)
    val = self._values.reshape(total)
    logp = self.log_probs.reshape(total)
    adv = self.advantages.reshape(total)
    ret = self._returns.reshape(total)
    pm = self.policy_mask.reshape(total)          # like advantages
    idx = th.randperm(total, device=self.device)
    if batch_size is None:
      batch_size = total
    for start in range(0, total, batch_size):
      b = idx[start:start + batch_size]
      yield MaskedRolloutBufferSamples(
        observations=obs[b], actions=act[b], old_values=val[b],
        old_log_prob=logp[b], advantages=adv[b], returns=ret[b],
        policy_mask=pm[b])


class TensorCumulativeRolloutBuffer(TensorSafetyRolloutBuffer):
  """Torch twin of :class:`safety_sb3.buffers_rollout.CumulativeRolloutBuffer` —
  the ordinary discounted-return backup, i.e. stock GAE on device."""

  _MODE = backups.CUMULATIVE
