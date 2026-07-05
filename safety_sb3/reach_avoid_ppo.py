"""Reach-avoid PPO (increment 1 of the on-policy ISAACS family).

On-policy counterpart of :class:`ReachAvoidSAC`: the avoid-only
:class:`SafetyPPO` backup ``V = min(g, V')`` is replaced with the reach-avoid
Bellman backup

    V(s) = min( g(s), max( l(s), gamma * V(s') ) )

where ``g(s) >= 0`` defines the safe set (rides on the ``reward`` channel, as
in all Safety* algorithms here) and ``l(s) >= 0`` the target set (supplied by
the env via ``info["l_x"]`` — the same contract as :class:`ReachAvoidSAC`).
``V(s) > 0`` iff a policy exists that reaches ``{l >= 0}`` without ever
leaving ``{g >= 0}``.

Unlike the SAC family this is fully on-policy: it pairs with vectorized envs
at large ``n_envs`` (the GPU-parallel regime where off-policy ISAACS saturates;
validated on a 1280-env MuJoCo quadruped in ``unitree_rl_mjlab``).

Inherits the rsl_rl-parity training recipe from :class:`SafetyPPO`
(timeout-bootstrap gating, KL-adaptive LR); it only needs to thread the target
margin ``l(s)`` into the buffer, which it does via ``_record_step_extras``.

References:
  K.-C. Hsu, V. Rubies-Royo, C. Tomlin, J. F. Fisac, "Safety and Liveness
  Guarantees through Reach-Avoid Reinforcement Learning", RSS 2021.
"""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.buffers import RolloutBuffer

from .safety_buffers import ReachAvoidRolloutBuffer
from .safety_ppo import SafetyPPO


class ReachAvoidPPO(SafetyPPO):
  """PPO with the reach-avoid Bellman backup (g = reward, l = info["l_x"])."""

  def __init__(self, *args, rollout_buffer_class=None, **kwargs):
    if rollout_buffer_class is None:
      rollout_buffer_class = ReachAvoidRolloutBuffer
    super().__init__(*args, rollout_buffer_class=rollout_buffer_class, **kwargs)

  def _setup_model(self) -> None:
    super()._setup_model()
    assert isinstance(self.rollout_buffer, ReachAvoidRolloutBuffer), (
      "ReachAvoidPPO requires ReachAvoidRolloutBuffer."
    )

  def _record_step_extras(self, rollout_buffer: RolloutBuffer, infos: list) -> None:
    """Capture the target margin ``l(s)`` at the slot ``add()`` will fill."""
    assert isinstance(rollout_buffer, ReachAvoidRolloutBuffer)
    rollout_buffer.l_x[rollout_buffer.pos] = np.array(
      [float(info.get("l_x", 0.0)) for info in infos], dtype=np.float32
    )
