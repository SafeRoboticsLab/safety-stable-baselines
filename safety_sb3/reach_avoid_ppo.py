"""On-policy reach-avoid RL: PPO with the reach-avoid Bellman backup.

    V(s) = min( g(s), max( l(s), V(s') ) )

``g`` (safety margin) rides on the reward channel; ``l`` (target margin) is
read from ``info["l_x"]`` on the numpy path or returned directly by
``TensorVecEnv.step_tensor`` on the GPU-resident path. The backup lives in the
paired rollout buffers (numpy: :class:`ReachAvoidRolloutBuffer`; tensor:
:class:`TensorReachAvoidRolloutBuffer` — identical math).

Unlike the SAC family this is fully on-policy: it pairs with vectorized envs
at large ``n_envs`` (the GPU-parallel regime where off-policy ISAACS saturates;
validated on a 1280-env MuJoCo quadruped in ``unitree_rl_mjlab``).

Inherits the rsl_rl-parity training recipe from :class:`SafetyPPO`
(timeout-bootstrap gating, KL-adaptive LR, obs normalization + reward-norm
guard); it only needs to thread the target margin ``l(s)`` into the buffer,
via ``_record_step_extras`` / ``_record_step_extras_tensor``.

References:
  K.-C. Hsu, V. Rubies-Royo, C. Tomlin, J. F. Fisac, "Safety and Liveness
  Guarantees through Reach-Avoid Reinforcement Learning", RSS 2021.
"""

from __future__ import annotations

import numpy as np
import torch as th
from stable_baselines3.common.buffers import RolloutBuffer

from .safety_buffers import ReachAvoidRolloutBuffer
from .safety_ppo import SafetyPPO
from .tensor_buffers import TensorReachAvoidRolloutBuffer


class ReachAvoidPPO(SafetyPPO):
  """PPO with the reach-avoid Bellman backup (g = reward, l = info["l_x"])."""

  numpy_rollout_buffer_class = ReachAvoidRolloutBuffer
  tensor_rollout_buffer_class = TensorReachAvoidRolloutBuffer

  def _setup_model(self) -> None:
    super()._setup_model()
    assert isinstance(
      self.rollout_buffer, (ReachAvoidRolloutBuffer, TensorReachAvoidRolloutBuffer)
    ), "ReachAvoidPPO requires a ReachAvoid rollout buffer (numpy or tensor)."

  def _record_step_extras(self, rollout_buffer: RolloutBuffer, infos: list) -> None:
    """Capture the target margin ``l(s)`` at the slot ``add()`` will fill."""
    assert isinstance(rollout_buffer, ReachAvoidRolloutBuffer)
    rollout_buffer.l_x[rollout_buffer.pos] = np.array(
      [float(info.get("l_x", 0.0)) for info in infos], dtype=np.float32
    )

  def _record_step_extras_tensor(self, rollout_buffer, l_x: th.Tensor) -> None:
    assert isinstance(rollout_buffer, TensorReachAvoidRolloutBuffer)
    rollout_buffer.l_x[rollout_buffer.pos] = l_x.reshape(rollout_buffer.n_envs)
