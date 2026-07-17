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

from . import backups
from .safety_buffers import ReachAvoidRolloutBuffer
from .safety_ppo import SafetyPPO
from .tensor_buffers import TensorReachAvoidRolloutBuffer


class ReachAvoidPPO(SafetyPPO):
  """PPO with the reach-avoid Bellman backup (g = reward, l = info["l_x"]).

  :param terminal_type: how terminal steps are valued -- ``"all"`` (default) ->
      ``min(l, g)``, the reach-avoid horizon terminal condition; ``"g"`` -> ``g``.
      See :func:`safety_sb3.backups.reach_avoid_target`. A first-class kwarg here
      (forwarded to the rollout buffer), mirroring :class:`ReachAvoidSAC`; it is
      the *algorithm-side* half of the pairing with the environment's
      ``end_criterion``. It is ignored in avoid mode (``IsaacsPPO``), whose
      buffer has no ``l``.
  """

  numpy_rollout_buffer_class = ReachAvoidRolloutBuffer
  tensor_rollout_buffer_class = TensorReachAvoidRolloutBuffer
  _MODE = backups.REACH_AVOID

  def __init__(self, *args, terminal_type: str = "all",
               rollout_buffer_kwargs=None, **kwargs):
    backups.check_terminal_type(terminal_type)
    rollout_buffer_kwargs = dict(rollout_buffer_kwargs or {})
    # Only the reach-avoid buffers accept terminal_type; the avoid buffers
    # (IsaacsPPO) do not, so do not inject it there.
    if self._MODE == backups.REACH_AVOID:
      rollout_buffer_kwargs.setdefault("terminal_type", terminal_type)
    self.terminal_type = terminal_type
    super().__init__(*args, rollout_buffer_kwargs=rollout_buffer_kwargs, **kwargs)

  @property
  def _is_reach_avoid(self) -> bool:
    """Does this class play the reach-avoid game (vs. the plain avoid game)?

    Subclasses that flip ``_MODE`` to :data:`safety_sb3.backups.AVOID` (e.g.
    :class:`~safety_sb3.isaacs_ppo.IsaacsPPO`) reuse every method here but have
    no target set, so all ``l``-dependent machinery must switch off. In the PPO
    family the backup itself is chosen by the rollout buffer, not by ``_MODE``;
    ``_MODE`` gates the ``l``-plumbing that feeds it.
    """
    return self._MODE == backups.REACH_AVOID

  def _setup_model(self) -> None:
    super()._setup_model()
    # Avoid-mode subclasses pair with the plain Safety buffers, which SafetyPPO
    # has already validated; only the reach-avoid game needs an l-carrying one.
    if self._is_reach_avoid:
      assert isinstance(
        self.rollout_buffer, (ReachAvoidRolloutBuffer, TensorReachAvoidRolloutBuffer)
      ), "ReachAvoidPPO requires a ReachAvoid rollout buffer (numpy or tensor)."

  def _record_step_extras(self, rollout_buffer: RolloutBuffer, infos: list) -> None:
    """Capture the target margin ``l(s)`` at the slot ``add()`` will fill.

    No-op in avoid mode: the avoid buffers have no ``l_x`` and the game has no
    target set.
    """
    if not self._is_reach_avoid:
      return
    assert isinstance(rollout_buffer, ReachAvoidRolloutBuffer)
    rollout_buffer.l_x[rollout_buffer.pos] = np.array(
      [float(info.get("l_x", 0.0)) for info in infos], dtype=np.float32
    )

  def _record_step_extras_tensor(self, rollout_buffer, l_x: th.Tensor) -> None:
    if not self._is_reach_avoid:
      return
    assert isinstance(rollout_buffer, TensorReachAvoidRolloutBuffer)
    rollout_buffer.l_x[rollout_buffer.pos] = l_x.reshape(rollout_buffer.n_envs)
