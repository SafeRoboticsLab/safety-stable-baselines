"""Reach-avoid SAC (single-agent) — increment 1 toward ISAACS.

A one-line specialization of :class:`~safety_sb3.sac_base.AbstractSAC` (same
update loop as :class:`~safety_sb3.safety_sac.SafetySAC`) with the
**reach-avoid** Bellman backup::

    non-terminal:  y = (1-γ)·min(l, g) + γ·min(g, max(l, V'))
    terminal:      y = min(l, g)

where ``g(s)`` is the safety/avoid margin (rides on the ``reward`` field) and
``l(s)`` is the target/reach margin (supplied by the env via ``info["l_x"]`` and
stored by :class:`ReachAvoidReplayBuffer`).  ``V'`` is SAC's soft next-state
value (twin-target min minus entropy).  This matches the discounted reach-avoid
update in ``safe_adaptation_dev`` (``utils/train.get_bellman_update``,
``mode='reach-avoid'``, ``terminal_type='all'``).

This is the single-agent (max-player only) base; the ISAACS disturbance actor +
leaderboard are added in later increments on top of this.
"""

from __future__ import annotations

from safety_sb3 import backups
from safety_sb3.isaacs_buffers import ReachAvoidReplayBuffer  # noqa: F401 (re-export)
from safety_sb3.sac_base import AbstractSAC


class ReachAvoidSAC(AbstractSAC):
  """SAC with the reach-avoid Bellman backup.

  Setting ``_MODE`` is the whole class: the base then defaults
  ``replay_buffer_class`` to :class:`ReachAvoidReplayBuffer` (which stores
  ``l``), validates it, and dispatches the TD target through
  :func:`safety_sb3.backups.target`. ``terminal_type`` is a base kwarg.

  Tensor path (``TensorVecEnv``): the device-resident
  :class:`~safety_sb3.tensor_replay.TensorReplayBuffer` is built with
  ``store_l=True`` (l comes from ``step_tensor``'s ``l_x`` return, not infos).
  """

  _MODE = backups.REACH_AVOID
  _tensor_store_l = True  # tensor buffer stores l(s)
