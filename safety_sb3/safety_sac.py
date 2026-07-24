"""Safety (avoid) SAC — single-player.

A one-line specialization of :class:`~safety_sb3.sac_base.AbstractSAC`: same
update loop, ``_MODE = backups.AVOID``.
"""
from __future__ import annotations

from . import backups
from .sac_base import AbstractSAC


class SafetySAC(AbstractSAC):
  """SAC with the AVOID (safety) Bellman backup — Fisac et al. 2019::

      V(s) = (1-γ)·g + γ·min(g, V')        terminal: V(s) = g

  ``g(s)`` is the safety margin (it rides on the ``reward`` field); ``V'`` is
  SAC's soft next-state value. ``V(s) >= 0`` iff the state is in the safe set.
  For a task with a *target* to reach use
  :class:`~safety_sb3.reach_avoid_sac.ReachAvoidSAC`; the two operators are not
  interchangeable (see :mod:`safety_sb3.backups`).

  Everything else — the tensor path, gamma annealing, the alpha floor/ceiling,
  the update loop — lives in :class:`~safety_sb3.sac_base.AbstractSAC`.
  """

  _MODE = backups.AVOID
  _tensor_store_l = False  # no l(s) in an avoid problem
