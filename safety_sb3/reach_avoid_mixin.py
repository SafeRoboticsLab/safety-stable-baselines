"""The reach-avoid plumbing for the on-policy family — a MIXIN, not a base.

Why a mixin. MAP is a product: Mode x Algorithm x Players. Single inheritance
can linearise only one axis, so one of them has to be composition, and the
players axis is the one that carries the code — the two on-policy update loops
are hundreds of lines apart, while the reach-avoid delta below is a couple of
dozen. Making reach-avoid a *parent* of the loops would therefore mean
duplicating a loop per mode; making it a mixin means it composes with both::

    class ReachAvoidPPO1P(_ReachAvoidPlumbing, AbstractPPO1P): _MODE = REACH_AVOID
    class ReachAvoidPPO2P(_ReachAvoidPlumbing, AbstractPPO2P): _MODE = REACH_AVOID
    class SafetyPPO1P(AbstractPPO1P):                          _MODE = AVOID

and — the part that matters for correctness — the avoid learners never receive
it at all. Before v0.4.0 the avoid classes *inherited* the reach-avoid class and
switched its machinery back off through an ``_is_reach_avoid`` predicate; every
piece of ``l``-handling then had to be guarded, and one missed guard silently
fed an avoid learner a target margin. There is nothing to switch off now.

What is left here is deliberately small. The capture of ``l(s)`` itself used to
live in the *algorithm* (a ``_record_step_extras`` hook on the rollout loop) and
was the bulk of this file; it now lives in the rollout buffers, which is where
the rest of the ``l`` plumbing already was — see
:mod:`safety_sb3.buffers_rollout`. What remains is one constructor argument and
the reach half of the two-player league's win condition.

SAC needs no equivalent: its mode delta is a handful of lines that already sit
inside :class:`~safety_sb3.sac_base.AbstractSAC` and dispatch on ``_MODE``.
"""

from __future__ import annotations

import numpy as np
import torch as th

from . import backups


class _ReachAvoidPlumbing:
    """Mixin: everything reach-avoid needs on top of an on-policy update loop.

    Compose it BEFORE the loop class so its ``__init__`` runs first::

        class ReachAvoidPPO2P(_ReachAvoidPlumbing, AbstractPPO2P): ...

    :param terminal_type: how terminal steps are valued — ``"all"`` (default) ->
        ``min(l, g)``, the reach-avoid horizon terminal condition;  ``"g"`` ->
        ``g``. Forwarded to the rollout buffer, which owns the operator. It is
        the *algorithm-side* half of the pairing with the environment's
        ``end_criterion``. See :func:`safety_sb3.backups.reach_avoid_target`.
    """

    def __init__(self, *args, terminal_type: str = "all",
                 rollout_buffer_kwargs=None, **kwargs):
        self.terminal_type = backups.check_terminal_type(terminal_type)
        rollout_buffer_kwargs = dict(rollout_buffer_kwargs or {})
        rollout_buffer_kwargs.setdefault("terminal_type", terminal_type)
        super().__init__(*args, rollout_buffer_kwargs=rollout_buffer_kwargs,
                         **kwargs)

    # --- the reach half of the two-player league's win condition -------------
    # Only reached when this mixin is composed with the TWO-player loop, which
    # scores an episode a ctrl win iff the ctrl player survived to the time
    # limit AND -- here -- reached the target at some point. The avoid game has
    # no target set, so AbstractPPO2P's defaults make these no-ops and the win
    # condition collapses to survival. (Composed with the 1P loop they are
    # simply never called: there is no league.)

    def _alloc_reach_flags(self, like) -> None:
        """``like`` is the freshly zeroed per-env avoid flag array, so cloning it
        matches its backend (numpy vs torch), device and width for free."""
        self._ever_l = like.clone() if th.is_tensor(like) else like.copy()

    def _note_reach(self, step_l) -> None:
        """Record this step's target-margin evidence.

        ``step_l`` is whatever the active collect loop has: the raw ``infos`` on
        the numpy path, the ``l_x`` tensor on the GPU-resident path.
        """
        l = (np.array([float(i.get("l_x", 0.0)) for i in step_l], np.float32)
             if isinstance(step_l, (list, tuple)) else step_l)
        self._ever_l |= l >= 0.0

    def _reach_flag(self):
        return self._ever_l

    def _clear_reach(self, dones) -> None:
        self._ever_l[dones] = False
