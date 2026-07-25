"""A2C, written without a fixed Bellman backup.

    AbstractA2C
    ├─ SafetyA2C1P        _MODE = AVOID
    ├─ ReachAvoidA2C1P    _MODE = REACH_AVOID   (+ _ReachAvoidPlumbing)
    └─ CumulativeA2C1P    _MODE = CUMULATIVE

A2C is single-player only — there is no ``AbstractA2C2P``. Nothing in the
two-player construction is A2C-specific, but nobody has needed it, and the
library does not ship learners it has not exercised.

Like PPO, A2C carries its backup in the **rollout buffer**, so a learner picks
its problem by declaring ``_MODE`` and the matching buffer pair follows. Unlike
PPO, A2C does not own its rollout loop: it uses SB3's stock
``OnPolicyAlgorithm.collect_rollouts`` unchanged, so there is no place to gate
the timeout value-bootstrap and no place to hand the buffer its per-step extras.

The extras still get through, because SB3's loop calls ``_update_info_buffer``
once per step between ``env.step()`` and ``rollout_buffer.add()`` — exactly the
point the reach-avoid buffer needs — and :meth:`AbstractA2C._update_info_buffer`
forwards from there. That is the whole cost of reach-avoid A2C.

.. warning::
   A2C here does NOT gate PPO's timeout value-bootstrap, because it does not
   override the collect loop. When episodes hit the time limit, SB3 adds
   ``gamma * V(terminal)`` to the reward — and in the two safety modes the
   reward is the physical margin ``g(s)``, so that corrupts the backup. Prefer
   the PPO-family learners for safety work; A2C is kept for parity and for
   cheap baselines.
"""
from __future__ import annotations

from stable_baselines3.a2c.a2c import A2C

from . import backups
from .buffers_rollout import (check_rollout_buffer_mode,
                              rollout_buffer_classes)
from .gamma_anneal import GammaAnnealMixin
from .reach_avoid_mixin import _ReachAvoidPlumbing


class AbstractA2C(GammaAnnealMixin, A2C):
    """A2C whose Bellman backup is chosen by ``_MODE``.

    ``gamma_anneal`` (ON by default) anneals the discount 0.99 -> 0.9999 over the
    first 50% of training (read off ``rollout_buffer.gamma`` in GAE); applied via
    ``_update_current_progress_remaining`` each iteration. See ``gamma_anneal.py``.
    """

    #: the Bellman operator this learner converges to; concretes set it
    _MODE = backups.AVOID

    def __init__(self, *args, rollout_buffer_class=None,
                 rollout_buffer_kwargs=None, gamma_anneal=True, **kwargs):
        backups.check_mode(self._MODE)
        if rollout_buffer_class is None:
            rollout_buffer_class, _ = rollout_buffer_classes(self._MODE)
        super().__init__(
            *args, rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs, **kwargs
        )
        self._setup_gamma_anneal(gamma_anneal)

    def _setup_model(self) -> None:
        # Builds policy, optimizer, and rollout buffer -> safe to check.
        super()._setup_model()
        check_rollout_buffer_mode(self, self.rollout_buffer)

    def _update_info_buffer(self, infos, dones=None) -> None:
        """SB3's per-step infos hook, borrowed to feed the rollout buffer.

        This is the one point in the stock on-policy loop that sees ``infos``
        and runs after ``env.step()`` but before ``rollout_buffer.add()`` —
        which is precisely the contract ``record_extras`` needs. The PPO family
        owns its loop and calls ``record_extras`` outright; A2C does not, so it
        piggybacks here. No-op for every buffer whose operator needs no extras.
        """
        super()._update_info_buffer(infos, dones)
        buffer = getattr(self, "rollout_buffer", None)
        if buffer is not None:
            buffer.record_extras(infos)


# ----------------------------------------------------------------- the modes

class SafetyA2C1P(AbstractA2C):
    """A2C with the **avoid** backup — ``V(s) = (1-γ)·g + γ·min(g, V')``."""

    _MODE = backups.AVOID


class ReachAvoidA2C1P(_ReachAvoidPlumbing, AbstractA2C):
    """A2C with the **reach-avoid** backup —
    ``V(s) = (1-γ)·min(l, g) + γ·min(g, max(l, V'))``.

    ``l`` arrives via ``info["l_x"]`` and is captured by the buffer (see the
    module docstring for how it reaches there without an A2C rollout loop).
    """

    _MODE = backups.REACH_AVOID


class CumulativeA2C1P(AbstractA2C):
    """Ordinary reward-maximizing A2C — ``V(s) = r + γ·V(s')``, i.e. stock A2C
    reached through this library's mode dispatch. Not a safety learner."""

    _MODE = backups.CUMULATIVE
