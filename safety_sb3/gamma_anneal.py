"""Discount-factor (gamma) annealing for the Safety Bellman backups.

WHY THIS EXISTS
---------------
The reach-avoid / avoid Safety Bellman operator

    V = (1 - gamma) * min(l, g)  +  gamma * min(g, max(l, V'))

is a contraction with a *sharp* fixed point only in the limit ``gamma -> 1``.
At a fixed ``gamma = 0.99`` the ``(1 - gamma)`` anchor keeps bleeding the
immediate margin into the value everywhere, so the learned zero-level set (the
safe/unsafe boundary we filter on) is smeared and the reach term never becomes
crisp.  Training the operator *directly* at ``gamma -> 1`` is ill-conditioned
(the contraction modulus -> 1, bootstrapping dominates, the critic diverges
early).  The reach-avoid literature (Fisac 2019; Hsu 2021; the ISAACS
``safe_adaptation_dev`` codebase) fixes this with DISCOUNT ANNEALING: start
contractive at ``gamma = 0.99`` for a well-conditioned early fixed point, then
anneal ``gamma -> 1`` so the boundary sharpens as the critic stabilises.

TWO SCHEDULERS (both provided; ``StepGammaAnneal`` is the DEFAULT)
-----------------------------------------------------------------
* :class:`StepGammaAnneal` — REFERENCE-FAITHFUL discrete jumps (``StepLRMargin``
  in ``safe_adaptation_dev/agent/scheduler.py``): the gap ``(goal - gamma)`` is
  multiplied by ``ratio`` at fixed training-fraction milestones.  Default
  0.99 -> 0.999 (at 20%) -> 0.9999 (at 40%), then hold.  This is ``is_stepwise``
  so each jump triggers an ENTROPY-TEMPERATURE (alpha) RESET in the SAC learners
  (the Q-scale shifts discontinuously at a jump, so the tuned alpha is stale —
  the reference resets both actors' alpha on every gamma jump; see
  ``GammaAnnealMixin._on_gamma_jump`` / ``SafetySAC``).

* :class:`GeometricGammaAnneal` — a smooth (continuous) log-space interpolation
  reaching ``end`` at ``anneal_frac`` then holding.  No discontinuity, so no
  alpha reset.  Available via ``gamma_anneal=GeometricGammaAnneal(...)``.

ON by default in every Safety* algorithm; ``gamma_anneal=False`` disables it, a
callable ``frac -> gamma`` (optionally exposing ``.is_stepwise``) is a custom
schedule.  All knobs (start/end/ratio/period/anneal fraction) are constructor
arguments.  Gamma is logged as ``train/gamma`` (+ ``train/gamma_jump`` on a step).
"""

from __future__ import annotations

from typing import Callable


class StepGammaAnneal:
    """Discrete-jump gamma schedule (reference ``StepLRMargin``).

    At training fraction ``f``, the number of elapsed jumps is
    ``n = floor(f / period_frac)`` and

        gamma(f) = goal - (goal - init) * ratio**n     (clamped to <= end)

    Defaults (init 0.99, ratio 0.1, period_frac 0.20, end 0.9999) reproduce the
    reference progression the user specified: 0.99 -> 0.999 @20% -> 0.9999 @40%,
    then hold.  ``is_stepwise = True`` so the SAC learners reset alpha on each jump.

    :param init: starting gamma (default 0.99).
    :param end: final gamma held after the last jump (default 0.9999).
    :param ratio: multiplicative decay of the gap ``(goal - gamma)`` per jump
        (default 0.1 -> one extra nine per jump).
    :param period_frac: training fraction between jumps (default 0.20; the user's
        "a jump every 10-20% of allotted time").
    :param goal: the limit the gap closes toward — 1.0 for HJ reachability.
    """

    is_stepwise = True

    def __init__(self, init: float = 0.99, end: float = 0.9999, ratio: float = 0.1,
                 period_frac: float = 0.20, goal: float = 1.0):
        init, end, ratio, period_frac, goal = (
            float(init), float(end), float(ratio), float(period_frac), float(goal))
        if not 0.0 < period_frac <= 1.0:
            raise ValueError(f"period_frac must be in (0, 1], got {period_frac}")
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"ratio must be in (0, 1), got {ratio}")
        if not init < goal or not end < goal:
            raise ValueError(f"init and end must be < goal={goal}; got {init}, {end}")
        self.init, self.end, self.ratio = init, end, ratio
        self.period_frac, self.goal = period_frac, goal
        self._gap0 = goal - init
        self._degenerate = init >= end

    def __call__(self, frac: float) -> float:
        f = min(max(float(frac), 0.0), 1.0)
        if self._degenerate:
            return self.init
        n = int(f / self.period_frac)              # elapsed jumps
        gamma = self.goal - self._gap0 * (self.ratio ** n)
        return min(gamma, self.end)                # clamp at the final value

    def __repr__(self) -> str:
        return (f"StepGammaAnneal(init={self.init}, end={self.end}, "
                f"ratio={self.ratio}, period_frac={self.period_frac})")


class GeometricGammaAnneal:
    """Smooth (continuous) gamma schedule — log-space interpolation of the gap.

    ``gamma(0) = init``; ``gamma(anneal_frac) = end``; constant ``end`` after.
    Equal fractions of progress multiply ``(goal - gamma)`` by a constant factor.
    ``is_stepwise = False`` -> no alpha reset (the change is continuous).

    :param init: gamma at fraction 0 (default 0.99).
    :param end: gamma reached at ``anneal_frac`` and held after (default 0.9999).
    :param goal: the limit the gap closes toward, ``1.0`` for HJ reachability.
    :param anneal_frac: training fraction at which ``end`` is reached (default 0.5).
    """

    is_stepwise = False

    def __init__(self, init: float = 0.99, end: float = 0.9999,
                 goal: float = 1.0, anneal_frac: float = 0.5):
        init, end, goal, anneal_frac = float(init), float(end), float(goal), float(anneal_frac)
        if not 0.0 < anneal_frac <= 1.0:
            raise ValueError(f"anneal_frac must be in (0, 1], got {anneal_frac}")
        if not init < goal or not end < goal:
            raise ValueError(f"init and end must be < goal={goal}, got init={init}, end={end}")
        self.init, self.end, self.goal, self.anneal_frac = init, end, goal, anneal_frac
        self._gap0 = goal - init   # initial gap, e.g. 0.01
        self._gap1 = goal - end    # final gap,   e.g. 0.0001
        self._degenerate = init >= end

    def __call__(self, frac: float) -> float:
        f = min(max(float(frac), 0.0), 1.0)
        if self._degenerate:
            return self.init
        if f >= self.anneal_frac:
            return self.end
        t = f / self.anneal_frac                       # 0 .. 1 across the ramp
        gap = self._gap0 * (self._gap1 / self._gap0) ** t   # geometric in log space
        return self.goal - gap

    def __repr__(self) -> str:
        return (f"GeometricGammaAnneal(init={self.init}, end={self.end}, "
                f"goal={self.goal}, anneal_frac={self.anneal_frac})")


# Type of a gamma schedule: training fraction in [0, 1] -> gamma. May expose an
# ``is_stepwise`` bool (whether a change is a discrete jump that resets alpha).
GammaSchedule = Callable[[float], float]


def make_default_gamma_schedule(init: float = 0.99) -> StepGammaAnneal:
    """The default on-by-default schedule: REFERENCE-FAITHFUL discrete jumps
    (``init`` -> 0.999 @20% -> 0.9999 @40%, hold), alpha reset on each jump."""
    return StepGammaAnneal(init=init, end=0.9999, ratio=0.1, period_frac=0.20)


class GammaAnnealMixin:
    """Mix-in that anneals ``self.gamma`` over training for every Safety* algo.

    Wiring:

    * ``_update_current_progress_remaining`` re-applies the schedule each
      iteration -- covers every NUMPY / on-policy path (SB3 calls it in both
      learn loops).
    * The GPU-resident collect loops bypass that call, so ``SafetyPPO`` /
      ``SafetySAC`` / the two-player variants also call ``_apply_gamma_anneal()``
      at the top of their ``collect_rollouts``.  ``_apply`` is idempotent.

    Gamma is applied at *consumption* time: ``rollout_buffer.gamma`` for PPO GAE,
    ``self.gamma`` for the SAC/DQN TD target.  On a DISCRETE jump (``StepGammaAnneal``)
    ``_on_gamma_jump`` fires -- a no-op here (PPO), overridden by the SAC learners
    to reset the entropy temperature(s), mirroring the reference.
    """

    _gamma_schedule: GammaSchedule | None = None
    _gamma_last: float | None = None

    def _setup_gamma_anneal(self, gamma_anneal) -> None:
        """Resolve the ``gamma_anneal`` constructor arg to a schedule (or None).

        ``True`` (default) -> the default DISCRETE-JUMP schedule seeded from the
        current ``self.gamma``; ``False``/``None`` -> disabled (constant gamma); a
        callable ``frac -> gamma`` -> used verbatim.
        """
        if gamma_anneal is False or gamma_anneal is None:
            self._gamma_schedule = None
        elif gamma_anneal is True:
            self._gamma_schedule = make_default_gamma_schedule(init=float(self.gamma))
        elif callable(gamma_anneal):
            self._gamma_schedule = gamma_anneal
        else:
            raise TypeError(
                "gamma_anneal must be a bool or a callable(frac)->gamma, got "
                f"{type(gamma_anneal).__name__}")
        self._gamma_last = None

    def _apply_gamma_anneal(self) -> None:
        """Recompute gamma from the current training fraction, propagate it, and
        fire ``_on_gamma_jump`` when a stepwise schedule crosses a jump."""
        sched = getattr(self, "_gamma_schedule", None)
        if sched is None:
            return
        total = getattr(self, "_total_timesteps", 0) or 0
        frac = 0.0 if total <= 0 else min(1.0, max(0.0, self.num_timesteps / float(total)))
        g = float(sched(frac))
        old = self._gamma_last
        self.gamma = g
        # On-policy: GAE reads gamma off the buffer(s). Two-player PPO keeps a
        # second (dstb) buffer -- update whichever exist.
        for name in ("rollout_buffer", "dstb_rollout_buffer"):
            buf = getattr(self, name, None)
            if buf is not None:
                buf.gamma = g
        logger = getattr(self, "logger", None)
        if logger is not None:
            logger.record("train/gamma", g)
        # Discrete jump -> alpha reset hook (stepwise schedules only).
        if (getattr(sched, "is_stepwise", False) and old is not None
                and abs(g - old) > 1e-12):
            if logger is not None:
                logger.record("train/gamma_jump", g)
            self._on_gamma_jump(old, g)
        self._gamma_last = g

    def _on_gamma_jump(self, old_gamma: float, new_gamma: float) -> None:
        """Hook fired when a DISCRETE (stepwise) gamma jump occurs. No-op for
        PPO/DQN; SAC learners override to reset the entropy temperature(s)."""

    def _update_current_progress_remaining(self, num_timesteps: int,
                                           total_timesteps: int) -> None:
        super()._update_current_progress_remaining(num_timesteps, total_timesteps)
        self._apply_gamma_anneal()
