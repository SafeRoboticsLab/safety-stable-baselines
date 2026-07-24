"""The Bellman backups — defined ONCE, used by every learner.

This library solves two different safety problems, and they take different value
operators. Before this module existed the operators were re-implemented at four
call sites (the numpy rollout buffer, the torch rollout buffer, and two SAC
``train()`` bodies); they silently diverged, and the PPO family spent the whole
2026 campaign optimizing a fixed point that was neither problem's value. Every
backup now routes through here so that cannot recur.

A learner is then written *without reference to a particular operator* — it
takes a ``mode`` and calls :func:`target`. Ordinary sum-of-rewards RL is one of
the modes (:data:`CUMULATIVE`), so it is a special case of the same code, not a
separate library.

Convention (throughout the library)::

    g(s) >= 0   <=>  s is OUTSIDE the failure set   ("safe")
    l(s) >= 0   <=>  s is INSIDE the target set     ("reached")
    V is MAXIMIZED;  V(s) >= 0  <=>  s is in the solution set

The operators
-------------

**Avoid** (stay safe forever) — Fisac et al. 2019; ISAACS eq. 6/7::

    V(s) = (1 - gamma) * g(s)  +  gamma * min( g(s), V(s') )

**Reach-avoid** (reach the target while staying safe throughout) — Hsu et al.
RSS'21 eq. 15; Gameplay Filters eq. 6a::

    V(s) = (1 - gamma) * min(l(s), g(s))
         + gamma * min( g(s), max( l(s), V(s') ) )

**Cumulative** (the STANDARD RL operator — maximize the discounted sum of
rewards; Bellman 1957, and every textbook since)::

    V(s) = r(s)  +  gamma * V(s')

This one is *not* a safety operator and does not follow the sign convention
above: its first argument is a **reward**, not a margin, ``V`` is an expected
discounted return with no zero-level-set meaning, and there is no ``g``, no
``l`` and no failure set. It is here because the learners in this library are
written against :func:`target` rather than against a fixed backup, so plain
sum-of-rewards RL costs one branch and comes out as the degenerate member of the
family. Use it to run a *nominal* (reward-maximizing) baseline through exactly
the same actor/critic/entropy code as the safety learners — the standard control
for "is the safety operator doing the work, or is it just SAC?". Do not read a
cumulative ``V`` as a safety certificate; ``V >= 0`` means nothing here.

The two safety operators share the shape ``(1 - gamma) * anchor + gamma * backup``. **The anchor is
the "episode terminates now" payoff** — ``1 - gamma`` is the per-step
termination probability of the discounted formulation, so the anchor is what the
trajectory scores if it stops here:

* avoid: stopping now is a win iff you are safe            -> ``g``
* reach-avoid: iff you are in the target AND safe          -> ``min(l, g)``

The reach-avoid anchor is the same expression as the finite-horizon terminal
condition ``V_H = min(l, g)`` (Gameplay Filters eq. 5b); that identity is the
structural tell that it is not a stylistic choice.

Why the anchors are not interchangeable
---------------------------------------

Anchoring reach-avoid on ``g`` makes "stay safe forever, never reach" a fixed
point at ``V = g > 0`` — a win — when the reach-avoid value of such a trajectory
is ``max_t min(l_t, min_{s<=t} g_s) = max_t l_t < 0``, a loss. The resulting
fixed point is neither problem's value; RSS'21's under-approximation theorem
(``RA_gamma`` nested inside ``RA``) stops applying, so the critic can wrongly
certify reachability and is unsound to shield with. RSS'21 says of the
g-anchored form (its eq. 13) that it approximates "safety or liveness problems,
**but not both**".

Note that ISAACS anchors on ``g`` and is correct to: it is a pure *avoid* game
with no target set and no ``l`` anywhere in the paper. Gameplay Filters extends
ISAACS to reach-avoid and changes the anchor when it does. The mixture — a ``g``
anchor with a ``max(l, V')`` recursion — appears in none of the papers.

Avoid is not expressible as a reach-avoid instance
--------------------------------------------------

Tempting, and wrong: to make the reach-avoid operator compute the avoid value
you would need the anchor to reduce (``min(l, g) = g``, i.e. ``l >= g``) *and*
the recursion to reduce (``max(l, V') = V'``, i.e. ``l <= V'``). Since the avoid
recursion caps ``V' <= g``, that needs ``l >= g >= V' >= l`` — satisfiable only
in the degenerate case. A large negative ``l`` buys the recursion and destroys
the anchor (``V == l`` everywhere, empty safe set); a large positive ``l`` buys
the anchor and destroys the recursion (``V == g`` everywhere, no lookahead —
coming failures never propagate). Use the avoid operator for avoid problems.

All functions here are elementwise and shape-agnostic, and accept either numpy
arrays or torch tensors (``not_done`` may be float 0/1 or bool-as-float).
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch as th

Array = Union[np.ndarray, th.Tensor]

#: the two-player and single-player *avoid* problem
AVOID = "safety"
#: the two-player and single-player *reach-avoid* problem
REACH_AVOID = "reach-avoid"
#: ordinary discounted-return RL (NOT a safety problem -- see the docstring)
CUMULATIVE = "cumulative"
MODES = (AVOID, REACH_AVOID, CUMULATIVE)
#: the modes whose value is a safety certificate (``V >= 0`` <=> in the solution set)
SAFETY_MODES = (AVOID, REACH_AVOID)

#: terminal-target options for the reach-avoid operator (see ``reach_avoid_target``)
TERMINAL_TYPES = ("all", "g")


def _xp(x: Array):
  return th if isinstance(x, th.Tensor) else np


def check_mode(mode: str) -> str:
  if mode not in MODES:
    raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
  return mode


def check_terminal_type(terminal_type: str) -> str:
  if terminal_type not in TERMINAL_TYPES:
    raise ValueError(
      f"terminal_type must be one of {TERMINAL_TYPES}, got {terminal_type!r}")
  return terminal_type


def avoid_target(g: Array, v_next: Array, not_done: Array,
                 gamma: float) -> Array:
  """Avoid (safety) target: ``(1-g)*g + g*min(g, V')``; terminal -> ``g``.

  :param g: safety margin ``g(s)``, ``>= 0`` iff safe.
  :param v_next: bootstrap value ``V(s')``.
  :param not_done: 1.0 on non-terminal steps, 0.0 on terminal steps.
  :param gamma: discount.
  """
  xp = _xp(g)
  v_to_go = xp.minimum(g, v_next)
  return (not_done * ((1.0 - gamma) * g + gamma * v_to_go)
          + (1.0 - not_done) * g)


def reach_avoid_target(g: Array, l: Array, v_next: Array, not_done: Array,
                       gamma: float, terminal_type: str = "all") -> Array:
  """Reach-avoid target: ``(1-g)*min(l,g) + g*min(g, max(l, V'))``.

  :param g: safety margin ``g(s)``, ``>= 0`` iff safe.
  :param l: target margin ``l(s)``, ``>= 0`` iff in the target set.
  :param v_next: bootstrap value ``V(s')``.
  :param not_done: 1.0 on non-terminal steps, 0.0 on terminal steps.
  :param gamma: discount.
  :param terminal_type: target at terminal steps. ``"all"`` (default) ->
      ``min(l, g)``, the finite-horizon terminal condition (Gameplay Filters
      eq. 5b); ``"g"`` -> ``g`` alone. Both are offered by the reference
      (``safe_adaptation_dev/utils/train.py``). With timeout bootstrapping
      disabled, timeouts arrive here as terminal steps and ``"all"`` is
      correct for them: a timeout IS the horizon cutoff of eq. 5b.
  """
  check_terminal_type(terminal_type)
  xp = _xp(g)
  anchor = xp.minimum(l, g)
  v_to_go = xp.minimum(g, xp.maximum(l, v_next))
  terminal = g if terminal_type == "g" else anchor
  return (not_done * ((1.0 - gamma) * anchor + gamma * v_to_go)
          + (1.0 - not_done) * terminal)


def cumulative_target(reward: Array, v_next: Array, not_done: Array,
                      gamma: float) -> Array:
  """Standard RL target: ``r + gamma * V'``; terminal -> ``r``.

  The ordinary discounted-cumulative-reward backup, so that reward-maximizing
  RL is a mode of this library rather than a separate implementation. Unlike
  the two safety operators this one carries no margin semantics: the first
  argument is a **reward**, and the resulting ``V`` is an expected return whose
  sign means nothing.

  :param reward: immediate reward ``r(s, a)`` (NOT a margin).
  :param v_next: bootstrap value ``V(s')``.
  :param not_done: 1.0 on non-terminal steps, 0.0 on terminal steps.
  :param gamma: discount.
  """
  return reward + gamma * not_done * v_next


def target(mode: str, g: Array, v_next: Array, not_done: Array, gamma: float,
           l: Optional[Array] = None, terminal_type: str = "all") -> Array:
  """Dispatch to the operator for ``mode``. See the module docstring.

  :param mode: ``backups.AVOID``, ``backups.REACH_AVOID`` or
      ``backups.CUMULATIVE``.
  :param g: the safety margin ``g(s)`` for the safety modes; for
      ``CUMULATIVE`` this slot carries the **reward** instead.
  :param l: required for ``REACH_AVOID``; ignored for the other modes.
  """
  check_mode(mode)
  if mode == AVOID:
    return avoid_target(g, v_next, not_done, gamma)
  if mode == CUMULATIVE:
    return cumulative_target(g, v_next, not_done, gamma)
  if l is None:
    raise ValueError("mode='reach-avoid' requires the target margin l")
  return reach_avoid_target(g, l, v_next, not_done, gamma, terminal_type)
