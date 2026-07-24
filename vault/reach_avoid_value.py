"""reach_avoid_value -- the realized (trajectory-level) reach-avoid objective.

Given a recorded trajectory's per-step avoid margins g_t = min(odd_margin, contact_margin) and
target margins l_t = target_margin, this computes the same backward-induction recursion the
reach-avoid Bellman equation defines (see reach_avoid_sac.py's docstring), evaluated on an ACTUAL
realized trajectory rather than a learned critic:

    V_T       = min(g_T, l_T)                    (base case, at the last recorded step)
    V_t       = min(g_t, max(l_t, V_{t+1}))       (t = T-1, ..., 0)

Positive iff the target was reached without ever failing first (any g_t<0 caps every V at-or-
before that step; running out of horizon without reaching target typically leaves l_T<0). Shared
by safety_filter.RolloutMonitor (predictive one-step-then-fallback rollouts) and
reach_avoid_eval.py / the training-time checkpoint evaluator (full realized fallback rollouts) --
same formula, different trajectories.
"""
from __future__ import annotations


def realized_value(gs: list[float], ls: list[float]) -> float:
    """Backward induction of V_t = min(g_t, max(l_t, V_{t+1})), base case V_T = min(g_T, l_T)."""
    if not gs:
        raise ValueError("realized_value: empty trajectory")
    v = min(gs[-1], ls[-1])
    for i in range(len(gs) - 2, -1, -1):
        v = min(gs[i], max(ls[i], v))
    return v
