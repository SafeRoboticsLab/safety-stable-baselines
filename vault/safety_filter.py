"""safety_filter -- modular fallback / monitor / intervention safety-filter architecture.

Vocabulary (Hsu, Hu, Fisac 2024, "The Safety Filter: A Unified View of Safety-Critical Control in
Autonomous Systems"; arXiv:2309.05837):
  fallback policy pi*   -- a safety-oriented backup control law, guaranteed safe from anywhere in
                           its own induced safe set (a subset of the constraint set).
  safety monitor Delta* -- Delta*(x, u) >= 0 iff committing to u now still leaves the state able
                           to reach the target set under pi*, without ever leaving the constraint
                           set in between (i.e., stays inside pi*'s induced safe set).
  intervention scheme phi -- how the filter acts on the monitor's verdict:
     switch-type       : u_task if Delta*(x,u_task)>=eps, else pi*(x) outright.
     optimization-type  : argmin_u ||u-u_task|| subject to Delta*(x,u)>=threshold -- the discrete
                          analogue of a CBF-QP.

These three pieces are designed independently (the paper's Prop. 1 "separation principle") and
compose via plain Python objects: any Fallback works with any Monitor, and any Monitor works with
any Intervention -- an OptimizationIntervention's internal search just needs
`monitor.evaluate(...)` callable for as many candidate actions as it wants to try.

Two Monitor implementations:
  ValueMonitor   -- queries a LEARNED critic Q(x,u) directly, no simulation. Fast, but only as
                    good as the critic's approximation of the true safe set (see the
                    vault-safety-analysis-reconciliation memory for why that gap matters -- a
                    constraint-satisfying state need not be inside the TRUE invariant safe set,
                    and a value-based monitor has no way to notice the difference beyond however
                    well the critic happens to have learned it).
  RolloutMonitor -- PREDICTIVE: actually simulates applying u for one step, then the fallback
                    policy thereafter, until the trajectory reaches the target set, the failure
                    set, or the horizon runs out; returns the realized reach-avoid objective along
                    that trajectory (backward induction over the recorded margins -- positive iff
                    the target was actually reached without failing first). Needs a fallback
                    that's actually trying to reach a target (train_reach_avoid.py's policy, not
                    the avoid-only one), and a shadow MujocoPlant to simulate on without disturbing
                    the real one.

Common calling convention: `x` is always the reduced 4-state (needed by Fallback.action and by
ValueMonitor); `plant` is passed too, and only required by RolloutMonitor (to clone the exact
MuJoCo state, including roll/lateral drift the 4-state can't represent). Monitor/Intervention
methods accept `plant=None` and ignore it unless they need it.
"""
from __future__ import annotations

import numpy as np
import torch

from . import config as C
from . import contact_margin as CM
from . import f_cert as F
from . import target_margin as T
from .filter import control_grid
from .mujoco_plant import MujocoPlant
from .reach_avoid_value import realized_value


# --- fallback -----------------------------------------------------------------------------

class Fallback:
    """A safety-oriented backup control policy pi*(x, mu) -> u=[tau_L,tau_R]."""

    def action(self, x, mu) -> np.ndarray:
        raise NotImplementedError


class SACFallback(Fallback):
    """A trained SafetySAC/ReachAvoidSafetySAC actor used as the fallback pi*."""

    def __init__(self, model, tau_max: float | None = None):
        self.model = model
        self.tau_max = C.TAU_MAX if tau_max is None else tau_max

    def action(self, x, mu) -> np.ndarray:
        obs5 = np.append(np.asarray(x, np.float32), np.float32(mu))
        a, _ = self.model.predict(obs5, deterministic=True)
        return np.clip(a, -1.0, 1.0) * self.tau_max


# --- monitor ------------------------------------------------------------------------------

def _q_safe(model, tau_max, x, mu, controls) -> np.ndarray:
    """min_i Q_i(x, u) for a batch of candidate raw-torque controls u=[tau_L,tau_R]."""
    n = len(controls)
    obs5 = np.append(np.asarray(x, np.float32), np.float32(mu))
    obs = np.tile(obs5, (n, 1))
    u_norm = np.clip(np.asarray(controls, float) / tau_max, -1.0, 1.0).astype(np.float32)
    device = next(model.critic.parameters()).device
    with torch.no_grad():
        q = model.critic(torch.from_numpy(obs).to(device), torch.from_numpy(u_norm).to(device))
        q_min = torch.min(q[0], q[1]) if isinstance(q, tuple) else torch.stack(q).min(0).values
    return q_min.squeeze(-1).cpu().numpy()


class Monitor:
    """Safety monitor Delta*(x, u, mu) -> real number; see module docstring."""

    def evaluate(self, x, u, mu, plant=None) -> float:
        raise NotImplementedError

    def evaluate_batch(self, x, controls, mu, plant=None) -> np.ndarray:
        """Default: loop the scalar evaluate(). Override for a vectorized fast path."""
        return np.array([self.evaluate(x, u, mu, plant=plant) for u in controls])


class ValueMonitor(Monitor):
    """Value-based monitor: queries a trained critic Q(x,u) directly (no rollout)."""

    def __init__(self, model, tau_max: float | None = None):
        self.model = model
        self.tau_max = C.TAU_MAX if tau_max is None else tau_max

    def evaluate(self, x, u, mu, plant=None) -> float:
        return float(_q_safe(self.model, self.tau_max, x, mu, np.asarray([u]))[0])

    def evaluate_batch(self, x, controls, mu, plant=None) -> np.ndarray:
        return _q_safe(self.model, self.tau_max, x, mu, controls)


class RolloutMonitor(Monitor):
    """Predictive monitor: apply `u` for one step, then follow `fallback` until the target set,
    the failure set, or `horizon` steps elapse; return the realized reach-avoid objective.

    Realized objective = backward induction of V_t = min(g_t, max(l_t, V_{t+1})) over the recorded
    per-step (avoid margin g_t, target margin l_t), base case V_T = min(g_T, l_T) at the last
    simulated step -- the trajectory-realized analogue of the reach-avoid Bellman recursion. This
    is > 0 only if the target was reached without ever failing first: any failure step forces
    g_t < 0 there, which caps every V at-or-before that step (min(g_t, ...) <= g_t); running out
    of horizon without reaching target typically leaves l_T < 0 (not yet at rest), so the base
    case is <= 0 too. Only a genuine "reached target, never failed" trajectory keeps every g_t>=0
    and ends with l_T>=0, propagating a positive value all the way back to t=0.
    """

    def __init__(self, shadow_plant: MujocoPlant, fallback: Fallback, horizon: int = 300):
        self.shadow = shadow_plant
        self.fallback = fallback
        self.horizon = horizon

    def evaluate(self, x, u, mu, plant=None) -> float:
        if plant is None:
            raise ValueError("RolloutMonitor needs the live plant (plant=...) to clone state from")
        qpos, qvel, t = plant.get_full_state()
        self.shadow.set_full_state(qpos, qvel, t)

        gs, ls = [], []

        def _record(state):
            g = min(float(F.odd_margin(state)), CM.margin(self.shadow))
            l = float(T.target_margin(state))
            gs.append(g)
            ls.append(l)
            return g, l

        x_t = self.shadow.step(np.asarray(u, float))
        g, l = _record(x_t)
        if not (g < 0.0 or l >= 0.0):
            for _ in range(self.horizon - 1):
                u_fb = self.fallback.action(x_t, mu)
                x_t = self.shadow.step(u_fb)
                g, l = _record(x_t)
                if g < 0.0 or l >= 0.0:
                    break

        return realized_value(gs, ls)


# --- intervention -------------------------------------------------------------------------

class SwitchIntervention:
    """phi(x,u_task) = u_task if monitor(x,u_task)>=eps else fallback.action(x,mu). The simplest
    intervention scheme: either fully trusts the task action or fully defers to the fallback."""

    def __init__(self, monitor: Monitor, fallback: Fallback, eps: float = 0.0):
        self.monitor = monitor
        self.fallback = fallback
        self.eps = eps

    def __call__(self, x, u_task, mu, plant=None):
        """Return (u_safe, overridden)."""
        q_task = self.monitor.evaluate(x, u_task, mu, plant=plant)
        if q_task >= self.eps:
            return np.asarray(u_task, float), False
        return self.fallback.action(x, mu), True


class OptimizationIntervention:
    """phi(x,u_task) = argmin_{u in U : monitor(x,u,mu)>=threshold(x)} ||u-u_task|| -- the
    discrete analogue of a CBF-QP: minimum deviation from the task action subject to the
    monitor's constraint, rather than an all-or-nothing switch. Calls monitor.evaluate (or
    evaluate_batch, if the monitor provides a vectorized fast path) once per candidate action --
    this IS the "internal optimization loop calling the monitor multiple times for different
    candidate control iterations." Works with ANY monitor (value-based or rollout-based).

    threshold(x) = max((1-gamma)*V(x), eps_floor): a hybrid of a proportional decay-rate condition
    (bounds how fast the monitor's value may drop in one step, relative to V(x) = monitor's
    estimate at the fallback's own action) and a hard floor (backstop against compounding
    approximation error under sustained pressure -- see vault-safety-analysis-reconciliation)."""

    def __init__(self, monitor: Monitor, fallback: Fallback, gamma: float = 0.2,
                 eps_floor: float = 0.05, controls=None):
        assert 0.0 < gamma <= 1.0, "gamma is a fractional one-step decay rate, must be in (0,1]"
        self.monitor = monitor
        self.fallback = fallback
        self.gamma = gamma
        self.eps_floor = eps_floor
        self.U = control_grid(n=21) if controls is None else np.asarray(controls, float)

    def value(self, x, mu, plant=None) -> float:
        """V(x) via the monitor evaluated at the fallback's own action."""
        return self.monitor.evaluate(x, self.fallback.action(x, mu), mu, plant=plant)

    def __call__(self, x, u_task, mu, plant=None):
        """Return (u_safe, overridden)."""
        v_x = self.value(x, mu, plant=plant)
        thresh = max((1.0 - self.gamma) * v_x, self.eps_floor)
        q_task = self.monitor.evaluate(x, u_task, mu, plant=plant)
        if q_task >= thresh:
            return np.asarray(u_task, float), False
        qs = self.monitor.evaluate_batch(x, self.U, mu, plant=plant)
        feasible = qs >= thresh
        if not feasible.any():           # grid too coarse to find a feasible point -- fall back
            return self.U[int(np.argmax(qs))].copy(), True
        cand = self.U[feasible]
        dists = np.linalg.norm(cand - np.asarray(u_task, float), axis=1)
        return cand[int(np.argmin(dists))].copy(), True
