"""reach_avoid_eval -- comparative rollout evaluation: avoid-only vs reach-avoid fallback.

Samples initial states spanning the CONSTRAINT SET (not the wider training domain -- states
already in violation at t=0 aren't a fair test of "can this fallback keep the robot safe"; see the
vault-safety-analysis-reconciliation memory), rolls out each of the two trained fallback policies
open-loop (no task command, no filter -- this evaluates the fallback itself) for up to
`--horizon` steps (default 300 = 3s), and classifies each rollout into exactly one of:

  (a) reached target safely     -- g(x_t) >= 0 for all t, and l(x_t) >= 0 at some t
  (b) safe but didn't reach     -- g(x_t) >= 0 for all t, but l never reached >= 0 by the horizon
  (c) failed before reaching    -- g(x_t) < 0 at some t, before l ever reached >= 0

Intuitively (per the reach-avoid formulation), the avoid-only fallback should show mostly (a)/(b)
with very few (c) (it wasn't trained to reach a stop, but it also wasn't trained to avoid it,
so wandering near the target region is incidental, not by design), while the reach-avoid fallback
should show mostly (a) -- if this robot's constraint set is (as expected) mostly a "come to a
stop" reachable region, reach-avoid training should convert most safe states into target-reaching
ones without materially increasing (c) (the reach term shouldn't make the fallback LESS safe).

  python -m vault.reach_avoid_eval --n 100 --horizon 300
"""
from __future__ import annotations

import argparse

import numpy as np
from safety_sb3 import SafetySAC

from . import config as C
from . import contact_margin as CM
from . import f_cert as F
from . import target_margin as T
from .mujoco_plant import MujocoPlant
from .reach_avoid_value import realized_value
from .safety_filter import SACFallback

Outcome = str  # one of "reached", "safe_no_reach", "failed"


def _sample_constraint_set_state(rng, plant: MujocoPlant) -> np.ndarray | None:
    """One rejection-sampling draw from C.DOMAIN_* that's actually inside the constraint set
    (odd_margin >= 0 AND contact_margin.margin >= 0 at that state). Returns None on a rejected
    draw so the caller can just retry."""
    x0 = np.array([rng.uniform(*C.DOMAIN_V), rng.uniform(*C.DOMAIN_THETA),
                   rng.uniform(*C.DOMAIN_THETA_DOT), rng.uniform(*C.DOMAIN_PSI_DOT)])
    if float(F.odd_margin(x0)) < 0.0:
        return None
    plant.reset(x0)
    if CM.margin(plant) < 0.0:
        return None
    return x0


def rollout_outcome(plant: MujocoPlant, fallback: SACFallback, x0: np.ndarray, mu: float,
                    horizon: int) -> tuple[Outcome, int, float]:
    """Roll out `fallback` open-loop from x0 for up to `horizon` steps. Returns
    (outcome, steps, realized_reach_avoid_score) -- the score is reach_avoid_value.realized_value
    over the recorded trajectory, positive iff outcome == "reached"."""
    plant.reset(x0)
    x = plant.get_state()
    gs, ls = [], []
    for t in range(horizon):
        g = min(float(F.odd_margin(x)), CM.margin(plant))
        l = float(T.target_margin(x))
        gs.append(g)
        ls.append(l)
        if g < 0.0:
            return "failed", t, realized_value(gs, ls)
        if l >= 0.0:
            return "reached", t, realized_value(gs, ls)
        x = plant.step(fallback.action(x, mu))
    return "safe_no_reach", horizon, realized_value(gs, ls)


def run(n: int, horizon: int, mu: float, seed: int, avoid_only_path: str, reach_avoid_path: str):
    rng = np.random.default_rng(seed)
    plant = MujocoPlant(wheel="cylinder", mu=mu, dt=0.0005, substeps=20, contact_geometry=True)

    x0s = []
    while len(x0s) < n:
        x0 = _sample_constraint_set_state(rng, plant)
        if x0 is not None:
            x0s.append(x0)
    print(f"sampled {n} initial states from the constraint set (odd_margin>=0, contact_margin>=0)\n")

    fallbacks = {
        "avoid-only": SACFallback(SafetySAC.load(avoid_only_path)),
        "reach-avoid": SACFallback(SafetySAC.load(reach_avoid_path)),
    }

    results = {name: {"reached": 0, "safe_no_reach": 0, "failed": 0} for name in fallbacks}
    steps_to_reach = {name: [] for name in fallbacks}
    scores = {name: [] for name in fallbacks}
    for x0 in x0s:
        for name, fb in fallbacks.items():
            outcome, t, score = rollout_outcome(plant, fb, x0, mu, horizon)
            results[name][outcome] += 1
            scores[name].append(score)
            if outcome == "reached":
                steps_to_reach[name].append(t)

    print(f"{'policy':12s} {'reached':>10s} {'safe/no-reach':>14s} {'failed':>8s}   "
          f"mean score   mean steps-to-reach")
    for name in fallbacks:
        r = results[name]
        mean_steps = np.mean(steps_to_reach[name]) if steps_to_reach[name] else float("nan")
        steps_str = f"{mean_steps:.1f} steps ({mean_steps * C.DT:.2f}s)" if steps_to_reach[name] else "n/a"
        print(f"{name:12s} {r['reached']:>6d}/{n:<3d} {r['safe_no_reach']:>10d}/{n:<3d} "
              f"{r['failed']:>4d}/{n:<3d}   {np.mean(scores[name]):+.3f}      {steps_str}")
    return results


def main():
    ap = argparse.ArgumentParser(description="Compare avoid-only vs reach-avoid fallback rollouts")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--horizon", type=int, default=300, help="steps (default 300 = 3s at DT=0.01)")
    ap.add_argument("--mu", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--avoid-only", type=str, default=str(C.MODELS / "contact_safety_sac_v3"))
    ap.add_argument("--reach-avoid", type=str, default=str(C.MODELS / "reach_avoid_safety_sac"))
    args = ap.parse_args()
    run(args.n, args.horizon, args.mu, args.seed, args.avoid_only, args.reach_avoid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
