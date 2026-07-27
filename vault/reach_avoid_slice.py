"""reach_avoid_slice -- 2D (v, theta) outcome grid for both fallback policies, theta_dot=psi_dot=0.

Grids the (v, theta) plane (theta_dot=psi_dot=0 held fixed) over C.DOMAIN_V x C.DOMAIN_THETA
(slightly wider than the constraint set, so the constraint-set boundary itself is visible in the
plot), classifies EVERY grid cell -- in or out of the constraint set -- via
reach_avoid_eval.rollout_outcome for both the avoid-only and reach-avoid fallback, and writes the
result as JSON for plotting. Reuses the exact same rollout/classification logic already validated
in reach_avoid_eval.py, just swapping random constraint-set sampling for a regular grid (including
cells outside the constraint set, which trivially classify as "failed" at t=0 -- that's what
delineates the constraint-set boundary in the plot).

  python -m vault.reach_avoid_slice --n 31 --horizon 300
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from . import config as C
from .checkpoints import load_safety_sac
from .mujoco_plant import MujocoPlant
from .reach_avoid_eval import rollout_outcome
from .safety_filter import SACFallback


def run(n: int, horizon: int, mu: float, model_paths: dict[str, str], out_path: str):
    """model_paths: {display_name: checkpoint_path (no .zip)} -- any number of policies."""
    v_grid = np.linspace(*C.DOMAIN_V, n)
    theta_grid = np.linspace(*C.DOMAIN_THETA, n)

    plant = MujocoPlant(wheel="cylinder", mu=mu, dt=0.0005, substeps=20, contact_geometry=True)
    fallbacks = {name: SACFallback(load_safety_sac(path, device="cpu"))
                for name, path in model_paths.items()}

    result = {"v_grid": v_grid.tolist(), "theta_grid": theta_grid.tolist(), "n": n,
             "horizon": horizon, "mu": mu, "grids": {}}
    for name, fb in fallbacks.items():
        t0 = time.time()
        grid = []
        for theta in theta_grid:
            row = []
            for v in v_grid:
                x0 = np.array([v, theta, 0.0, 0.0])
                outcome, _, _ = rollout_outcome(plant, fb, x0, mu, horizon)
                row.append(outcome)
            grid.append(row)
        result["grids"][name] = grid
        print(f"{name}: {n * n} cells in {time.time() - t0:.1f}s")

    with open(out_path, "w") as f:
        json.dump(result, f)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description="2D (v,theta) outcome grid for N fallback policies")
    ap.add_argument("--n", type=int, default=31, help="grid resolution per axis")
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--mu", type=float, default=0.8)
    ap.add_argument("--model", action="append", default=[], metavar="NAME=PATH",
                     help="a named policy checkpoint (no .zip); repeat for multiple. If omitted, "
                          "defaults to avoid_only + reach_avoid for backward compatibility.")
    ap.add_argument("--out", type=str, default=str(C.MODELS / "reach_avoid_slice.json"))
    args = ap.parse_args()
    if args.model:
        model_paths = dict(m.split("=", 1) for m in args.model)
    else:
        model_paths = {"avoid_only": str(C.MODELS / "contact_safety_sac_v3"),
                       "reach_avoid": str(C.MODELS / "reach_avoid_safety_sac_ext600k")}
    run(args.n, args.horizon, args.mu, model_paths, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
