"""In-the-loop value-filter evaluation in MuJoCo.

Runs the DEPLOYED controller (from the vault-controller repo) in the MuJoCo plant under aggressive
commands from in-set initial conditions, with the value filter ON vs OFF, and reports forward-
invariance (0 topple / liftoff / ODD-exit) plus the intervention rate (least-restrictiveness).

REQUIRES the vault-controller repo (the real deployed controller; no fallback by design):
    git clone <vault-controller>
    make -C vault-controller/balance_controller/c lib
    export PYTHONPATH=/path/to/vault-controller:$PYTHONPATH
    python -m vault.evaluate
"""
from __future__ import annotations

import math

import numpy as np

from . import config as C
from .filter import ValueFilter
from .mujoco_plant import MujocoPlant

try:
    from balance_controller.c_coupled_controller import CCoupledController
except ImportError as e:  # fail loud — the eval is meaningless without the real controller
    raise SystemExit(
        "vault.evaluate requires the vault-controller repo (the deployed C iLQR), not on PYTHONPATH.\n"
        "  1) clone vault-controller beside this repo\n"
        "  2) make -C vault-controller/balance_controller/c lib\n"
        "  3) export PYTHONPATH=/path/to/vault-controller:$PYTHONPATH\n"
        f"(import error: {e})"
    )

ROLL_LIMIT_DEG = 33.0   # axle roll = wheel liftoff / rollover onset
N_ICS, HORIZON = 100, 150


def _run(mu, plant, ctrl, vf, eps, filter_on, seed=7):
    rng = np.random.default_rng(seed)
    ics = []
    while len(ics) < N_ICS:
        ic = np.array([rng.uniform(0.0, 1.5), rng.uniform(-0.5, 0.5),
                       rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)])
        if vf._V(ic[None], mu)[0] >= eps:                              # certified in-set
            ics.append(ic)
    topple = lift = oodd = interv = steps = 0
    for ic in ics:
        ref = np.array([1.0, float(rng.uniform(-2.5, 2.5))])           # aggressive forward + turn
        plant.reset(ic.copy()); ctrl.reset()
        x = plant.get_state()
        for t in range(HORIZON):
            u = np.clip(np.asarray(ctrl.step(x, ref, t * C.DT), float), -C.TAU_MAX, C.TAU_MAX)
            if filter_on:
                u, overridden = vf.filter(x, u, mu)
                interv += int(overridden)
            x = plant.step(u); steps += 1
            if not np.all(np.isfinite(x)) or abs(float(x[1])) > C.THETA_MAX:
                topple += 1; break
            if abs(plant.roll_deg()) > ROLL_LIMIT_DEG:
                lift += 1; break
            if not (C.V_ODD[0] <= float(x[0]) <= C.V_ODD[1] + 0.06) or abs(float(x[3])) > C.PSI_ODD + 0.1:
                oodd += 1; break
    return topple, lift, oodd, interv, steps


def main():
    print(f"value filter on the DEPLOYED controller | {N_ICS} in-set ICs/mu, aggressive refs, MuJoCo\n")
    ctrl = CCoupledController()
    vf = ValueFilter.from_mlp()
    passed = {}
    for mu in (0.3, 1.0):
        plant = MujocoPlant(wheel="cylinder", mu=mu, dt=0.0005, substeps=20)
        to, li, oo, _, _ = _run(mu, plant, ctrl, vf, 0.0, filter_on=False)
        print(f"  mu={mu}: OFF  -> topple {to} liftoff {li} oodd {oo}")
        for eps in (0.0, 0.05, 0.10):
            to, li, oo, iv, st = _run(mu, plant, ctrl, vf, eps, filter_on=True)
            invariant = (to == 0 and li == 0 and oo == 0)
            print(f"    ON eps={eps:.2f} -> topple {to} liftoff {li} oodd {oo} | "
                  f"intervene {100 * iv / max(st, 1):.0f}%{'  <-- forward-invariant' if invariant else ''}")
            if invariant and mu not in passed:
                passed[mu] = eps
    ok = all(m in passed for m in (0.3, 1.0))
    print(f"\nGATE: {'PASS -- forward-invariant at eps=' + str(passed) if ok else 'REVIEW -- some mu lacked a clean eps'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
