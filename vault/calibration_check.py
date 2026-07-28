"""calibration_check -- does the trained critic actually anticipate future safety violations?

A reach-avoid safety filter is only as good as its value function's ability to PREDICT trouble
before it happens. This script tests that directly: for a broad sample of states x (matching
C.DOMAIN_* -- the same domain ContactSafetyEnv resets across), it compares

  predicted := Q(x, pi_safe(x))                     (the critic's own claim about x)
  realized  := min margin actually seen while rolling pi_safe(x) forward K steps in the real plant

If the critic is well-calibrated, predicted should correlate strongly with realized, and in
particular: states with predicted < eps should mostly realize a violation (realized < 0) if not
corrected, and states with predicted >> eps should reliably stay safe. Poor correlation or a lot
of "predicted safe, realized unsafe" cases means the filter is flying blind, not anticipating.

  python -m vault.calibration_check --model vault/models/contact_safety_sac_v3 --n 200 --horizon 80
"""
from __future__ import annotations

import argparse

import numpy as np

from . import config as C
from . import contact_margin as CM
from . import f_cert as F
from .checkpoints import load_safety_sac
from .mujoco_plant import MujocoPlant


def _fallback_action(model, x, mu, tau_max=None):
    tau_max = C.TAU_MAX if tau_max is None else tau_max
    obs5 = np.append(np.asarray(x, np.float32), np.float32(mu))
    a, _ = model.predict(obs5, deterministic=True)
    return np.clip(a, -1.0, 1.0) * tau_max


def _q_at(model, x, mu, u, tau_max=None):
    tau_max = C.TAU_MAX if tau_max is None else tau_max
    import torch
    obs5 = np.append(np.asarray(x, np.float32), np.float32(mu))[None]
    u_norm = np.clip(np.asarray(u, float) / tau_max, -1.0, 1.0).astype(np.float32)[None]
    device = next(model.critic.parameters()).device
    with torch.no_grad():
        q = model.critic(torch.from_numpy(obs5).to(device), torch.from_numpy(u_norm).to(device))
        q_min = torch.min(q[0], q[1]) if isinstance(q, tuple) else torch.stack(q).min(0).values
    return float(q_min.item())


def run(model_path: str, n: int, horizon: int, mu: float, seed: int):
    model = load_safety_sac(model_path)
    rng = np.random.default_rng(seed)
    plant = MujocoPlant(wheel="cylinder", mu=mu, dt=0.0005, substeps=20, contact_geometry=True)

    predicted, realized = [], []
    for i in range(n):
        x0 = np.array([rng.uniform(*C.DOMAIN_V), rng.uniform(*C.DOMAIN_THETA),
                       rng.uniform(*C.DOMAIN_THETA_DOT), rng.uniform(*C.DOMAIN_PSI_DOT)])
        plant.reset(x0)
        x = plant.get_state()
        u0 = _fallback_action(model, x, mu)
        predicted.append(_q_at(model, x, mu, u0))

        worst = 1.0
        for _ in range(horizon):
            u = _fallback_action(model, x, mu)
            x = plant.step(u)
            g = min(float(F.odd_margin(x)), CM.margin(plant))
            worst = min(worst, g)
            if worst < -1.0:      # already badly failed, no need to keep simulating
                break
        realized.append(worst)

    predicted, realized = np.array(predicted), np.array(realized)
    corr = np.corrcoef(predicted, realized)[0, 1]

    print(f"n={n} horizon={horizon} mu={mu}")
    print(f"predicted: min {predicted.min():+.3f} max {predicted.max():+.3f} mean {predicted.mean():+.3f}")
    print(f"realized : min {realized.min():+.3f} max {realized.max():+.3f} mean {realized.mean():+.3f}")
    print(f"Pearson correlation(predicted, realized) = {corr:.3f}")

    for eps in (0.0, 0.05, 0.1):
        pred_safe = predicted >= eps
        real_safe = realized >= 0.0
        tp = np.sum(pred_safe & real_safe)
        fp = np.sum(pred_safe & ~real_safe)     # predicted safe, but actually violated -- the bad case
        fn = np.sum(~pred_safe & real_safe)     # predicted unsafe, but actually fine -- overly conservative
        tn = np.sum(~pred_safe & ~real_safe)
        print(f"  eps={eps:.2f}: predicted-safe-but-violated (dangerous false negatives) = {fp}/{n} "
              f"({100 * fp / n:.0f}%) | predicted-unsafe-but-fine = {fn}/{n} | "
              f"correctly-safe={tp} correctly-unsafe={tn}")
    return predicted, realized


def main():
    ap = argparse.ArgumentParser(description="Check Q-value calibration against realized rollouts")
    ap.add_argument("--model", type=str, default=str(C.MODELS / "contact_safety_sac_v3"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=80)
    ap.add_argument("--mu", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.model, args.n, args.horizon, args.mu, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
