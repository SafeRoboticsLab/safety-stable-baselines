"""Train SafetySAC (reach-avoid V + fallback pi_safe) on BalanceSafetyEnv.

SafetySAC (Fisac et al., reach-avoid RL) learns the reach-avoid value V(s) = min-over-trajectory
g(s) and a fallback policy pi_safe. The learned V is the function approximator counterpart of the
exact grid value (grid.py); compare them with distill.py / grid.py for over-optimism. This script
validates the env API, runs the training, and prints the learned V at probe states.

  python -m vault.train --steps 3000      # smoke (validate API + short train)
  python -m vault.train --steps 300000    # full
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from safety_sb3 import SafetySAC
from stable_baselines3.common.env_checker import check_env

from . import config as C
from .env import BalanceSafetyEnv


def main():
    ap = argparse.ArgumentParser(description="Train SafetySAC on the balance reach-avoid env")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--learning-starts", type=int, default=500)
    ap.add_argument("--gamma", type=float, default=0.999, help="safety discount")
    ap.add_argument("--save", type=str, default=str(C.MODELS / "balance_safety_sac"))
    args = ap.parse_args()

    env = BalanceSafetyEnv(seed=0)
    check_env(env, warn=True)
    print("env API OK")
    gs = []
    env.reset(seed=0)
    for _ in range(200):
        _, r, term, trunc, _ = env.step(env.action_space.sample())
        gs.append(r)
        if term or trunc:
            env.reset()
    print(f"random-policy margins g: min {min(gs):+.2f} max {max(gs):+.2f}  (g<0 = breach)\n")

    print(f"SafetySAC train: {args.steps} steps, gamma={args.gamma}")
    model = SafetySAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=100_000,
                      learning_starts=args.learning_starts, batch_size=256, tau=0.01,
                      gamma=args.gamma, train_freq=(1, "step"), gradient_steps=1,
                      ent_coef="auto", seed=0, device="cpu", verbose=1)
    t0 = time.time()
    model.learn(args.steps, progress_bar=False)
    C.MODELS.mkdir(exist_ok=True)
    model.save(args.save)
    print(f"trained {args.steps} steps in {time.time() - t0:.1f}s -> {args.save}.zip")

    import torch
    print("\nlearned V ~= min_i q_i(s, pi_safe(s)) at probe states:")
    for x in ([0.5, 0.0, 0.0, 0.0, 0.8], [1.0, 0.3, 0.0, 1.5, 0.8],
              [1.0, 0.0, 0.0, 0.0, 0.2], [1.0, 1.0, 3.0, 0.0, 0.3]):
        a, _ = model.predict(np.array(x, np.float32), deterministic=True)
        with torch.no_grad():
            q = model.critic(torch.from_numpy(np.array([x], np.float32)),
                             torch.from_numpy(np.asarray(a, np.float32)[None]))
            v = float(torch.min(q[0], q[1])) if isinstance(q, tuple) else float(min(t.item() for t in q))
        print(f"  x={x} -> pi_safe={np.round(a, 2)}  V~{v:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
