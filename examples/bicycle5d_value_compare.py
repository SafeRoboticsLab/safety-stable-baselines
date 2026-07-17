"""Reach-avoid value function: PPO vs SAC, on the rescaled-l bicycle.

Trains ReachAvoidPPO (on-policy) and ReachAvoidSAC (off-policy), then renders
their value maps side-by-side on the 4 eval maps. The point:

* PPO learns V(s) only on its ON-POLICY tube; the value-map probes (v=0, every
  position, heading at the goal) are off that tube, so PPO's V extrapolates and
  looks asymmetric / noisy off-trajectory.
* SAC learns Q(s, a) over the action space; its V(s) = min_i Q_i(s, pi(s)) is
  calibrated off the on-policy tube -> a cleaner reachable-safe set.

Both share the rescaled l (piecewise: +1 at the goal center) that gives V real
positive dynamic range, so the V>=0 certificate is readable at all.

    python examples/bicycle5d_value_compare.py                 # wandb + PNGs
    python examples/bicycle5d_value_compare.py --no-wandb --out /tmp/vcmp
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from safety_sb3 import ReachAvoidPPO, ReachAvoidSAC
from safety_sb3.testing.bicycle5d_vec import BicycleGoalVec
from safety_sb3.testing.bicycle5d_render import (
  EVAL_MAPS, compare_value_maps, multi_car_rollout)


def coverage(m):
  r = t = 0
  for cfg in EVAL_MAPS:
    _, st = multi_car_rollout(m, cfg)
    r += sum(s == "reached" for s in st); t += len(st)
  return r / t


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--ppo-steps", type=int, default=1_500_000)
  p.add_argument("--sac-steps", type=int, default=400_000)
  p.add_argument("--no-wandb", action="store_true")
  p.add_argument("--out", default=None, help="also save comparison PNGs here")
  p.add_argument("--seed", type=int, default=0)
  a = p.parse_args()

  run = None
  if not a.no_wandb:
    import wandb
    run = wandb.init(project="safety_sb3", group="bicycle5d-value-compare",
                     name="ppo-vs-sac", job_type="value-compare",
                     config=dict(ppo_steps=a.ppo_steps, sac_steps=a.sac_steps))

  import matplotlib.pyplot as plt
  from safety_sb3.testing.bicycle5d_render import value_map
  if a.out:
    os.makedirs(a.out, exist_ok=True)

  def log_per_model(tag, model):
    """Log this model's coverage + its 4 value maps NOW (so wandb populates
    incrementally, not only at the very end)."""
    cov = coverage(model)
    print(f"    {tag} coverage = {cov:.0%}", flush=True)
    if run is not None:
      import wandb
      run.log({f"coverage/{tag}": cov})
      for cfg in EVAL_MAPS:
        fig = value_map(model, cfg, title=tag)
        run.log({f"value_{tag}/{cfg['name']}": wandb.Image(fig)})
        plt.close(fig)
    return cov

  print("=== training ReachAvoidPPO (on-policy, wide spawn, adaptive_lr) ===", flush=True)
  ppo = ReachAvoidPPO("MlpPolicy", BicycleGoalVec(256, seed=a.seed, spawn="wide"),
                      n_steps=64, batch_size=4096, gamma=0.99, ent_coef=1e-3,
                      learning_rate=5e-4, adaptive_lr=True, desired_kl=0.01,
                      seed=a.seed, verbose=0, device="cpu")
  ppo.learn(a.ppo_steps)
  ppo_cov = log_per_model("PPO", ppo)                    # <-- wandb populates here

  print("=== training ReachAvoidSAC (off-policy) ===", flush=True)
  sac = ReachAvoidSAC("MlpPolicy", BicycleGoalVec(16, seed=a.seed, spawn="wide"),
                      buffer_size=500_000, learning_starts=5000, batch_size=512,
                      train_freq=(16, "step"), gradient_steps=16, gamma=0.99,
                      seed=a.seed, verbose=0, device="cpu")
  sac.learn(a.sac_steps)
  sac_cov = log_per_model("SAC", sac)

  # the side-by-side comparison (shared color scale)
  for cfg in EVAL_MAPS:
    fig = compare_value_maps({"PPO (on-policy V)": ppo, "SAC (off-policy Q)": sac},
                             cfg, suptitle=f"reach-avoid value(x,y) — {cfg['name']}")
    if run is not None:
      import wandb
      run.log({f"compare/{cfg['name']}": wandb.Image(fig)})
    if a.out:
      fig.savefig(os.path.join(a.out, f"value_{cfg['name']}.png"), dpi=110,
                  bbox_inches="tight")
    plt.close(fig)

  print(f"\nPPO coverage {ppo_cov:.0%}  |  SAC coverage {sac_cov:.0%}")
  if run is not None:
    run.summary.update({"ppo_coverage": ppo_cov, "sac_coverage": sac_cov})
    run.finish()
  if a.out:
    print(f"wrote comparison PNGs to {a.out}")


if __name__ == "__main__":
  main()
