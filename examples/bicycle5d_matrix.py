"""The post-refactor validation matrix: all 8 learners on bicycle5d.

Trains every (mode x family x players) cell of the MAP product that bicycle5d
can express -- 2 modes (Safety = avoid, ReachAvoid) x 2 families (PPO, SAC) x 2
player counts (1P, 2P) = 8 learners -- and asserts the env's reason for
existing, per (family, player-count) cell:

    reach_rate(reach-avoid)  >>  reach_rate(avoid)

**avoid is the negative control.** Nothing rewards motion and `g > 0` already
holds at spawn, so the optimal avoid policy is to sit still. If a reach-avoid
learner also sits still the reach-avoid anchor is broken (a `g`-anchored backup
values loitering at `V = g > 0`, which beats driving) and the two problems
become indistinguishable -- a regression that value unit tests cannot see. See
RELEASE_NOTES v0.2.0 and :mod:`safety_sb3.backups`.

Run the whole matrix (no wandb needed)::

    python examples/bicycle5d_matrix.py --out-dir /tmp/bicycle5d_matrix

or one cell while iterating::

    python examples/bicycle5d_matrix.py --cells ppo-2p --seeds 0 1 2

Writes per-cell GIFs + a `results.json` / `matrix.md` under `--out-dir`, and
prints the PASS/FAIL table. Evaluation is `bicycle5d_demo.rollout` -- one source
of truth for what "reached" means, and it runs on the numpy `BicycleGoal`
whatever the training backend is.

**Backends** (`--backend`, default `auto`). `auto` = numpy for PPO, tensor for
SAC, which is what the defaults below are tuned for:

* `numpy` -- `BicycleGoalVec` on CPU. What the PPO cells were logged with; PPO
  is on-policy and CPU-bound by its 256-env rollout either way.
* `tensor` -- `BicycleGoalTensorVec` on GPU. The SAC default: no numpy on the hot
  path (the learners see `is_tensor_env` and switch to the torch-native collect +
  `TensorReplayBuffer`), and 256 envs of data instead of 16.

**`num_envs` is not a throughput knob for SAC -- it sets the update-to-data
ratio.** SB3 counts `train_freq` in VECTOR steps, so `train_freq=(16, "step")`
with `gradient_steps=16` runs 16 gradient steps per `16 * num_envs` env-steps.
Over 2M env-steps that is 124,992 gradient steps at 16 envs but 7,808 at 256 --
a 16x difference at the SAME data budget, and most of why the 16-env CPU cells
took ~35 min (they were buying 16x the updates, not merely running a slow env).

Raising `num_envs` alone therefore changes the learning problem, measurably.
Measured here, ReachAvoidSAC1P / 2M steps / seed 0: at 7,808 gradient steps the
policy is under-trained (0% reach and only 44% SAFE -- it wanders into obstacles);
at 124,992 it converges to exactly what the 16-env CPU run converged to (0%
reach, 100% safe, 0.4 m of path -- loitering). Same budget, same answer, whatever
the backend.

So the SAC defaults hold the GRADIENT-STEP BUDGET fixed at the documented
reference (`docs/environments/bicycle5d.md`: 124,992 per 2M steps) and vary only
how much data feeds it -- `--sac-envs 256 --sac-train-freq 1` on tensor,
`16`/`16` on numpy. Both are 124,992 gradient steps; tensor just gets there ~4x
faster (8.6 min vs ~36 min per arm). The resolved count is printed and stored in
`results.json`, so it can never go unnoticed again.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bicycle5d_demo import rollout                                # noqa: E402
from safety_sb3 import (ReachAvoidPPO1P, ReachAvoidPPO2P, ReachAvoidSAC1P,   # noqa: E402
                        ReachAvoidSAC2P, SafetyPPO1P, SafetyPPO2P,
                        SafetySAC1P, SafetySAC2P)
from safety_sb3.testing.bicycle5d_render import (EVAL_MAPS, map_gif,   # noqa: E402
                                                 multi_car_rollout)
from safety_sb3.testing.bicycle5d_tensor import BicycleGoalTensorVec  # noqa: E402
from safety_sb3.testing.bicycle5d_vec import BicycleGoalVec       # noqa: E402

#: cell -> (reach-avoid class, avoid class, adversary)
CELLS = {
  "ppo-1p": (ReachAvoidPPO1P, SafetyPPO1P, False),
  "ppo-2p": (ReachAvoidPPO2P, SafetyPPO2P, True),
  "sac-1p": (ReachAvoidSAC1P, SafetySAC1P, False),
  "sac-2p": (ReachAvoidSAC2P, SafetySAC2P, True),
}

#: the contrast is called PASS when reach-avoid beats avoid by this margin
#: (same bar `bicycle5d_demo.py` prints against).
CONTRAST_MARGIN = 0.40


#: SAC gradient steps per 2M env-steps in the documented reference config
#: (16 envs, train_freq=(16,"step"), gradient_steps=16). The defaults below keep
#: this fixed across backends -- see the module docstring.
REFERENCE_GRAD_STEPS = 124_992


def plan(a, family):
  """Resolve (backend, n_envs, device, train_freq) for one family.

  Kept in one place because these are not independent: the model's device must
  be the env's device, and for SAC ``n_envs`` and ``train_freq`` jointly set the
  gradient-step budget (module docstring)."""
  backend = a.backend
  if backend == "auto":
    backend = "tensor" if family == "sac" else "numpy"
  n_envs = a.sac_envs if family == "sac" else a.ppo_envs
  if n_envs is None:                       # family default, per backend
    n_envs = 256 if (family == "ppo" or backend == "tensor") else 16
  device = a.device
  if device == "auto":
    device = ("cuda" if backend == "tensor" and th.cuda.is_available()
              else "cpu")
  # train_freq is in VECTOR steps; the default keeps gradient_steps/env-step at
  # the reference 1/16 -> tf = gradient_steps * 16 / n_envs.
  train_freq = a.sac_train_freq
  if train_freq is None:
    train_freq = max(1, round(a.sac_gradient_steps * 16 / n_envs))
  return backend, int(n_envs), device, int(train_freq)


def grad_steps(steps, n_envs, train_freq, gradient_steps):
  """Gradient steps SB3 will actually take -- train_freq counts VECTOR steps."""
  return int(steps // (train_freq * n_envs)) * gradient_steps


def build(cls, adversary, seed, family, n_envs, spawn, backend, device,
          train_freq=16, gradient_steps=16):
  """Model + env for one arm. Hyperparameters mirror `bicycle5d_demo.py`."""
  if backend == "tensor":
    venv = BicycleGoalTensorVec(n_envs, adversary=adversary, seed=seed,
                                spawn=spawn, device=device)
  else:
    venv = BicycleGoalVec(n_envs, adversary=adversary, seed=seed, spawn=spawn)
  kw = dict(ctrl_action_dim=2) if adversary else {}
  if family == "sac":
    return cls("MlpPolicy", venv, seed=seed, buffer_size=500_000,
               learning_starts=5000, batch_size=512,
               train_freq=(train_freq, "step"), gradient_steps=gradient_steps,
               gamma=0.99, verbose=0, device=device, **kw)
  return cls("MlpPolicy", venv, seed=seed, n_steps=64, batch_size=4096,
             gamma=0.99, ent_coef=1e-3, learning_rate=5e-4, adaptive_lr=True,
             desired_kl=0.01, verbose=0, device=device, **kw)


def measure(model, adversary, n=32, eval_seed=1234, from_standstill=False):
  """Episode stats over `n` randomized maps. Uses the demo's rollout verbatim."""
  rng = np.random.default_rng(eval_seed)
  R = [rollout(model, adversary, int(rng.integers(1 << 30)), from_standstill)
       for _ in range(n)]
  return dict(
    reach_rate=float(np.mean([r["reached"] for r in R])),
    safe_rate=float(np.mean([not r["collided"] for r in R])),
    ep_len=float(np.mean([len(r["trail"]) - 1 for r in R])),
    path_len=float(np.mean([r["path"] for r in R])),   # ~0 => never moved
    dist_to_goal=float(np.mean([r["dist"] for r in R])),
  )


class ReachProbe(BaseCallback):
  """Records (steps, reach_rate, path_len) during training -- the learning curve.

  Cheap (12 rollouts), so a diverged or never-converging run leaves evidence
  instead of one final number. `path_len ~ 0` is the loiter signature.
  """

  def __init__(self, adversary, every):
    super().__init__()
    self.adversary, self.every, self._next, self.curve = adversary, every, 0, []

  def _on_step(self) -> bool:
    if self.num_timesteps < self._next:
      return True
    self._next = self.num_timesteps + self.every
    m = measure(self.model, self.adversary, n=12, eval_seed=7)
    self.curve.append(dict(steps=int(self.num_timesteps),
                           reach_rate=m["reach_rate"], safe_rate=m["safe_rate"],
                           path_len=round(m["path_len"], 3)))
    return True


def coverage(model, adversary):
  """HONEST reach rate: 8 coverage spawns (from standstill) x 4 eval maps."""
  reached = safe = total = 0
  for m in EVAL_MAPS:
    _, status = multi_car_rollout(model, m, adversary=adversary)
    reached += sum(s == "reached" for s in status)
    safe += sum(s != "collided" for s in status)
    total += len(status)
  return reached / total, safe / total


def run_arm(cell, arm, cls, adversary, seed, steps, a, out_dir):
  family = cell.split("-")[0]
  backend, n_envs, device, train_freq = plan(a, family)
  model = build(cls, adversary, seed, family, n_envs, a.spawn, backend, device,
                train_freq, a.sac_gradient_steps)
  gs = (grad_steps(steps, n_envs, train_freq, a.sac_gradient_steps)
        if family == "sac" else None)
  if gs is not None:
    print(f"  [{cell} {arm}] {backend}@{n_envs}/{device} "
          f"train_freq={train_freq} x gradient_steps={a.sac_gradient_steps} "
          f"-> {gs:,} gradient steps over {steps:,} env-steps "
          f"(reference: {REFERENCE_GRAD_STEPS:,})", flush=True)
  probe = ReachProbe(adversary, a.probe_every)
  t0 = time.time()
  model.learn(total_timesteps=steps, callback=probe)
  wall = time.time() - t0

  r = measure(model, adversary)
  r["curve"] = probe.curve
  r["reach_from_standstill"] = measure(model, adversary,
                                       from_standstill=True)["reach_rate"]
  r["coverage_reach"], r["coverage_safe"] = coverage(model, adversary)
  r.update(cell=cell, arm=arm, algo=cls.__name__, seed=seed, steps=steps,
           wall_clock_s=round(wall, 1), n_envs=n_envs, backend=backend,
           device=device, steps_per_s=round(steps / max(wall, 1e-9)),
           train_freq=train_freq, grad_steps=gs)

  if a.gifs:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=64)
    for m in EVAL_MAPS[:a.gif_maps]:
      frames = map_gif(model, m, adversary=adversary, fig=fig, ax=ax,
                       title=f"{cls.__name__} s{seed} @ {steps/1e6:.1f}M")
      if frames:
        p = os.path.join(out_dir, f"{cell}_{arm}_{m['name']}_seed{seed}.gif")
        Image.fromarray(frames[0]).save(
          p, save_all=True, append_images=[Image.fromarray(f) for f in frames[1:]],
          duration=50, loop=0)
        r.setdefault("gifs", []).append(p)
    plt.close(fig)

  if a.save_dir:
    os.makedirs(a.save_dir, exist_ok=True)
    model.save(os.path.join(a.save_dir, f"{cell}_{arm}_seed{seed}"))
  return r


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--cells", nargs="+", default=list(CELLS), choices=list(CELLS))
  p.add_argument("--arms", nargs="+", default=["reach-avoid", "avoid"],
                 choices=["reach-avoid", "avoid"])
  p.add_argument("--seeds", nargs="+", type=int, default=[0])
  p.add_argument("--ppo-steps", type=int, default=2_000_000)
  p.add_argument("--sac-steps", type=int, default=2_000_000)
  p.add_argument("--backend", choices=["auto", "numpy", "tensor"],
                 default="auto", help="auto = numpy for PPO, tensor for SAC")
  p.add_argument("--device", default="auto",
                 help="'auto' = cuda on the tensor backend, else cpu")
  p.add_argument("--ppo-envs", type=int, default=None,
                 help="default 256 on either backend")
  p.add_argument("--sac-envs", type=int, default=None,
                 help="default 256 (tensor) / 16 (numpy) — NOT interchangeable, "
                      "num_envs sets the update-to-data ratio")
  p.add_argument("--sac-train-freq", type=int, default=None,
                 help="in VECTOR steps. Default keeps the gradient-step budget "
                      "at the reference 124,992 per 2M env-steps: "
                      "gradient_steps * 16 / sac_envs (=1 at 256 envs)")
  p.add_argument("--sac-gradient-steps", type=int, default=16)
  p.add_argument("--spawn", choices=["edge", "wide", "map"], default="wide")
  p.add_argument("--out-dir", default="experiments/bicycle5d_matrix")
  p.add_argument("--save-dir", default=None, help="save trained models here")
  p.add_argument("--no-gifs", dest="gifs", action="store_false", default=True)
  p.add_argument("--gif-maps", type=int, default=1, help="how many EVAL_MAPS to animate")
  p.add_argument("--probe-every", type=int, default=200_000,
                 help="in-training reach-rate probe interval (steps)")
  a = p.parse_args()

  os.makedirs(a.out_dir, exist_ok=True)
  rows, t_all = [], time.time()
  for cell in a.cells:
    ra_cls, av_cls, adv = CELLS[cell]
    steps = a.sac_steps if cell.startswith("sac") else a.ppo_steps
    for seed in a.seeds:
      for arm in a.arms:               # reach-avoid first: the interesting arm
        cls = ra_cls if arm == "reach-avoid" else av_cls
        r = run_arm(cell, arm, cls, adv, seed, steps, a, a.out_dir)
        rows.append(r)
        print(f"[{cell:7s} {arm:11s} {r['algo']:16s} s{seed}] "
              f"reach={r['reach_rate']:5.0%} safe={r['safe_rate']:5.0%} "
              f"cover={r['coverage_reach']:5.0%} standstill={r['reach_from_standstill']:5.0%} "
              f"path={r['path_len']:5.2f}m ep_len={r['ep_len']:5.0f} "
              f"({r['wall_clock_s']:.0f}s, {r['backend']}@{r['n_envs']}/"
              f"{r['device']}, {r['steps_per_s']:,} steps/s)", flush=True)

  # --- the assertion, per (family, player-count) cell ------------------------
  verdicts = []
  for cell in a.cells:
    for seed in a.seeds:
      def pick(arm):
        m = [r for r in rows if r["cell"] == cell and r["seed"] == seed
             and r["arm"] == arm]
        return m[0] if m else None
      ra, av = pick("reach-avoid"), pick("avoid")
      if not (ra and av):
        continue
      for key in ("reach_rate", "coverage_reach"):
        verdicts.append(dict(
          cell=cell, seed=seed, metric=key, ra=ra[key], avoid=av[key],
          passed=bool(ra[key] > av[key] + CONTRAST_MARGIN)))

  lines = ["| cell | seed | metric | reach-avoid | avoid | verdict |",
           "|---|---|---|---|---|---|"]
  for v in verdicts:
    lines.append(f"| {v['cell']} | {v['seed']} | {v['metric']} | {v['ra']:.0%} | "
                 f"{v['avoid']:.0%} | {'PASS' if v['passed'] else 'FAIL'} |")
  table = "\n".join(lines)
  print(f"\n=== contrast: reach_rate(reach-avoid) > reach_rate(avoid) + "
        f"{CONTRAST_MARGIN:.0%} ===\n{table}")
  print(f"\ntotal wall clock: {time.time() - t_all:.0f}s")

  with open(os.path.join(a.out_dir, "results.json"), "w") as f:
    json.dump(dict(rows=rows, verdicts=verdicts, args=vars(a)), f, indent=2)
  with open(os.path.join(a.out_dir, "matrix.md"), "w") as f:
    f.write("# bicycle5d validation matrix\n\n" + table + "\n\n")
    f.write("| cell | arm | algo | seed | steps | reach | safe | coverage_reach | "
            "standstill | path (m) | ep_len | backend | envs | grad steps | "
            "wall (s) | steps/s |\n")
    f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
      gs = f"{r['grad_steps']:,}" if r["grad_steps"] else "—"   # PPO: n/a
      f.write(f"| {r['cell']} | {r['arm']} | {r['algo']} | {r['seed']} | "
              f"{r['steps']:,} | {r['reach_rate']:.0%} | {r['safe_rate']:.0%} | "
              f"{r['coverage_reach']:.0%} | {r['reach_from_standstill']:.0%} | "
              f"{r['path_len']:.2f} | {r['ep_len']:.0f} | "
              f"{r['backend']}/{r['device']} | {r['n_envs']} | {gs} | "
              f"{r['wall_clock_s']:.0f} | {r['steps_per_s']:,} |\n")
  print(f"wrote {a.out_dir}/results.json and matrix.md")


if __name__ == "__main__":
  main()
