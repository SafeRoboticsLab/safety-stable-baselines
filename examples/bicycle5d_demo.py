"""Avoid vs reach-avoid on the 5-D bicycle -- the contrast, rendered.

    python examples/bicycle5d_demo.py                 # SafetyPPO vs ReachAvoidPPO
    python examples/bicycle5d_demo.py --adversary     # IsaacsPPO vs GameplayPPO
    python examples/bicycle5d_demo.py --no-wandb --render out.png

What you should see:

  avoid       -- the car does NOT go anywhere. Nothing rewards motion; g > 0 is
                 already satisfied where it starts, so braking to a stop and
                 sitting there is optimal. Correct, and the negative control.
  reach-avoid -- the car drives to the goal, steering around the obstacles.

`reach_rate(RA) >> reach_rate(avoid)` is the whole assertion, and it is what a
wrong reach-avoid anchor fails: a g-anchored backup values sitting still at
V = g > 0, which beats driving, so the RA car sits still too and the two GIFs
become identical. See RELEASE_NOTES v0.2.0 and safety_sb3.backups.

wandb: project `safety_sb3`, group `bicycle5d[-adv]`, one run per arm, with a
GIF of a rollout logged every --video-every steps.
"""
from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from safety_sb3 import (GameplayPPO, GameplaySAC, IsaacsPPO, IsaacsSAC,
                        ReachAvoidPPO, ReachAvoidSAC, SafetyPPO, SafetySAC,
                        StdCapCallback)
from safety_sb3.testing import BicycleGoal
from safety_sb3.testing.bicycle5d_vec import BicycleGoalVec


def _act(model, obs, adversary):
  a, _ = model.predict(obs, deterministic=True)
  a = np.asarray(a).reshape(-1)
  # the learner is the CTRL player; freeze the disturbance at zero for eval
  return np.concatenate([a[:2], np.zeros(5)]) if adversary else a


def rollout(model, adversary, seed, from_standstill=False, frames=False,
            fig=None, ax=None, title=None):
  env = BicycleGoal(adversary=adversary)
  obs, _ = env.reset(seed=seed)
  if from_standstill:
    env.s[2] = 0.0                      # probe the initiation question directly
    obs = env._obs(env.s)
  trail, imgs = [env.s[:2].copy()], []
  reached = collided = False
  for t in range(env.timeout):
    obs, g, term, trunc, info = env.step(_act(model, obs, adversary))
    trail.append(env.s[:2].copy())
    if frames and t % 3 == 0:           # ~20 fps of sim at dt=0.05
      imgs.append(env.render_rgb(np.asarray(trail), fig=fig, ax=ax, title=title))
    if term or trunc:
      reached, collided = info["reached"], info["collided"]
      break
  dist = float(np.hypot(env.s[0] - env.goal[0], env.s[1] - env.goal[1]))
  path = float(np.abs(np.diff(np.asarray(trail), axis=0)).sum())
  return dict(reached=reached, collided=collided, dist=dist, path=path,
              trail=np.asarray(trail), env=env, frames=imgs)


def evaluate(model, adversary, n=32, seed=1234, from_standstill=False):
  rng = np.random.default_rng(seed)
  R = [rollout(model, adversary, int(rng.integers(1 << 30)), from_standstill)
       for _ in range(n)]
  return dict(reach=float(np.mean([r["reached"] for r in R])),
              collide=float(np.mean([r["collided"] for r in R])),
              dist=float(np.mean([r["dist"] for r in R])),
              path=float(np.mean([r["path"] for r in R])))


class WandbProbe(BaseCallback):
  """Log the discriminating metrics + a GIF. `reach` IS the claim."""

  def __init__(self, adversary, tag, every=25_000, video_every=50_000):
    super().__init__()
    self.adversary, self.tag = adversary, tag
    self.every, self.video_every = every, video_every
    self._next, self._next_vid = 0, video_every

  def _on_step(self) -> bool:
    import wandb
    if self.num_timesteps < self._next:
      return True
    self._next = self.num_timesteps + self.every
    m = evaluate(self.model, self.adversary, n=12)
    ss = evaluate(self.model, self.adversary, n=12, from_standstill=True)
    log = {f"probe/{k}": v for k, v in m.items()}
    log["probe/reach_from_standstill"] = ss["reach"]
    log["probe/path_len"] = m["path"]     # ~0 => the car never went anywhere
    # HONEST coverage: reach rate over the 8 coverage spawns x 4 eval maps (32
    # cars all over the map) -- this is what the coverage GIFs show, vs the easy
    # near-origin reach above.
    try:
      from safety_sb3.testing.bicycle5d_render import EVAL_MAPS, multi_car_rollout
      reached = total = 0
      for mp_ in EVAL_MAPS:
        _, status = multi_car_rollout(self.model, mp_, adversary=self.adversary)
        reached += sum(s == "reached" for s in status); total += len(status)
      log["probe/coverage_reach"] = reached / total
    except Exception:
      pass
    if self.num_timesteps >= self._next_vid:
      self._next_vid = self.num_timesteps + self.video_every
      try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from safety_sb3.testing.bicycle5d_render import (
          EVAL_MAPS, map_gif, value_map)
        fig, ax = plt.subplots(figsize=(6, 3.2), dpi=64)
        title = f"{self.tag} @ {self.num_timesteps/1e6:.1f}M"
        for m in EVAL_MAPS:                      # per map: an 8-car GIF + a V(x,y) map
          frames = map_gif(self.model, m, adversary=self.adversary,
                           title=title, fig=fig, ax=ax)
          if frames:
            vid = np.stack(frames).transpose(0, 3, 1, 2)  # (T,C,H,W)
            log[f"video/{m['name']}"] = wandb.Video(vid, fps=20, format="gif")
          vfig = value_map(self.model, m, title=title)
          log[f"value/{m['name']}"] = wandb.Image(vfig)
          plt.close(vfig)
        plt.close(fig)
      except Exception as e:  # GIFs are a nicety; never let them kill training
        if not getattr(self, "_warned_video", False):
          print(f"[warn] GIF logging disabled: {e}")
          self._warned_video = True
    wandb.log(log, step=self.num_timesteps)
    return True


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--steps", type=int, default=300_000)
  p.add_argument("--adversary", action="store_true")
  p.add_argument("--arms", choices=["both", "ra", "avoid"], default="both",
                 help="which arms to run (reach-avoid first)")
  p.add_argument("--family", choices=["ppo", "sac"], default="ppo",
                 help="ppo (fast, muddy value) or sac (calibrated value landscape)")
  p.add_argument("--sac-envs", type=int, default=16)
  p.add_argument("--render", default=None, help="also write a static PNG here")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--n-envs", type=int, default=256)
  p.add_argument("--n-steps", type=int, default=64)
  p.add_argument("--batch-size", type=int, default=4096)
  p.add_argument("--no-wandb", action="store_true")
  p.add_argument("--video-every", type=int, default=50_000)
  p.add_argument("--probe-every", type=int, default=25_000)
  p.add_argument("--save-dir", default=None, help="save final models here")
  p.add_argument("--spawn", choices=["edge", "wide", "map"], default="wide",
                 help="train spawns: near-origin (edge) or whole-map (map)")
  p.add_argument("--lr", type=float, default=5e-4)
  p.add_argument("--ent-coef", type=float, default=1e-3)
  p.add_argument("--adaptive-lr", action="store_true", default=True)
  p.add_argument("--no-adaptive-lr", dest="adaptive_lr", action="store_false")
  p.add_argument("--max-std", type=float, default=0.0,
                 help="cap action std (0 disables; StdCap destabilizes adaptive_lr)")
  a = p.parse_args()

  sac = a.family == "sac"
  if sac:
    avoid_cls = IsaacsSAC if a.adversary else SafetySAC
    ra_cls = GameplaySAC if a.adversary else ReachAvoidSAC
  else:
    avoid_cls = IsaacsPPO if a.adversary else SafetyPPO
    ra_cls = GameplayPPO if a.adversary else ReachAvoidPPO
  group = ("bicycle5d-sac" if sac else "bicycle5d") + ("-adv" if a.adversary else "")
  # reach-avoid FIRST (the interesting arm); avoid is the control. --arms lets
  # you run just one while iterating.
  all_arms = [("reach-avoid", ra_cls), ("avoid", avoid_cls)]
  arms = {"both": all_arms, "ra": all_arms[:1], "avoid": all_arms[1:]}[a.arms]
  print(f"=== bicycle5d | {'two-player' if a.adversary else 'single-player'} "
        f"| {a.steps:,} steps/arm | group={group} | arms={a.arms} ===\n")

  results, panels = {}, {}
  for tag, cls in arms:
    label = f"{tag} ({cls.__name__})"
    cb = None
    if not a.no_wandb:
      import wandb
      run = wandb.init(project="safety_sb3", entity="buzinguyen", group=group,
                       name=f"{group}_{tag}", job_type="bicycle5d", reinit=True,
                       config=dict(arm=tag, algo=cls.__name__, steps=a.steps,
                                   seed=a.seed, n_envs=a.n_envs,
                                   adversary=a.adversary))
      cb = WandbProbe(a.adversary, tag, every=a.probe_every,
                      video_every=a.video_every)
    kw = dict(ctrl_action_dim=2) if a.adversary else {}
    # Vectorized batched env: steps all n_envs bicycles as numpy arrays.
    n_envs = a.sac_envs if sac else a.n_envs
    venv = BicycleGoalVec(n_envs, adversary=a.adversary, seed=a.seed,
                          spawn=a.spawn)
    if sac:
      # off-policy Q: gives the calibrated value landscape (PPO's on-policy V
      # is muddy off its tube). Fewer envs, replay buffer, UTD ~= 1.
      model = cls("MlpPolicy", venv, seed=a.seed, buffer_size=500_000,
                  learning_starts=5000, batch_size=512, train_freq=(16, "step"),
                  gradient_steps=16, gamma=0.99, verbose=0, device="cpu", **kw)
      cbs = [cb] if cb else []
    else:
      # KL-adaptive LR paces the updates so learning is gradual, not front-loaded.
      model = cls("MlpPolicy", venv, seed=a.seed, n_steps=a.n_steps,
                  batch_size=a.batch_size, gamma=0.99, ent_coef=a.ent_coef,
                  learning_rate=a.lr, adaptive_lr=a.adaptive_lr, desired_kl=0.01,
                  verbose=0, device="cpu", **kw)
      cbs = [cb] if cb else []
      if a.max_std:                            # keep exploration alive
        cbs.append(StdCapCallback(max_std=a.max_std))
    model.learn(total_timesteps=a.steps,
                callback=cbs if len(cbs) != 1 else cbs[0])
    if a.save_dir:
      import os
      os.makedirs(a.save_dir, exist_ok=True)
      model.save(os.path.join(a.save_dir, f"{group}_{tag}"))

    r = evaluate(model, a.adversary)
    ss = evaluate(model, a.adversary, from_standstill=True)
    r["reach_standstill"] = ss["reach"]
    results[label] = r
    if not a.no_wandb:
      wandb.summary.update({f"final/{k}": v for k, v in r.items()})
      run.finish()
    if a.render:
      panels[label] = rollout(model, a.adversary, 7, from_standstill=True)
    print(f"[{label:28s}] reach={r['reach']:5.0%} (from standstill "
          f"{r['reach_standstill']:4.0%})  collide={r['collide']:4.0%}  "
          f"path={r['path']:5.2f}m  dist_to_goal={r['dist']:.2f}m")

  ra_key = f"reach-avoid ({ra_cls.__name__})"
  av_key = f"avoid ({avoid_cls.__name__})"
  if ra_key in results and av_key in results:      # both arms ran -> the contrast
    ra, av = results[ra_key]["reach"], results[av_key]["reach"]
    print(f"\n=== reach-avoid {ra:.0%} vs avoid {av:.0%} ===")
    if ra > av + 0.4:
      print("  => the reach term does what it should. Anchor healthy.")
    else:
      print("  => NO CONTRAST. Either the RA anchor is wrong (a g-anchored backup "
            "makes sitting still worth V=g>0, beating driving), or l is not "
            "informative across the map, or training is short.")

  if a.render:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 3 * len(panels)))
    for ax, (label, r) in zip(np.atleast_1d(axes), panels.items()):
      r["env"].render_frame(ax=ax, trail=r["trail"])
      ax.set_title(f"{label} — reach {results[label]['reach']:.0%}")
    fig.tight_layout()
    fig.savefig(a.render, dpi=110)
    print(f"\nwrote {a.render}")


if __name__ == "__main__":
  main()
