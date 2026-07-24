"""Train ReachAvoidSafetySAC (reach-avoid V + fallback pi_safe) on ContactSafetyEnv(reach_avoid=True).

Same CLI/training shape as train_contact.py, but uses reach_avoid_sac.ReachAvoidSafetySAC (the
target-set-aware Bellman backup) and terminates episodes on reaching the target set too, not just
on failure -- see mujoco_env.py's reach_avoid flag and target_margin.py for the target set itself.

Produces a SEPARATE checkpoint from train_contact.py's avoid-only one -- that one is kept
(not overwritten) specifically so the two fallbacks can be compared (see reach_avoid_eval.py).

--resume-from lets you CONTINUE training an existing checkpoint for additional steps (num_timesteps
keeps counting up, so learning_starts's warmup doesn't re-trigger -- SB3 semantics), with
--ckpt-dir/--eval-every/--eval-n wiring in reach_avoid_callback.ReachAvoidCheckpointEval to
periodically snapshot the model and evaluate its mean realized reach-avoid score on a FIXED sample
of constraint-set states, building a learning-curve history.json for exactly this purpose: seeing
whether performance is still improving or has plateaued. NOTE: the replay buffer is NOT
saved/restored across a resume (only the policy/critic weights) -- a fresh buffer is filled from
scratch during the resumed run.

  python -m vault.train_reach_avoid --steps 3000      # smoke
  python -m vault.train_reach_avoid --steps 300000    # full
  python -m vault.train_reach_avoid --steps 300000 --resume-from vault/models/reach_avoid_safety_sac \\
      --ckpt-dir vault/models/reach_avoid_ckpts --eval-every 10000 --eval-n 25   # extend + track
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from safety_sb3 import SafetySAC
from stable_baselines3.common.env_checker import check_env

from . import config as C
from .mujoco_env import ContactSafetyEnv
from .reach_avoid_sac import ReachAvoidSafetySAC

HORIZON_STEPS = 300   # 3s at DT=0.01, matches the requested "come to a stop" evaluation horizon


def main():
    ap = argparse.ArgumentParser(description="Train ReachAvoidSafetySAC on the MuJoCo contact reach-avoid env")
    ap.add_argument("--steps", type=int, default=3000, help="additional steps to run this invocation")
    ap.add_argument("--learning-starts", type=int, default=500)
    ap.add_argument("--gamma", type=float, default=0.999, help="safety discount")
    ap.add_argument("--save", type=str, default=str(C.MODELS / "reach_avoid_safety_sac"))
    ap.add_argument("--resume-from", type=str, default=None,
                     help="existing checkpoint (.zip path, no extension) to continue training")
    ap.add_argument("--ckpt-dir", type=str, default=None,
                     help="if set, periodically save checkpoints + a learning-curve history.json here")
    ap.add_argument("--eval-every", type=int, default=10_000)
    ap.add_argument("--eval-n", type=int, default=25)
    ap.add_argument("--avoid-value-model", type=str, default=None,
                     help="path to a FROZEN trained SafetySAC (e.g. the avoid-only fallback) whose "
                          "own value replaces the raw margin as g(x) -- 'reach-and-always-avoid'")
    ap.add_argument("--saturate-target", action="store_true",
                     help="smooth-floor the target margin at -1 (see reach_avoid_sac.py)")
    ap.add_argument("--critic-warmup-steps", type=int, default=0,
                     help="freeze actor updates for this many gradient steps (use with --resume-from "
                          "to let the critic adapt to a warm-started actor before it starts moving)")
    ap.add_argument("--reset-timesteps", action="store_true",
                     help="force num_timesteps back to 0 even when --resume-from is set, so a "
                          "warm-started run's learning curve starts at the same x=0 origin as a "
                          "from-scratch run (for comparing training-step budgets on equal footing)")
    args = ap.parse_args()

    env = ContactSafetyEnv(seed=0, reach_avoid=True, max_steps=HORIZON_STEPS)
    check_env(env, warn=True)
    print("env API OK")
    gs, ls = [], []
    env.reset(seed=0)
    for _ in range(HORIZON_STEPS):
        _, r, term, trunc, info = env.step(env.action_space.sample())
        gs.append(r)
        ls.append(info["l"])
        if term or trunc:
            env.reset()
    print(f"random-policy margins g: min {min(gs):+.2f} max {max(gs):+.2f}  (g<0 = breach)")
    print(f"random-policy margins l: min {min(ls):+.2f} max {max(ls):+.2f}  (l>=0 = at target)\n")

    avoid_value_model = None
    if args.avoid_value_model:
        print(f"loading frozen avoid-value oracle from {args.avoid_value_model}.zip")
        avoid_value_model = SafetySAC.load(args.avoid_value_model, device="cpu")

    if args.resume_from:
        print(f"resuming from {args.resume_from}.zip (critic_warmup_steps={args.critic_warmup_steps})")
        model = ReachAvoidSafetySAC.load(args.resume_from, env=env, device="cpu",
                                         avoid_value_model=avoid_value_model,
                                         saturate_target=args.saturate_target,
                                         critic_warmup_steps=args.critic_warmup_steps)
    else:
        print(f"ReachAvoidSafetySAC train: {args.steps} steps, gamma={args.gamma}")
        model = ReachAvoidSafetySAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=100_000,
                                    learning_starts=args.learning_starts, batch_size=256, tau=0.01,
                                    gamma=args.gamma, train_freq=(1, "step"), gradient_steps=1,
                                    ent_coef="auto", seed=0, device="cpu", verbose=1,
                                    avoid_value_model=avoid_value_model,
                                    saturate_target=args.saturate_target,
                                    critic_warmup_steps=args.critic_warmup_steps)

    callback = None
    if args.ckpt_dir:
        from .reach_avoid_callback import ReachAvoidCheckpointEval
        callback = ReachAvoidCheckpointEval(args.ckpt_dir, eval_every=args.eval_every,
                                            n_eval=args.eval_n)

    t0 = time.time()
    reset_ts = args.reset_timesteps or not args.resume_from
    model.learn(args.steps, progress_bar=False, reset_num_timesteps=reset_ts,
               callback=callback)
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
