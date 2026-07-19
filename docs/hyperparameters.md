# Tunable hyperparameters

Every safety learner (`SafetyPPO`/`SafetySAC` and their reach-avoid / two-player
subclasses `ReachAvoidPPO/SAC`, `GameplayPPO/SAC`, `IsaacsPPO/SAC`) takes the
knobs below as constructor arguments. Defaults are chosen to match the reference
ISAACS codebase (`safe_adaptation_dev`) out of the box. Anything not listed here
is a stock Stable-Baselines3 argument.

## Discount annealing (gamma)

The reach-avoid / avoid Bellman operator only produces a **sharp** safe/unsafe
boundary as `gamma -> 1`, but training directly at `gamma ≈ 1` is
ill-conditioned. So gamma is **annealed** from a contractive start up toward 1.
**On by default** in every safety learner.

| arg | default | meaning |
|---|---|---|
| `gamma` | `0.99` | starting discount (the anneal's `init`) |
| `gamma_anneal` | `True` | `True` = default schedule; `False` = constant `gamma`; or pass a schedule object / callable `frac -> gamma` |

`gamma_anneal=True` installs the **reference-faithful discrete-jump** schedule
`StepGammaAnneal` (analog of `safe_adaptation_dev`'s `StepLRMargin`):

- `StepGammaAnneal(init=0.99, end=0.9999, ratio=0.1, period_frac=0.20)` —
  the gap `(1 - gamma)` is multiplied by `ratio` every `period_frac` of the
  training horizon: **0.99 → 0.999 (at 20%) → 0.9999 (at 40%), then hold**.
  Tune `init`/`end` for the range, `period_frac` for how often it jumps
  (the reference uses one jump every 10–20% of the horizon), `ratio` for how big
  each jump is (`0.1` = one extra nine per jump).
- On **each discrete jump** the SAC learners **reset the entropy temperature**
  (both actors) — the Q-scale shifts at a jump, so the tuned alpha is stale
  (reference behavior; logged as `train/alpha_reset_gamma`).

For a **smooth** anneal instead, pass
`gamma_anneal=GeometricGammaAnneal(init=0.99, end=0.9999, anneal_frac=0.5)`
(continuous log-space interpolation reaching `end` at `anneal_frac`, then hold;
no discrete jump, so no alpha reset).

`train/gamma` is logged every update so you can watch the schedule in wandb.

## Entropy temperature (alpha) floor / ceiling

SB3's auto-tuned entropy coefficient is unbounded and can collapse toward 0
(deterministic, no exploration), especially right after a gamma-jump alpha reset.
A floor prevents this (reference `min_alpha`).

| arg | default | meaning |
|---|---|---|
| `min_alpha` | `1e-3` | floor on the learned entropy temperature (`None` = no floor) |
| `max_alpha` | `None` | optional ceiling (`None` = unbounded, standard SAC) |

Applies to `ent_coef="auto..."` only (a fixed `ent_coef` is untouched). For the
two-player learners the same bounds clamp **both** the ctrl and dstb alphas.

## Per-agent / per-network learning rates (two-player SAC)

In the two-player games (`GameplaySAC` / `IsaacsSAC`) each network can take its
own learning rate. Each defaults to `None` → falls back to the shared
`learning_rate`, so single-lr callers are unchanged.

| arg | default | meaning |
|---|---|---|
| `learning_rate` | `3e-4` | shared lr; also the **ctrl actor** lr |
| `critic_learning_rate` | `None`→shared | twin-critic optimizer lr |
| `dstb_learning_rate` | `None`→shared | disturbance (min-player) actor lr |
| `ent_coef_lr` | `None`→shared | ctrl entropy-temperature optimizer lr |
| `dstb_ent_coef_lr` | `None`→shared | dstb entropy-temperature optimizer lr |

An **optional StepLR-style decay** (off by default) can decay the ctrl/dstb/critic
lrs over training, mirroring the reference:

| arg | default | meaning |
|---|---|---|
| `lr_schedule` | `False` | enable StepLR decay of the network lrs |
| `lr_period` | `1_000_000` | env-steps between decay steps |
| `lr_decay` | `0.1` | multiplicative factor per decay step |
| `lr_end` | `0.0` | lr floor |

(The PPO two-player learners already expose `dstb_learning_rate` / `dstb_ent_coef`
and a per-player KL-adaptive lr via `adaptive_lr` / `desired_kl`.)

## Leaderboard (two-player opponent sampling)

The league of past checkpoints; the disturbance opponent for each rollout is
sampled by a softmax over pairwise reach-avoid success scores.

| arg | default | meaning |
|---|---|---|
| `use_leaderboard` | `False` (constructor) | enable the league. **On by default in the 2-agent training path** (`train.py --adversary`, `train_gameplay_sac.py`) |
| `leaderboard_eval_env` | `None` | env used to score pairings (required for the league to actually evaluate) |
| `softmax_rationality` | `3.0` | softmax temperature β on the `[0,1]` success scores; higher β concentrates sampling on the strongest opponents (β=3 already puts ~56% of the mass on the top quartile; use ~5 for stronger dominance) |
| `leaderboard_freq` | `10_000` | env-steps between league evaluations |
| `n_eval_episodes` | `10` | eval batches per pairing (effective trajectories = `num_envs * n_eval_episodes` on a vec eval env) |
| `save_top_k_ctrl` / `save_top_k_dstb` | `5` / `5` | league size per player |

> **⚡ Throughput — the league eval can dominate wall-clock.** Each `_leaderboard_step`
> runs `~(nc+nd+2)` pairings, and each pairing steps the sim `n_eval_episodes × episode_len`
> times. Profiling a two-player GameplaySAC found ONE `_leaderboard_step` ≈ **100 s** vs a
> ~90 ms train cycle — the league was **~97 % of wall-clock** at 1024 envs with the old
> `leaderboard_freq=10_000` / `n_eval_episodes=10`, capping throughput at ~500 FPS. The cost is
> the **volume of sim steps**, not the obs transport. Two levers, both safety-neutral (the league
> is a *relative* ranking):
>
> 1. **Raise `leaderboard_freq`** — 10k→2M fires ~200× less often. 2. **Lower `n_eval_episodes`**
>    — 10→3 cuts each firing ~3×. 3. **Pass a `TensorVecEnv` eval env** — dispatches to the
>    on-device `_eval_pair_tensor` (no numpy VecEnv, no per-step host↔device sync; obs normalized
>    via the live training normalizer).
>
> Together these took a 1024-env GameplaySAC from **~500 → ~19,000 FPS (~30×)** — a 100 M-step run
> from ~55 h to ~1.5 h. The zoo `examples/train_sac.py` uses these throughput defaults
> (`--leaderboard-freq 2_000_000 --leaderboard-episodes 3`, raw tensor eval env).

## Safe-rate / success-rate evaluation

`SafeSuccessRateEvalCallback` logs `eval/safe_rate`, `eval/success_rate`,
`eval/ep_len_mean` to wandb periodically.

| arg | default | meaning |
|---|---|---|
| `n_rollouts` | `100` | episodes per eval (reference ~100); runs `ceil(n_rollouts/num_envs)` parallel batches |
| `eval_freq` | `1_000_000` | env-steps between evals |
| `reach_avoid` | `True` | reach-avoid: success = reached target AND safe; avoid-only: success == safe_rate |

Definitions: **safe** = never entered the failure set (`g < 0`); **reached** =
ever hit the target (`l_x >= 0`). The train scripts expose `--eval-rollouts`,
`--eval-freq`, `--eval-envs`.

## Common SAC knobs (reference values for go2)

`learning_rate=1e-4`, `tau=0.01`, `target_update_interval=2`,
`ent_coef="auto_0.1"` (auto-tune alpha from 0.1), `buffer_size`, `batch_size`,
`train_freq=1`, `gradient_steps` (a small int on the tensor path — **never `-1`**,
which means `num_envs` updates per vector step), `learning_starts`.
