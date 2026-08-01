# Evaluate a policy

**Goal.** Measure whether a trained policy is actually safe — and whether its learned
certificate holds — with metrics that mean the same thing across runs.

**Prerequisites.** A trained model and an evaluation environment that delivers the
[environment contract](../concepts/environment-contract.md).

## The two rates

| metric | definition |
|---|---|
| **safe rate** | fraction of episodes that **never entered the failure set** (`g < 0` never occurred) |
| **reach-avoid success rate** | fraction of episodes that **reached the target (`l ≥ 0`) *and* stayed safe throughout** |

For an **avoid** task there is no target, so success rate is defined to equal the safe
rate. For a **reach-avoid** task, success is the stricter joint event.

Two accounting rules keep the numbers honest:

- **First-episode accounting.** With a vector env, sub-envs auto-reset at different times.
  Score each sub-env's **first** completed episode only, so a fast-failing env cannot be
  over-counted by looping many short episodes while a slow one runs once.
- **Deterministic actions.** Evaluate with the deterministic policy, not the exploration
  policy, so the number reflects the deployed behavior.

## `SafeSuccessRateEvalCallback`

The built-in callback logs `eval/safe_rate`, `eval/success_rate`, and `eval/ep_len_mean`
periodically during training. It runs on a **`TensorVecEnv`** (GPU-resident) eval env —
one that exposes `reset()` and `step_tensor(actions) -> (obs, g, dones, timeouts, l_x)` —
kept **separate** from the training env:

```python
from safety_sb3 import ReachAvoidSAC1P, SafeSuccessRateEvalCallback

eval_cb = SafeSuccessRateEvalCallback(
    eval_env,                 # a TensorVecEnv (optionally TensorVecNormalize-wrapped)
    n_rollouts=100,           # >= 100 first-episodes scored per eval
    eval_freq=1_000_000,      # env-steps between evals
    reach_avoid=True,         # success = reached AND safe (False -> success == safe_rate)
    deterministic=True,       # deployed behavior
)

model = ReachAvoidSAC1P("MlpPolicy", train_env, gamma=0.995)
model.learn(10_000_000, callback=eval_cb)
```

It runs `ceil(n_rollouts / num_envs)` parallel batches and scores the first episode of
each. A capped, never-violated episode (`max_ep_steps`) counts as safe.

The train scripts surface these as `--eval-rollouts`, `--eval-freq`, `--eval-envs`. See
the [hyperparameters](../hyperparameters.md#safe-rate-success-rate-evaluation) page for
defaults.

## Grading the certificate (precision / calibration)

The rates above grade the *policy*. To grade the *certificate* — does `V̂(s) ≥ 0` predict
that the policy actually reach-avoids from `s`? — measure **on-policy soundness**:

- **precision** = `P(policy reach-avoids | V̂ ≥ 0)` — of the states the certificate calls
  safe, how many the policy actually keeps safe.
- **calibration** — bin sampled states by `V̂` and plot empirical success per bin; a sound
  certificate rises sharply through `V̂ = 0`.

This is coverage-independent, unlike raw overlap with an oracle set. The full
methodology, with a Hamilton–Jacobi ground truth on Bicycle5D, is the
[oracle validation](../validation-oracle.md) study.

## Two-player evaluation

For a `*2P` model, `self.policy` is the control policy — evaluate it as above. To measure
**robustness**, evaluate the control policy against a *held-out* disturbance (a fixed
adversary checkpoint, a random disturbance, or the league's strongest opponent), not the
co-trained one it saw during training. The two-player league scores exactly these
pairings during training — see [Train an adversarial policy](train-adversarial.md#the-self-play-league-leaderboard).

## Video and reproducibility

- **Record eval videos on a fixed cadence from the start.** Behavioral failure modes
  (idling, suicide, rocking) are obvious in ten seconds of video and invisible in
  scalars.
- **Log against a fixed bank of held-out initial states**, evaluated deterministically —
  not just rolling training stats — so the number is comparable across runs.
- **Keep every variant as a named config in code**, not uncommitted edits.

More field rules: [best practices → reproducibility](../best-practices.md).

## Next steps

- [Oracle validation](../validation-oracle.md) — certificate soundness against ground truth.
- [Deploy a value-based safety filter](deploy-filter.md) — use the certificate online.
