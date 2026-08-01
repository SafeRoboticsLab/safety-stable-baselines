# The environment contract

This is the canonical contract for what a `safety_sb3` learner expects from an
environment — the `g`/`l` channel definition every learner is written against. Get these
right and everything works; get `g` wrong and the value function's zero level set — the
safety boundary — moves.

For the *meaning* of the margins see [Safety margins and certificates](safety-margins.md);
for a complete copy-paste wrapper see [Adapt a Gymnasium environment](../how-to/adapt-environment.md).

## The channels

Every learner consumes an environment through two channels plus a done signal:

| channel | symbol | meaning | where |
|---|---|---|---|
| reward | `g(s)` | **safety margin**. `g ≥ 0` ⟺ outside the failure set. | the reward field |
| info | `l(s)` | **target margin**. `l ≥ 0` ⟺ inside the target set. Reach-avoid only. | `info["l_x"]` (NumPy) / `step_tensor`'s 5th return (tensor) |
| dones | | terminate when `g < 0`; timeouts are never value-bootstrapped by default | standard Gym |

## The rules

Each of these has caused a real debugging headache (see [best practices](../best-practices.md)):

- **Never normalize the reward.** `g` *is* the margin. `VecNormalize(norm_reward=True)`
  or any reward scaler silently destroys the backup. Observation normalization is fine.
- **Terminate on `g < 0`.** Letting the sim run past a violation leaks post-failure
  states into the value target.
- **`l` is used only by the `ReachAvoid*` learners.** The `Safety*` learners do not use
  it — they have nowhere to put it. Their buffers carry no `l` column at all, which is a
  structural guarantee rather than a convention. An avoid task should not invent one —
  see [Backups → avoid is not reach-avoid](backups.md#avoid-is-not-a-reach-avoid-instance).

## Timeouts

`bootstrap_on_timeout=False` (the default) treats a timeout as terminal for the backup —
correct for the `g`/`l` margin semantics, which are **absolute margins, not returns to
bootstrap**. Under `terminal_type="all"` a timeout is valued at `min(l, g)`, which is the
horizon cutoff by construction. See [Backups](backups.md).

## The NumPy path vs the tensor path { #the-tensor-path }

There are two ways to deliver the contract, and every learner auto-detects which one it
is handed (`is_tensor_env`).

### NumPy path (standard SB3 `VecEnv`)

`g` on `reward`, `l` on `info["l_x"]`. Used by the SAC family and by PPO on CPU envs.
This is the ordinary Gymnasium / SB3 interface — wrap any env so the reward is the margin
and breaches terminate.

### Tensor path (`TensorVecEnv`, GPU-resident)

For simulators that live on the GPU (thousands of parallel envs), the NumPy `VecEnv`
round-trip dominates. Subclass `TensorVecEnv` and implement a single method that returns
device tensors:

```python
def step_tensor(self, actions):          # all torch, on env.device
    return obs, reward_g, dones, timeouts, l_x
```

Every algorithm detects it and switches to torch-native rollout collection with the
`Tensor*RolloutBuffer` matching its Mode (on-policy) or `TensorReplayBuffer`
(off-policy) — **identical backup math, no NumPy on the hot path**. `TensorVecNormalize`
provides on-device running observation normalization (observations only — never the
reward). This is the ~50k-step/s regime for massively-parallel envs; the PPO family is
the usual choice here.

`tests/test_tensor_sac.py` is a complete minimal example — a 64-env double integrator,
CPU-runnable. See also [Build a TensorVecEnv](../how-to/adapt-environment.md#tensor-path)
for a walkthrough, and the reference classes
[`TensorVecEnv` / `TensorVecNormalize`](../reference.md#environments-and-normalization).

## Composing with the environment layer

The [`robot-safety-sandbox`](https://github.com/SafeRoboticsLab/robot-safety-sandbox)
task layer builds these channels for you: a task declares its `mode` (via which margins
it produces) and whether it `supports_adversary`, and the trainer resolves the learner
class by formula. See [The MAP convention → environment layer](../map.md#how-the-environment-layer-composes-with-the-map).

## Related API

- [`TensorVecEnv`, `TensorVecNormalize`](../reference.md#environments-and-normalization)
- [Rollout / replay buffers](../reference.md#buffers)
- [Backups](backups.md) — the operators these channels feed.
