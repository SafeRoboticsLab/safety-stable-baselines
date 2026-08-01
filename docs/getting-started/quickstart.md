# Quickstart: train a safety policy

**Goal.** Train a policy on `Pendulum-v1` that keeps the pole within ±30° of upright,
and read the learned safety certificate — in a few minutes on CPU.

**Prerequisites.** A working [install](installation.md) (`python -c "import safety_sb3"`
succeeds). Gymnasium ships `Pendulum-v1`, so no extra environments are needed.

## The idea in one line

A safety learner does not maximize a reward. It maximizes a **safety margin** `g(s)` —
a number that is **positive in allowed states and negative after a violation**. You
feed `g` on the reward channel and terminate the episode when it goes negative. What you
get back is a value function `V` whose zero level set `{V ≥ 0}` is a **certificate**:
the states from which the policy can stay safe forever. See
[Safety margins and certificates](../concepts/safety-margins.md).

## Complete working example

```python
import gymnasium as gym
import numpy as np
from safety_sb3 import SafetySAC1P


class PendulumSafety(gym.Wrapper):
    """Reward == safety margin g(s) = pi/6 - |theta|  (safe iff |theta| <= 30 deg).

    theta is the pole angle (0 = upright). Pendulum's obs is [cos t, sin t, tdot].
    """

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)
        theta = np.arctan2(obs[1], obs[0])          # recover the angle
        g = np.pi / 6 - abs(theta)                  # >= 0 iff within +/- 30 deg
        terminated = terminated or g < 0.0          # terminate on a safety breach
        return obs, float(g), terminated, truncated, info


env = PendulumSafety(gym.make("Pendulum-v1"))

model = SafetySAC1P(
    "MlpPolicy", env,
    gamma=0.995,          # safety discount
    learning_starts=5_000,
    device="cpu",
    verbose=1,
)
model.learn(100_000)      # a few minutes on CPU
model.save("pendulum_safety")
```

## Expected output

SB3's standard training table prints as it learns. Because the reward *is* the margin,
`rollout/ep_rew_mean` is the **mean per-step safety margin** — it should climb toward a
small positive number as the policy learns to hold the pole upright, and
`rollout/ep_len_mean` should rise toward the episode cap (200) as breaches become rare:

```
| rollout/            |          |
|    ep_len_mean      | 200      |
|    ep_rew_mean      | 0.34     |   <- mean margin per step, now positive
| train/              |          |
|    gamma            | 0.999    |   <- discount is annealed up automatically
```

Read the certificate for a state — `V(s) = min_i Q_i(s, pi(s))`:

```python
import torch
obs, _ = env.reset(seed=0)
o = torch.as_tensor(obs, dtype=torch.float32, device=model.device).unsqueeze(0)
with torch.no_grad():
    a = model.policy.actor(o)
    v = torch.cat(model.policy.critic(o, a), dim=1).min(dim=1).values
print("V(s) =", float(v), "-> safe-controllable" if v >= 0 else "-> not certified")
```

## Explanation

- **`SafetySAC1P`** = **Safety** mode (avoid), **SAC** algorithm, **1** player. The
  name is the whole spec — see [the MAP convention](../map.md).
- **`gamma=0.995`** is the safety discount. It is **annealed toward 1** automatically
  (`train/gamma` in the log); the safe/unsafe boundary only sharpens as `γ → 1`. See
  [Backups](../concepts/backups.md) and the
  [hyperparameters](../hyperparameters.md#discount-annealing-gamma) page.
- **Terminating on `g < 0`** is mandatory: letting the sim run past a violation leaks
  post-failure states into the value target and moves the certificate boundary.

## Reach-avoid { #reach-avoid }

To *reach a target while staying safe*, additionally return a **target margin** `l(s)`
on `info["l_x"]` (`l ≥ 0` inside the target) and switch to a `ReachAvoid*` learner — the
class name is the only structural change:

```python
class PendulumReachAvoid(gym.Wrapper):
    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)
        theta = np.arctan2(obs[1], obs[0])
        g = np.pi / 6 - abs(theta)                  # safety margin (as before)
        info["l_x"] = 0.05 - abs(theta)             # target: within +/- ~3 deg of upright
        return obs, float(g), terminated or g < 0.0, truncated, info
```

```python
from safety_sb3 import ReachAvoidSAC1P
model = ReachAvoidSAC1P("MlpPolicy", PendulumReachAvoid(gym.make("Pendulum-v1")),
                        gamma=0.995, terminal_type="all", device="cpu")
model.learn(100_000)
```

A complete PPO reach-avoid script ships as
[`examples/pendulum_reach_avoid_ppo_train.py`](https://github.com/SafeRoboticsLab/safety-stable-baselines/blob/main/examples/pendulum_reach_avoid_ppo_train.py).

!!! warning "Do not degenerate `l` to fake an avoid task"
    Running an avoid task on a reach-avoid learner by pinning `l` to a constant does
    **not** reduce to the avoid backup — it silently produces either an empty safe set
    or a certificate with no lookahead. Use a `Safety*` learner for avoid. The
    [Backups](../concepts/backups.md#avoid-is-not-a-reach-avoid-instance) page proves why.

## Common failures

- **`ep_rew_mean` looks huge / tiny and never stabilizes.** You are normalizing the
  reward. `VecNormalize(norm_reward=True)` (or any reward scaler) destroys the margin —
  the reward *is* `g`. Observation normalization is fine.
- **The value never goes positive.** The task may genuinely be infeasible from your
  reset states, or you terminated too late (states past a breach poison the target).
- **`ReachAvoidSAC1P` needs `info["l_x"]`.** Without a target margin it has nothing to
  read; use a `Safety*` learner instead.

## Next steps

- [Choose a learner](choose-a-learner.md) — SAC vs PPO, 1P vs 2P, avoid vs reach-avoid.
- [Adapt a Gymnasium environment](../how-to/adapt-environment.md) — wrap your own env.
- [Evaluate a policy](../how-to/evaluate.md) — measure safe-rate / success-rate.
