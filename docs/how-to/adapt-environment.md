# Adapt a Gymnasium environment

**Goal.** Turn your own environment into one a `safety_sb3` learner can train on, by
delivering the [environment contract](../concepts/environment-contract.md): the safety
margin `g` on the reward channel, the target margin `l` on `info["l_x"]` (reach-avoid),
and termination on breach.

**Prerequisites.** A working [install](../getting-started/installation.md) and an env you
can compute a signed safety margin for.

## 1. Safety-only (avoid) Gymnasium environment

Wrap the base env so the reward becomes `g(s)` and the episode ends when `g < 0`. Nothing
else about the env changes — observation and action spaces are untouched.

```python
import gymnasium as gym
import numpy as np


class MySafety(gym.Wrapper):
    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)
        g = self.safety_margin(obs)        # >= 0 outside the failure set
        terminated = terminated or g < 0.0 # terminate on breach
        return obs, float(g), terminated, truncated, info

    def safety_margin(self, obs) -> float:
        # e.g. signed distance to the failure boundary (positive = safe)
        ...
```

Train it with a `Safety*` learner:

```python
from safety_sb3 import SafetySAC1P
model = SafetySAC1P("MlpPolicy", MySafety(gym.make("YourEnv-v0")), gamma=0.995)
model.learn(200_000)
```

## 2. Reach-avoid Gymnasium environment

Add the **target margin** `l(s)` on `info["l_x"]` (`l ≥ 0` inside the target). Only the
`ReachAvoid*` learners read it; everything else is identical.

```python
class MyReachAvoid(gym.Wrapper):
    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)
        g = self.safety_margin(obs)        # >= 0 outside the failure set
        info["l_x"] = self.target_margin(obs)   # >= 0 inside the target set
        terminated = terminated or g < 0.0
        return obs, float(g), terminated, truncated, info
```

```python
from safety_sb3 import ReachAvoidSAC1P
model = ReachAvoidSAC1P("MlpPolicy", MyReachAvoid(gym.make("YourEnv-v0")),
                        terminal_type="all", gamma=0.995)
model.learn(200_000)
```

!!! danger "Do not fake an avoid task with a constant `l`"
    If your `l` exists only to be ignored, you have an avoid task — use a `Safety*`
    learner. A constant `l` does not reduce the reach-avoid backup to avoid; it produces
    an empty safe set or a certificate with no lookahead. See
    [Backups](../concepts/backups.md#avoid-is-not-a-reach-avoid-instance).

## 3. Vectorized environment

Any standard SB3 `VecEnv` works — the margin goes on each sub-env's reward, and `l_x` in
each sub-env's `info` dict. The shipped [Bicycle5D](../environments/bicycle5d.md)
(`BicycleGoalVec`) is a worked NumPy `VecEnv` example (batched, ~50k steps/s on CPU):

```python
from safety_sb3.testing.bicycle5d_vec import BicycleGoalVec
env = BicycleGoalVec(16, spawn="wide")     # 16 parallel envs, reward = g, info["l_x"] = l
```

## 4. Two-player (adversarial) action space

The `*2P` learners take a **single concatenated action** `Box(ctrl_dim + dstb_dim)`,
split at `ctrl_action_dim`. The env applies the first `ctrl_action_dim` components as the
control and the rest as the disturbance. See
[Train an adversarial policy](train-adversarial.md) for the full flow; the constructor
side is:

```python
from safety_sb3 import SafetyPPO2P
model = SafetyPPO2P("MlpPolicy", adversary_env, ctrl_action_dim=2)   # avoid: no l
```

## 5. TensorVecEnv (GPU-resident) { #tensor-path }

For thousands of GPU-resident envs, implement `step_tensor` returning device tensors —
no NumPy on the hot path. The learner auto-detects it and uses the tensor buffers with
identical backup math.

```python
from safety_sb3 import TensorVecEnv

class MyTensorEnv(TensorVecEnv):
    def step_tensor(self, actions):          # all torch, on self.device
        ...
        return obs, reward_g, dones, timeouts, l_x   # l_x here, not in an info dict
```

Pair it with `TensorVecNormalize` for on-device **observation** normalization (never the
reward). `tests/test_tensor_sac.py` is a complete CPU-runnable minimal example. See
[the tensor path](../concepts/environment-contract.md#the-tensor-path).

## Validate the contract

A learner will train on almost anything; a *wrong* margin fails silently. Before a long
run, assert the contract on a few steps:

```python
import numpy as np

env = MyReachAvoid(gym.make("YourEnv-v0"))
obs, info = env.reset(seed=0)
saw_breach_terminates = False
for _ in range(2000):
    obs, g, terminated, truncated, info = env.step(env.action_space.sample())
    assert np.isscalar(g) or np.ndim(g) == 0, "reward must be the scalar margin g"
    assert "l_x" in info, "reach-avoid env must put the target margin on info['l_x']"
    if g < 0.0:
        assert terminated, "must terminate when g < 0"
        saw_breach_terminates = True
    if terminated or truncated:
        obs, info = env.reset()
print("contract holds; observed a terminating breach:", saw_breach_terminates)
```

## Common failures

- **Reward looks scaled / drifting.** You wrapped the env in `VecNormalize(norm_reward=True)`
  or another reward scaler. Remove it — the reward *is* the margin. Observation
  normalization is fine.
- **Value never goes positive.** The margin may be violated by your reset distribution
  itself (that region is condemned by construction), or you terminate too late so
  post-breach states poison the target.
- **`ReachAvoidSAC1P` errors / ignores the target.** `info["l_x"]` is missing, or you are
  on the tensor path and forgot to return `l_x` as `step_tensor`'s 5th value.

## Next steps

- [The environment contract](../concepts/environment-contract.md) — the canonical spec.
- [Evaluate a policy](evaluate.md) — check the certificate holds.
- [Bicycle5D](../environments/bicycle5d.md) — a complete reference env to copy from.
