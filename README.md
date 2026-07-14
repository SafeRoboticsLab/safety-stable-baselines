# safety-stable-baselines

Lightweight add-on for [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
implementing **Hamilton–Jacobi safety RL** (Fisac et al., ICRA '19), **reach-avoid RL**
(Hsu et al., RSS '21) and **adversarial reach-avoid / ISAACS** (Hsu, Nguyen et al.,
L4DC '23) — plus a GPU-resident tensor path for massively parallel simulators
(mjlab / Isaac-style).

**Design principle: keep upstream SB3 untouched.** Everything lives in this separate
package and still feels native to SB3 users: same constructors, same `learn()`, same
callbacks and loggers.

## Algorithms

| class | base | backup (value target) | policies |
|---|---|---|---|
| `SafetySAC` | SAC | `min(g, V')` | 1 (avoid) |
| `SafetyDQN` | DQN | `min(g, V')` | 1 (avoid) |
| `SafetyPPO` | PPO | `min(g, V')` | 1 (avoid) |
| `SafetyA2C` | A2C | `min(g, V')` | 1 (avoid) |
| `ReachAvoidSAC` | SAC | `min(g, max(l, V'))` | 1 (reach-avoid) |
| `ReachAvoidPPO` | PPO | `min(g, max(l, V'))` | 1 (reach-avoid) |
| `IsaacsSAC` | ReachAvoidSAC | `min(g, max(l, V'))`, two-player | ctrl (max) + dstb (min) |
| `IsaacsPPO` | ReachAvoidPPO | `min(g, max(l, V'))`, two-player | ctrl (max) + dstb (min) |

All backups use the time-discounted convention
`target = (1 − γ·nt)·g + γ·nt·backup` with `nt = 1` on non-terminal steps, so a
terminating step returns exactly `g` (the terminal anchor is `g`, **not** `min(g, l)` —
anchoring on `min(g, l)` injects the large negative off-target `l` into every episode
end and stalls learning; see `safety_sb3/safety_buffers.py`).

### Margin conventions (the env contract)

- **`g(s)` — safety margin — rides on the reward channel.** `g ≥ 0` iff the state is
  outside the failure set. The env must `terminate` the episode when `g < 0`.
- **`l(s)` — target margin — rides on `info["l_x"]`** (numpy path) or is returned
  directly by `step_tensor` (tensor path). `l ≥ 0` iff the state is inside the target
  set. Only the ReachAvoid/Isaacs algorithms read it.
- **Never normalize rewards** — the reward *is* the margin; `VecNormalize(norm_reward=True)`
  corrupts the backup. Observation normalization is fine.
- A trained value function satisfies `V(s) ≥ 0` ⇔ (avoid) "the policy can stay safe
  forever from `s`" / (reach-avoid) "the policy can reach the target without ever
  failing" — this is what makes it usable as a runtime **safety filter**
  (switch to the safety policy when the nominal's next state has `V < 0`).

## The tensor path (GPU-resident training)

For simulators that live on the GPU (thousands of parallel envs), the numpy `VecEnv`
round-trip dominates. Subclass `safety_sb3.TensorVecEnv` and implement

```python
def step_tensor(self, actions):          # all torch, on env.device
    return obs, reward_g, dones, timeouts, l_x
```

and every algorithm above detects it (`is_tensor_env`) and switches to torch-native
rollout collection with `TensorSafetyRolloutBuffer` / `TensorReachAvoidRolloutBuffer`
(on-policy) or `TensorReplayBuffer` (off-policy) — identical backup math, no numpy on
the hot path. `TensorVecNormalize` provides on-device running observation
normalization. See `tests/test_tensor_sac.py` for a complete minimal example
(a 64-env double integrator, CPU-runnable).

## Installation

```bash
conda create --name safety_sb3 python=3.10
conda activate safety_sb3

git clone git@github.com:SafeRoboticsLab/safety-stable-baselines.git
cd safety-stable-baselines
pip install -e .
```

That is the whole core install (deps: `stable-baselines3`, `torch`, `gymnasium`,
`numpy`, `tensorboard`, `wandb`).

Optional extras for the bundled benchmark environments (only needed to run the
`examples/`): this repo includes
[`safety-gymnasium` (SafeRoboticsLab fork)](https://github.com/SafeRoboticsLab/safety-gymnasium)
and `rl_baselines3_zoo` as git submodules — python 3.10 is pinned for
`safety-gymnasium`'s sake:

```bash
git submodule update --init
pip install -e integrations/rl_baselines3_zoo
pip install -e integrations/safety-gymnasium
```

## Quick start

Wrap any gym env so the reward is the safety margin and breaches terminate:

```python
import gymnasium as gym
import numpy as np
from safety_sb3 import SafetySAC


class PendulumSafety(gym.Wrapper):
    """Reward == safety margin g(s) = pi/6 - |theta| (safe iff |theta| <= 30 deg)."""

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        theta = np.arctan2(obs[1], obs[0])
        g = np.pi / 6 - abs(theta)
        return obs, float(g), terminated or g < 0, truncated, info


model = SafetySAC("MlpPolicy", PendulumSafety(gym.make("Pendulum-v1")),
                  gamma=0.995, verbose=1)
model.learn(100_000)
```

For reach-avoid, additionally return the target margin in the info dict
(`info["l_x"] = l`) and train `ReachAvoidPPO` / `ReachAvoidSAC` the same way.
`examples/pendulum_reach_avoid_ppo_train.py` is the runnable version.

## Fine-tuning stability

`safety_sb3.StdCapCallback(max_std=...)` clamps the policy's action std at every
rollout start. Margin-only objectives carry no action-quality gradient, so PPO's std
can inflate organically and erode a converged motor skill during safety fine-tuning —
the cap is the one-line remedy. Pair it with SB3's native `target_kl`. The full set of
hard-won usage rules (margin scaling, warm starts, curricula) is in
[BEST_PRACTICES.md](BEST_PRACTICES.md).

## Tests

```bash
pip install pytest
python -m pytest tests/ -q
```

- `tests/test_backups.py` — unit tests of the safety / reach-avoid Bellman recursions
  (terminal anchoring, timeout bootstrap, target banking, fixed-point consistency).
- `tests/test_ppo_smoke.py` — `SafetyPPO`/`ReachAvoidPPO` end-to-end on a 1-D double
  integrator (seconds, CPU).
- `tests/test_tensor_sac.py` — tensor-path buffer semantics + `SafetySAC`/
  `ReachAvoidSAC` learning on the same task (~2 min, CPU).

## References

- J. Fisac et al., "[Bridging Hamilton-Jacobi Safety Analysis and Reinforcement
  Learning](https://ieeexplore.ieee.org/document/8794107)," ICRA 2019.
- K.-C. Hsu, V. Rubies-Royo, C. Tomlin, J. Fisac, "[Safety and Liveness Guarantees
  through Reach-Avoid Reinforcement Learning](https://arxiv.org/abs/2112.12288)," RSS 2021.
- K.-C. Hsu*, D. P. Nguyen*, J. Fisac, "[ISAACS: Iterative Soft Adversarial
  Actor-Critic for Safety](https://arxiv.org/abs/2212.03228)," L4DC 2023.
