# safety-stable-baselines

Lightweight add-on for [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
implementing **Hamilton–Jacobi safety RL** (Fisac et al., ICRA '19), **reach-avoid RL**
(Hsu et al., RSS '21) and **adversarial (two-player) safety RL** (Hsu, Nguyen et al.,
L4DC '23) — plus a GPU-resident tensor path for massively parallel simulators
(mjlab / Isaac-style).

**Design principle: keep upstream SB3 untouched.** Everything lives in this separate
package and still feels native to SB3 users: same constructors, same `learn()`, same
callbacks and loggers.

📖 **[docs/API.md](docs/API.md)** is the canonical API reference — the env contract,
the MAP taxonomy, the backups, and `terminal_type`. Start there to integrate the
algorithms into your own project.

## Algorithms — here's the MAP

**Here's the MAP to navigate the codebase — Mode. Algorithm. Players.**

```
M = Mode       Safety | ReachAvoid | Cumulative    (which Bellman operator)
A = Algorithm  PPO | SAC | A2C | DQN               (which RL method)
P = Players    1P | 2P                             (single | zero-sum game)
```

Every class name is those three axes, in that order, and nothing else:

```
SafetyPPO1P        avoid, PPO, single-player
ReachAvoidSAC2P    reach-avoid, SAC, control-vs-disturbance
CumulativePPO1P    ordinary reward-maximizing PPO
```

So the roster is just the product. Pick the **Mode** that matches your task first —
the three take *genuinely different* value operators, and avoid is **not**
expressible as a reach-avoid instance (see below).

| Algorithm | `Safety` (stay safe forever) | `ReachAvoid` (reach it, staying safe throughout) | `Cumulative` (ordinary RL) |
|---|---|---|---|
| **PPO** | `SafetyPPO1P` `SafetyPPO2P` | `ReachAvoidPPO1P` `ReachAvoidPPO2P` | `CumulativePPO1P` |
| **SAC** | `SafetySAC1P` `SafetySAC2P` | `ReachAvoidSAC1P` `ReachAvoidSAC2P` | `CumulativeSAC1P` |
| **A2C** | `SafetyA2C1P` | `ReachAvoidA2C1P` | `CumulativeA2C1P` |
| **DQN** | `SafetyDQN1P` | — *(not possible, see below)* | `CumulativeDQN1P` |

| Mode | backup (value target) | anchor |
|---|---|---|
| `Safety` | `min(g, V')` | `g` |
| `ReachAvoid` | `min(g, max(l, V'))` | `min(l, g)` |
| `Cumulative` | `r + γ·V'` | — (not a margin) |

In `2P`, the control player maximizes the value and a disturbance player minimizes it,
with a league of archived opponents to damp cycling.

Two gaps in the product are deliberate. **DQN has no reach-avoid variant**: SB3's
discrete-action replay buffer carries no target margin `l(s)`, so the operator is not
computable there — asking for it raises rather than quietly computing something else.
**`Cumulative` has no 2P variant**: an adversarial game whose value is a discounted
return is a different research question, not a mode of these.

`Cumulative` is not a safety mode — its reward is a reward, not a margin, and `V ≥ 0`
means nothing. It exists so a *nominal* baseline runs through every line of the same
code as the safety learners: the control for "is the safety operator doing the work,
or is it just PPO?"

<details>
<summary>Which papers do these correspond to?</summary>

You should not need this to use the library — that is the point of MAP — but for
citation: `Safety*1P` is Fisac et al. ICRA'19; `ReachAvoid*1P` is Hsu et al. RSS'21;
`Safety*2P` is **ISAACS** (Hsu, Nguyen et al. L4DC'23), the two-player *avoid* game,
whose paper has no target set and no `l`; `ReachAvoid*2P` is **Gameplay Filters**
(Hsu et al. 2024), which extends ISAACS to reach-avoid. Before v0.4.0 those last two
were named `Isaacs*` and `Gameplay*` — see [RELEASE_NOTES.md](RELEASE_NOTES.md).

</details>

Every backup is defined once in [`safety_sb3/backups.py`](safety_sb3/backups.py)
and shared by all learners; read that module for the operators and their
derivations. Learners call `backups.target(mode, …)` rather than owning a
backup, so a class above is essentially a mode choice (`mode=` also works
per instance) — and ordinary reward-maximizing RL is the third mode,
`backups.CUMULATIVE`, for a nominal baseline on the same code.

All backups use the time-discounted convention

```
target = nt·( (1 − γ)·anchor + γ·backup ) + (1 − nt)·terminal
```

with `nt = 1` on non-terminal steps. **The anchor differs by problem.** It is the
"episode terminates now" payoff (`1 − γ` is the termination probability), so:

- **avoid** → `g`: stopping now scores well iff you are safe.
- **reach-avoid** → `min(l, g)`: stopping now scores well iff you are **in the target
  _and_ safe**.

The reach-avoid anchor is the discounted reach-avoid Bellman equation of Hsu et al.
(RSS'21, eq. 15) and Gameplay Filters (eq. 6a), and is the same expression as their
finite-horizon terminal condition `V_H = min(l, g)` (eq. 5b).

ISAACS (eq. 6) anchors on `g`, but it is a pure **avoid** game — no target set, no `l`
anywhere. That anchor does not carry over: Gameplay Filters exists precisely to extend
ISAACS to reach-avoid, and it changes the anchor when it does.

Anchoring reach-avoid on `g` makes "stay safe forever, never reach" a fixed point at
`V = g > 0` — a win — when its true reach-avoid value is `maxₜ l(sₜ) < 0`. The result is
neither the reach-avoid value nor the avoid value; RSS'21's under-approximation theorem
stops applying, so the critic can wrongly certify reachability. RSS'21 says of the
`g`-anchored form (its eq. 13) that it approximates "safety or liveness problems, **but
not both**".

The `ReachAvoid*` learners take `terminal_type` (`"all"` → `min(l, g)`, the default
and the horizon condition; `"g"` → `g` alone), matching the reference implementation.

#### Avoid is not a reach-avoid instance — don't degenerate `l`

A recurring temptation is to run an avoid task on a reach-avoid learner by pinning `l`
to a constant. It cannot work. The reduction needs the anchor to reduce
(`min(l,g) = g` ⟹ `l ≥ g`) **and** the recursion to reduce (`max(l,V') = V'` ⟹
`l ≤ V'`); since `V' ≤ g`, that demands `l ≥ g ≥ V' ≥ l`. No `l` satisfies it:

- `l ≡ −C` (large negative) buys the recursion, destroys the anchor → `V ≡ −C`
  everywhere, independent of the dynamics: **an empty safe set, with healthy-looking
  `ep_len`/`ep_rew`/`critic_loss` throughout.**
- `l ≡ 0` or `+C` buys the anchor, destroys the recursion → the target is everywhere, so
  you are "already done" at `t=0`, `max(l, ·)` clips every negative future, and `V ≡ g`:
  a myopic "am I safe right now" with no lookahead.

Use the avoid row of the table. That is what the reference does — it switches operator
rather than hunting for a clever `l`.

### Margin conventions (the env contract)

- **`g(s)` — safety margin — rides on the reward channel.** `g ≥ 0` iff the state is
  outside the failure set. The env must `terminate` the episode when `g < 0`.
- **`l(s)` — target margin — rides on `info["l_x"]`** (numpy path) or is returned
  directly by `step_tensor` (tensor path). `l ≥ 0` iff the state is inside the target
  set. Only the `ReachAvoid*` learners read it — an avoid learner has nowhere to put
  it, by construction.
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
rollout collection with the `Tensor*RolloutBuffer` matching its Mode
(on-policy) or `TensorReplayBuffer` (off-policy) — identical backup math, no numpy on
the hot path. `TensorVecNormalize` provides on-device running observation
normalization. See `tests/test_tensor_sac.py` for a complete minimal example
(a 64-env double integrator, CPU-runnable).

## Installation

```bash
conda create --name safety_sb3 python=3.10   # any >=3.10 works for the core
conda activate safety_sb3

git clone git@github.com:SafeRoboticsLab/safety-stable-baselines.git
cd safety-stable-baselines
pip install -e .
```

That is the whole core install (deps: `stable-baselines3`, `torch`, `gymnasium`,
`numpy`, `tensorboard`, `wandb`). To depend on a released version from another
project (e.g. [robot-safety-sandbox](https://github.com/SafeRoboticsLab/robot-safety-sandbox)):

```
safety_sb3 @ git+https://github.com/SafeRoboticsLab/safety-stable-baselines.git@v0.1.0
```

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
from safety_sb3 import SafetySAC1P


class PendulumSafety(gym.Wrapper):
    """Reward == safety margin g(s) = pi/6 - |theta| (safe iff |theta| <= 30 deg)."""

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        theta = np.arctan2(obs[1], obs[0])
        g = np.pi / 6 - abs(theta)
        return obs, float(g), terminated or g < 0, truncated, info


model = SafetySAC1P("MlpPolicy", PendulumSafety(gym.make("Pendulum-v1")),
                  gamma=0.995, verbose=1)
model.learn(100_000)
```

For reach-avoid, additionally return the target margin in the info dict
(`info["l_x"] = l`) and train `ReachAvoidPPO1P` / `ReachAvoidSAC1P` the same way — the
class name is the only thing that changes.
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
- `tests/test_taxonomy.py` — **MAP itself**: every class's `_MODE` and player count
  match its name, the exported roster is exactly the product, buffers carry a Mode and
  no player count, reach-avoid is a mixin that avoid learners never receive, and DQN
  refuses reach-avoid.
- `tests/test_ppo_smoke.py` — `SafetyPPO1P`/`ReachAvoidPPO1P` end-to-end on a 1-D double
  integrator (seconds, CPU).
- `tests/test_tensor_sac.py` — tensor-path buffer semantics + `SafetySAC1P`/
  `ReachAvoidSAC1P` (and the 2P pair) learning on the same task (~2 min, CPU).
- `tests/test_abstract_dp.py` — the backup is a *parameter*: mode dispatch on the
  SAC/DQN/buffer paths, the cumulative (standard-RL) operator, `train()` living on the
  players axis, and the regression guard for the entropy-temperature clamp the old
  copied `train()` had dropped.
- `tests/test_two_player_lr.py` — per-network / per-actor learning rates and StepLR
  decay in the two-player SAC learners.

## References

- J. Fisac et al., "[Bridging Hamilton-Jacobi Safety Analysis and Reinforcement
  Learning](https://ieeexplore.ieee.org/document/8794107)," ICRA 2019.
- K.-C. Hsu, V. Rubies-Royo, C. Tomlin, J. Fisac, "[Safety and Liveness Guarantees
  through Reach-Avoid Reinforcement Learning](https://arxiv.org/abs/2112.12288)," RSS 2021.
- K.-C. Hsu*, D. P. Nguyen*, J. Fisac, "[ISAACS: Iterative Soft Adversarial
  Actor-Critic for Safety](https://arxiv.org/abs/2212.03228)," L4DC 2023.
