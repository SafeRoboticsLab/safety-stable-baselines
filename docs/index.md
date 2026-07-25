# safety-stable-baselines

An add-on for [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
implementing **Hamilton–Jacobi safety RL** (Fisac et al. 2019), **reach-avoid RL**
(Hsu et al. 2021), and **adversarial (two-player) safety RL** (Hsu, Nguyen et al.) —
plus a GPU-resident tensor path for massively parallel simulators.

Pairs with [robot-safety-sandbox](https://github.com/SafeRoboticsLab/robot-safety-sandbox),
the mjlab environment layer.

## Here's a MAP

**Here's a MAP to navigate the codebase — Mode. Algorithm. Players.**

```
M = Mode       Safety | ReachAvoid | Cumulative    (which Bellman operator)
A = Algorithm  PPO | SAC | A2C | DQN               (which RL method)
P = Players    1P | 2P                             (single | zero-sum game)
```

Class names are those three axes and nothing else, so the roster is the product and
you can read any name without looking anything up:

| Algorithm | `Safety` | `ReachAvoid` | `Cumulative` |
|---|---|---|---|
| **PPO** | `SafetyPPO1P` `SafetyPPO2P` | `ReachAvoidPPO1P` `ReachAvoidPPO2P` | `CumulativePPO1P` |
| **SAC** | `SafetySAC1P` `SafetySAC2P` | `ReachAvoidSAC1P` `ReachAvoidSAC2P` | `CumulativeSAC1P` |
| **A2C** | `SafetyA2C1P` | `ReachAvoidA2C1P` | `CumulativeA2C1P` |
| **DQN** | `SafetyDQN1P` | — *(no `l` in a discrete replay buffer)* | `CumulativeDQN1P` |

`Safety` = stay safe forever (avoid). `ReachAvoid` = reach the target while staying
safe throughout. `Cumulative` = ordinary discounted-return RL, included so a nominal
baseline shares every line of code with the safety learners. In `2P` a control player
maximizes the value and a disturbance player minimizes it.

The three Modes are genuinely different operators, and avoid is **not** expressible as
a reach-avoid instance with a degenerate `l` — the [API guide](API.md) proves it.

## Start here

- **[Environments](environments/index.md)** — reference envs that ship with the
  library (train in minutes on CPU), each with a GIF and a value-function figure.
- **[API guide](API.md)** — the env contract, the MAP taxonomy, the backups, `terminal_type`.
- **[Best practices](best-practices.md)** — training recipes and the pitfalls that cost real debugging.
- **[Release notes](release-notes.md)** — **v0.4.0 is a breaking rename**; the entry is a migration guide.
- **[Code reference](reference.md)** — auto-generated from source docstrings.

```python
from safety_sb3 import ReachAvoidPPO1P
model = ReachAvoidPPO1P("MlpPolicy", env, terminal_type="all")
model.learn(2_000_000_000)
```
