# The MAP convention

Every learner in `safety_sb3` is named by a three-axis law — **M**ode · **A**lgorithm ·
**P**layers. The name *is* the specification: given a name, you know exactly which Bellman operator,
which RL update, and whether there is an adversary. This page is the canonical reference for the
naming law and the class hierarchy behind it.

```
M = Mode        Safety | ReachAvoid | Cumulative     which Bellman operator (safety_sb3.backups)
A = Algorithm   PPO | SAC | A2C | DQN                which RL update rule
P = Players     1P | 2P                              single-agent | zero-sum (control vs disturbance)
```

A concrete learner is `‹Mode›‹Algorithm›‹Players›`; e.g., **`ReachAvoidSAC2P`** means the reach-avoid
Bellman backup, SAC update, two-player game.

## The roster

| Algorithm | `Safety` (stay safe ∀t) | `ReachAvoid` (reach the target, staying safe) | `Cumulative` (ordinary RL) |
|---|---|---|---|
| **PPO** | `SafetyPPO1P` `SafetyPPO2P` | `ReachAvoidPPO1P` `ReachAvoidPPO2P` | `CumulativePPO1P` |
| **SAC** | `SafetySAC1P` `SafetySAC2P` | `ReachAvoidSAC1P` `ReachAvoidSAC2P` | `CumulativeSAC1P` |
| **A2C** | `SafetyA2C1P` | `ReachAvoidA2C1P` | `CumulativeA2C1P` |
| **DQN** | `SafetyDQN1P` | — *(no reach-avoid)* | `CumulativeDQN1P` |

Two structural gaps, both enforced (not silent):

- **DQN has no `ReachAvoid`.** The reach-avoid operator needs a target-margin term that the discrete-action
  DQN path does not carry; `mode="reach-avoid"` on DQN raises at construction rather than computing
  something else. Use `ReachAvoidSAC1P` for continuous actions.
- **`Cumulative` has no `2P`.** A zero-sum game whose payoff is a discounted cumulative return has no
  reach-avoid/HJ fixed point to solve for, so the axis is intentionally absent.

## The three axes are orthogonal

- **Mode** picks the *Bellman operator*, and nothing else. It is carried by the class attribute
  `_MODE` (one of `backups.AVOID | REACH_AVOID | CUMULATIVE`) and consumed by
  [`safety_sb3.backups`](API.md); the same actor/critic/entropy machinery runs underneath every mode.
  In tasks, you usually don't name the class — you set `mode`, and the target's presence determines it:
  a task that declares a reach target (`l`) is `reach-avoid`, one that only has a safety margin (`g`)
  is `safety`, and an ordinary-reward task is `cumulative`.
- **Algorithm** picks the *update rule* — on-policy (`PPO`, `A2C`) or off-policy (`SAC`, `DQN`).
- **Players** picks *1P* (control only) or *2P* (control vs. a worst-case disturbance adversary — a
  zero-sum game, the ISAACS setting).

## The abstract hierarchy

Each algorithm has an abstract base that implements the *update loop* once; the `1P`/`2P` split adds
the player structure; and the three `Mode` leaves are one line each — they only set `_MODE` (the
reach-avoid leaves also mix in the small buffer/plumbing needed to store the target margin `l`). The
mode's behavior lives entirely in the shared backup, so the leaves stay trivial and the operator
cannot drift between algorithms.

```
AbstractSAC                         (sac_base.py — ctor, buffers, entropy temp, γ-anneal, TD target)
├─ AbstractSAC1P                    single-actor update loop
│   ├─ SafetySAC1P        _MODE = AVOID
│   ├─ ReachAvoidSAC1P    _MODE = REACH_AVOID   (+ stores l)
│   └─ CumulativeSAC1P    _MODE = CUMULATIVE
└─ AbstractSAC2P                    two actors (ctrl, dstb) over a concatenated action + a league
    ├─ SafetySAC2P        _MODE = AVOID
    └─ ReachAvoidSAC2P    _MODE = REACH_AVOID

AbstractPPO                         (ppo_base.py — the mode operator substitutes into the GAE return)
├─ AbstractPPO1P
│   ├─ SafetyPPO1P  ·  ReachAvoidPPO1P (+_ReachAvoidPlumbing)  ·  CumulativePPO1P
└─ AbstractPPO2P
    ├─ SafetyPPO2P  ·  ReachAvoidPPO2P (+_ReachAvoidPlumbing)

AbstractA2C  →  SafetyA2C1P · ReachAvoidA2C1P (+_ReachAvoidPlumbing) · CumulativeA2C1P
AbstractDQN  →  SafetyDQN1P · CumulativeDQN1P
```

The one-line leaf is deliberate and load-bearing: because a mode is *only* a `_MODE` value plus the
shared backup, adding an algorithm means writing its abstract update once and getting all its modes
for free. A mode fix (e.g., the correct reach-avoid target) lands in exactly one place for every
algorithm at once.

### `2P` is not one algorithm

`*PPO2P` and `*SAC2P` both hold two actors but play the game differently (on-policy alternation vs.
off-policy joint-action critic with a self-play league). See [API guide → 2P](API.md) for the
per-family details and the leaderboard knobs in [Hyperparameters](hyperparameters.md).

## How the environment layer composes with the MAP

The [`robot-safety-sandbox`](https://github.com/SafeRoboticsLab/robot-safety-sandbox) zoo never
hard-codes a learner. A task declares its `mode` (via its margins) and whether it
`supports_adversary`; the trainer then resolves the class by a **formula, not a lookup**:

```python
algo_name(task_id, adversary=False, family="on_policy") -> "ReachAvoidSAC2P"   # e.g.
#          └ Mode from the task            └ Players            └ Algorithm from --family
```

Thus, `train.py --config … --family {on_policy|off_policy}` on a task picks exactly one MAP cell. The two
layers stay decoupled: the registry re-declares the mode strings as literals (pinned by a test) and
returns *names only*, so the zoo never imports `safety_sb3`. This is why a task written once trains
under any algorithm/player combination the MAP allows.

Pick **Mode** by the problem, **Players** by whether you train against a disturbance, and
**Algorithm** as the ordinary RL choice. The name records all three.
