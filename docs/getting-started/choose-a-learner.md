# Choose a learner

Every learner class is named by three choices — **M**ode · **A**lgorithm · **P**layers
(the [MAP convention](../map.md)). This page turns that into a practical decision: start
from your problem, read off a class, and only then learn what the axes mean.

## Start from your problem

| Your problem | Actions | Data regime | Start with |
|---|---|---|---|
| Remain safe (avoid a failure set) | Continuous | Sample-efficient | [`SafetySAC1P`](../reference.md#off-policy-learners-sac-family) |
| Reach a goal while staying safe | Continuous | Sample-efficient | [`ReachAvoidSAC1P`](../reference.md#off-policy-learners-sac-family) |
| Reach a goal while staying safe | Continuous | Many parallel envs (GPU) | [`ReachAvoidPPO1P`](../reference.md#on-policy-learners-ppo-family) |
| Remain safe against a disturbance | Continuous | Off-policy | [`SafetySAC2P`](../reference.md#off-policy-learners-sac-family) |
| Reach a goal robustly (vs disturbance) | Continuous | Many parallel envs (GPU) | [`ReachAvoidPPO2P`](../reference.md#on-policy-learners-ppo-family) |
| Ordinary RL baseline (no safety) | Continuous | Off-policy | [`CumulativeSAC1P`](../reference.md#off-policy-learners-sac-family) |
| Remain safe | Discrete | Off-policy | [`SafetyDQN1P`](../reference.md#a2c-and-dqn) |

If you want the certificate to be a clean, calibrated **safety filter**, prefer the SAC
family: `V(s) = min_i Q_i(s, π(s))` is calibrated over the action space, not just the
on-policy trajectory tube (see the [Bicycle5D value comparison](../environments/bicycle5d.md)).

## The three choices, after the recommendation

### 1. Mode — which safety problem

The Mode picks the Bellman operator (see [Backups](../concepts/backups.md)):

- **`Safety`** — *avoid*. Stay out of the failure set forever. Needs only a safety
  margin `g`.
- **`ReachAvoid`** — reach a target set while staying safe throughout. Needs a target
  margin `l` on `info["l_x"]` in addition to `g`.
- **`Cumulative`** — ordinary discounted-return RL. **Not a safety mode** — its value
  carries no certificate; it exists so a nominal baseline can share every line of code
  with the safety learners.

!!! danger "Avoid is not reach-avoid with a constant `l`"
    A recurring temptation is to run an avoid task on a reach-avoid learner by pinning
    `l` to a constant. It cannot work — a negative constant empties the safe set, a
    non-negative one strips the lookahead. Pick the `Safety` Mode.
    [Why, in detail.](../concepts/backups.md#avoid-is-not-a-reach-avoid-instance)

### 2. Algorithm — the ordinary RL choice

- **`SAC`** (off-policy) — sample-efficient; the calibrated critic makes the best
  certificates. The default choice for single-machine work and for filters.
- **`PPO`** / **`A2C`** (on-policy) — favored when you have many parallel environments,
  especially the GPU-resident [tensor path](../concepts/environment-contract.md#the-tensor-path).
- **`DQN`** (off-policy, discrete) — discrete actions only; **no reach-avoid variant**
  (a discrete replay buffer carries no `l`).

### 3. Players — single-agent or adversarial

- **`1P`** — control only.
- **`2P`** — control **vs** a worst-case disturbance adversary: a zero-sum game whose
  value is a *robust* safety certificate. The `*2P` learners take one concatenated
  action split by `ctrl_action_dim`, and `self.policy` is always the deployable control
  policy. See [Train an adversarial policy](../how-to/train-adversarial.md).

## The full roster

The classes are exactly the product of the three axes:

| Algorithm | `Safety` (stay safe) | `ReachAvoid` (reach it, safely) | `Cumulative` (ordinary RL) |
|---|---|---|---|
| **PPO** | `SafetyPPO1P` `SafetyPPO2P` | `ReachAvoidPPO1P` `ReachAvoidPPO2P` | `CumulativePPO1P` |
| **SAC** | `SafetySAC1P` `SafetySAC2P` | `ReachAvoidSAC1P` `ReachAvoidSAC2P` | `CumulativeSAC1P` |
| **A2C** | `SafetyA2C1P` | `ReachAvoidA2C1P` | `CumulativeA2C1P` |
| **DQN** | `SafetyDQN1P` | — *(no `l` in a discrete replay buffer)* | `CumulativeDQN1P` |

Two gaps are deliberate: **DQN has no reach-avoid** (the operator is not computable
without `l`), and **`Cumulative` has no `2P`** (an adversarial game whose value is a
discounted return is a different problem). For the naming law and the class hierarchy,
see [The MAP convention](../map.md).

## Next steps

- [Quickstart](quickstart.md) — a runnable `SafetySAC1P` / `ReachAvoidSAC1P` example.
- [Hyperparameters](../hyperparameters.md) — which knobs apply to which learner.
- [API reference](../reference.md) — constructor signatures per family.
