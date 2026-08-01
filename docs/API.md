# API guide (moved)

The single "API guide" has been split into focused pages so each topic has one home.
The `g`/`l` channel contract this page used to define is now the
[environment contract](concepts/environment-contract.md) — still the source of truth
shared with [robot-safety-sandbox](https://github.com/SafeRoboticsLab/robot-safety-sandbox);
keep the two in sync.

Where each part went:

| You were looking for | Now at |
|---|---|
| The environment contract (`g`, `l`, dones, NumPy vs tensor) | [Concepts → The environment contract](concepts/environment-contract.md) |
| What `V ≥ 0` means (margins, certificates) | [Concepts → Safety margins and certificates](concepts/safety-margins.md) |
| The backups, `terminal_type`, "avoid is not reach-avoid" | [Concepts → Backups and terminal handling](concepts/backups.md) |
| The MAP naming law and the learner roster | [The MAP convention](map.md) |
| Two-player (`*2P`) training, leagues | [How-to → Train an adversarial policy](how-to/train-adversarial.md) |
| Constructor arguments and applicability | [Hyperparameters](hyperparameters.md) |
| Signatures, parameters, return types | [API reference](reference.md) |

New here? Start at the [Quickstart](getting-started/quickstart.md).
