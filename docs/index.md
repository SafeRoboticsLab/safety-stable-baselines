# safety-stable-baselines

An add-on for [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
implementing **Hamilton–Jacobi safety RL** (Fisac et al. 2019), **reach-avoid RL**
(Hsu et al. 2021), and **adversarial reach-avoid / ISAACS** (Hsu, Nguyen et al.) —
plus a GPU-resident tensor path for massively parallel simulators.

Pairs with [robot-safety-sandbox](https://github.com/SafeRoboticsLab/robot-safety-sandbox),
the mjlab environment layer.

## Start here

- **[Environments](environments/index.md)** — reference envs that ship with the
  library (train in minutes on CPU), each with a GIF and a value-function figure.
- **[API guide](API.md)** — the env contract, the 2×2 of learners, the backups, `terminal_type`.
- **[Best practices](best-practices.md)** — training recipes and the pitfalls that cost real debugging.
- **[Release notes](release-notes.md)** — what changed in v0.2.0 (breaking: the reach-avoid anchor fix + rename).
- **[Code reference](reference.md)** — auto-generated from source docstrings.

```python
from safety_sb3 import ReachAvoidPPO
model = ReachAvoidPPO("MlpPolicy", env, terminal_type="all")
model.learn(2_000_000_000)
```
