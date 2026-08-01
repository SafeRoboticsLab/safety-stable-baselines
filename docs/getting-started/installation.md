# Installation

`safety_sb3` is a pure-Python add-on for Stable-Baselines3. The core install pulls in
SB3, PyTorch, Gymnasium and NumPy and nothing exotic.

!!! info "Not on PyPI"
    The package is **not published on PyPI**. `pip install safety_sb3` will not find
    it. Install from the GitHub repository, pinned to a release tag for
    reproducibility (the current release is `v0.4.0`).

## Supported versions

| | requirement |
|---|---|
| Python | ≥ 3.10 (validated on 3.10 and 3.11) |
| Stable-Baselines3 | ≥ 2.0.0 |
| PyTorch | ≥ 1.13 |
| Gymnasium | ≥ 0.28 |
| NumPy | ≥ 1.23 |

The optional benchmark submodules (below) pin **Python 3.10 exactly** for
`safety-gymnasium`'s sake; the core library itself is happy on any ≥ 3.10.

## Install from GitHub (recommended)

Pin to a release tag so an environment is reproducible:

```bash
pip install "safety_sb3 @ git+https://github.com/SafeRoboticsLab/safety-stable-baselines.git@v0.4.0"
```

To depend on it from another project's requirements, use the same specifier. **Pin the
tag** — `v0.4.0` is a breaking rename of every learner class (see the
[release notes](../release-notes.md)); an unpinned dependency can move under you.

If you need the pre-rename (`Isaacs*` / `Gameplay*`) class names, pin the last release
that had them:

```bash
pip install "safety_sb3 @ git+https://github.com/SafeRoboticsLab/safety-stable-baselines.git@v0.3.0"
```

## Editable / contributor install

Clone and install in editable mode to hack on the library or run the examples:

```bash
conda create --name safety_sb3 python=3.10   # 3.10 keeps the optional submodules happy
conda activate safety_sb3

git clone https://github.com/SafeRoboticsLab/safety-stable-baselines.git
cd safety-stable-baselines
pip install -e .
pip install -e ".[tests]"       # adds pytest, to run the test suite
```

## Optional: benchmark / integration dependencies

Only needed to run the bundled `examples/` against the external benchmark environments.
This repo vendors [`safety-gymnasium` (SafeRoboticsLab fork)](https://github.com/SafeRoboticsLab/safety-gymnasium)
and `rl_baselines3_zoo` as git submodules (Python 3.10 is pinned for
`safety-gymnasium`):

```bash
git submodule update --init
pip install -e integrations/rl_baselines3_zoo
pip install -e integrations/safety-gymnasium
```

The shipped reference environment ([Bicycle5D](../environments/bicycle5d.md)) and the
Pendulum quickstart need **none** of these — they are NumPy-only and CPU-trainable.

## GPU considerations

The core library trains fine on CPU (the [Quickstart](quickstart.md) and
[Bicycle5D](../environments/bicycle5d.md) converge in minutes on a laptop). A CUDA
build of PyTorch is only worth it for the **tensor path** — thousands of parallel,
GPU-resident environments via `TensorVecEnv`. Install the appropriate PyTorch CUDA
wheel for your driver *before* installing `safety_sb3` if you need it; nothing in this
package constrains the CUDA version.

## Verify the install

```bash
python -c "import safety_sb3; print('installation OK')"
```

For a stronger check that the learner classes import cleanly:

```bash
python -c "from safety_sb3 import SafetySAC1P, ReachAvoidPPO1P, SafetySAC2P; print('learners OK')"
```

Then run the test suite (editable install):

```bash
python -m pytest tests/ -q
```

## Next steps

- [Quickstart](quickstart.md) — train your first safety policy on Pendulum.
- [Choose a learner](choose-a-learner.md) — pick the right class for your problem.
