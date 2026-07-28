# vault — certifiable safety for the 4-state balance robot

Reach-avoid Hamilton–Jacobi safety for the Vault delivery robot's balance subsystem, over the
reduced state **`x = [v, θ, θ̇, ψ̇]`** (forward speed, pitch, pitch rate, yaw rate). It provides the
exact grid value function, a conservative deployable value net, a least-restrictive CBF-QP value
filter, the SafetySAC RL environment, and a high-fidelity MuJoCo plant for adversarial RL / ISAACS.

This is **one of two repos**:

| repo | contents |
|------|----------|
| **this** (`safety-stable-baselines/vault`) | the safety method + RL + MuJoCo sim |
| **`vault-controller`** (private) | authoritative v2.2 model release and C iLQR controller source |

## Module map
| module | what it is |
|--------|-----------|
| `model_release.py` | strict loader for the sibling controller release and model-input lock |
| `config.py` | release-backed robot params plus this package's ODD and disturbance bounds |
| `dynamics.py` | opt6 reduced dynamics compiled from the single kernel in `vault-controller` |
| `f_cert.py` | the certified one-step model + ODD margins (one source for env/grid/filter) |
| `grid.py` | 4D grid HJ reach-avoid value iteration — **regenerate the value function** |
| `distill.py` | conservative deployable `V_mlp` distilled from the grid value |
| `filter.py` | ODD grid filter (`from_grid`), release MLP, and explicit capability-grid loader |
| `env.py` | `BalanceSafetyEnv` — fast f_cert reach-avoid RL env (SafetySAC) |
| `train.py` | SafetySAC training (reach-avoid V + `pi_safe`) |
| `mujoco_plant.py` (+ `mujoco_model.py`) | high-fidelity MuJoCo plant + adversary disturbance hook |
| `evaluate.py` | in-the-loop filter evaluation using the controller source checkout |

## Install
Requires **Python 3.10+** and a **C compiler** (`cc`/`gcc`) on PATH — the reduced-dynamics kernel
is compiled once on first import and cached.

```bash
# 1. environment
conda create -n safe-sb3 python=3.10 -y && conda activate safe-sb3
#    (or:  python -m venv .venv && source .venv/bin/activate)

# 2. install SafetySAC (this package lives inside the safety-stable-baselines repo) + deps
cd safety-stable-baselines
pip install -e .                          # provides safety_sb3 + stable-baselines3
pip install -r vault/requirements.txt

# 3. clone the exact controller model release beside this checkout
git clone <vault-controller-url> ../vault-controller
git -C ../vault-controller checkout feature/v2.2-model-release
export VAULT_CONTROLLER_ROOT="$PWD/../vault-controller"

# 4. verify the install (validates the lock, compiles the kernel, then solves a smoke grid)
python -m vault.grid --smoke              # should converge + print a safe-set %
```

All model-dependent modules require the pinned `vault-controller` checkout. By
default it is located at `../vault-controller`; set `VAULT_CONTROLLER_ROOT` to
an alternate path. On first artifact access, `model_release.py` requires
`vault/data/MODEL_INPUTS.lock.json` to be byte-identical to
`vault-controller/models/MODEL_INPUTS.lock.json` and verifies each consumed
artifact hash.

To run `evaluate.py`, also build the controller library:
```bash
make -C ../vault-controller/balance_controller/c lib
PYTHONPATH=$PWD:../vault-controller python -m vault.evaluate
```

Run everything as a module from the `safety-stable-baselines` repo root: `python -m vault.<name>`.

## Pipeline — and how to regenerate
```bash
python -m vault.grid                       # 4D HJ reach-avoid -> data/grid_reachavoid_odd.npz
python -m vault.distill                    # conservative V_mlp -> models/v_mlp.{pt,json}
python -m vault.train --steps 300000       # SafetySAC reach-avoid V + pi_safe
python -m vault.evaluate                   # filter using the pinned controller source checkout
```
`grid.py --smoke` runs a small foreground solve to sanity-check changes. The full grid solve writes
the oracle the deployable net is verified conservative against (`{V_mlp≥0} ⊆ {V_grid≥0}`).

## Adversarial RL / ISAACS
- `env.py` — fast f_cert env with a **bounded** disturbance (analytic worst case); good for the
  ego policy and quick iteration.
- `mujoco_plant.py` — the high-fidelity plant. `apply_disturbance(force, torque)` is the action
  channel for a **learned** adversary (e.g. an ISAACS xfrc-force adversary). `step(u)` takes the
  ego's `[tau_L, tau_R]`; `get_state()` returns the 4-state; `roll_deg()` flags liftoff.

A typical ISAACS loop: ego controls `step(u)`, adversary sets `apply_disturbance(...)` each step,
both trained against the reach-avoid margin (`f_cert.margin` / `f_cert.odd_margin`).

## Evaluating the filter with the controller
`evaluate.py` requires the **vault-controller** source and C library; there is
no fallback controller. This is a source/simulation evaluation and does not by
itself establish what firmware image is flashed on hardware.
```bash
make -C ../vault-controller/balance_controller/c lib
PYTHONPATH=../vault-controller python -m vault.evaluate
```

## Release-backed vs regenerable
- `data/` (committed): the byte-matching release lock, checkpoint quarantine
  manifest, and `grid_reachavoid_odd.npz` with its model-hash sidecar.
- Robot parameters, residuals, controller limits, the opt6 kernel, capability
  grids, and release MLPs are loaded from the pinned controller release rather
  than copied here.
- `models/` (git-ignored): `v_mlp.*` and the SafetySAC checkpoint — regenerate with
  `distill.py` / `train.py`.

## The method (brief)
`f_cert` = release opt6 coupled dynamics + per-wheel friction-cone cap (slip) +
the release coupled residual fit, one symplectic-Euler step. The grid solver computes the robust
reach-avoid value `V(x;μ) = min(g_odd, max_u min_d V(f_cert(x,u)+d))`, where `g_odd` is the signed
distance to the ODD (negative outside → leaving the envelope is failure by construction). The safe
set is `{V ≥ 0}`. The filter admits the task control while the worst-case one-step value stays
`≥ eps`, else applies the value-maximizing control (least-restrictive), rendering `{V ≥ 0}`
forward-invariant.
