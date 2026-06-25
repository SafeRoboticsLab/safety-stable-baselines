# vault — certifiable safety for the 4-state balance robot

Reach-avoid Hamilton–Jacobi safety for the Vault delivery robot's balance subsystem, over the
reduced state **`x = [v, θ, θ̇, ψ̇]`** (forward speed, pitch, pitch rate, yaw rate). It provides the
exact grid value function, a conservative deployable value net, a least-restrictive CBF-QP value
filter, the SafetySAC RL environment, and a high-fidelity MuJoCo plant for adversarial RL / ISAACS.

This is **one of two repos**:

| repo | contents |
|------|----------|
| **this** (`safety-stable-baselines/vault`) | the safety method + RL + MuJoCo sim |
| **`vault-controller`** (private) | the deployed C iLQR balance controller — `evaluate.py` imports it |

## Module map
| module | what it is |
|--------|-----------|
| `config.py` | single source of robot params, the ODD, and disturbance bounds |
| `dynamics.py` (+ `csrc/`) | opt6 reduced 4-state dynamics — `f` + analytic Jacobians (vendored C kernel) |
| `f_cert.py` | the certified one-step model + ODD margins (one source for env/grid/filter) |
| `grid.py` | 4D grid HJ reach-avoid value iteration — **regenerate the value function** |
| `distill.py` | conservative deployable `V_mlp` distilled from the grid value |
| `filter.py` | least-restrictive CBF-QP value filter (`from_mlp` / `from_grid`) |
| `env.py` | `BalanceSafetyEnv` — fast f_cert reach-avoid RL env (SafetySAC) |
| `train.py` | SafetySAC training (reach-avoid V + `pi_safe`) |
| `mujoco_plant.py` (+ `mujoco_model.py`) | high-fidelity MuJoCo plant + adversary disturbance hook |
| `evaluate.py` | in-the-loop filter eval on the **deployed controller** (needs `vault-controller`) |

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

# 3. verify the install (compiles the kernel, then a ~10s grid solve)
python -m vault.grid --smoke              # should converge + print a safe-set %
```

To run `evaluate.py` (the filter on the deployed controller), also clone + build **vault-controller**:
```bash
git clone <vault-controller-url> ../vault-controller
make -C ../vault-controller/balance_controller/c lib
PYTHONPATH=$PWD:../vault-controller python -m vault.evaluate
```

Run everything as a module from the `safety-stable-baselines` repo root: `python -m vault.<name>`.

## Pipeline — and how to regenerate
```bash
python -m vault.grid                       # 4D HJ reach-avoid -> data/grid_reachavoid_odd.npz
python -m vault.distill                    # conservative V_mlp -> models/v_mlp.{pt,json}
python -m vault.train --steps 300000       # SafetySAC reach-avoid V + pi_safe
python -m vault.evaluate                   # filter on the deployed controller (needs vault-controller)
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

## Evaluating the filter on the real controller
`evaluate.py` requires the **vault-controller** repo (the deployed C iLQR; there is no fallback
controller — by design, so results reflect what ships). Clone it beside this repo and:
```bash
make -C ../vault-controller/balance_controller/c lib
PYTHONPATH=../vault-controller python -m vault.evaluate
```

## Vendored vs regenerable
- `data/` (committed): `composite_params.json`, `coupled_residual_fit.json`, and
  `grid_reachavoid_odd.npz` (the exact value function — regenerate with `grid.py`).
- `models/` (git-ignored): `v_mlp.*` and the SafetySAC checkpoint — regenerate with
  `distill.py` / `train.py`.

## The method (brief)
`f_cert` = opt6 lossless coupled dynamics + per-wheel friction-cone cap (slip) + the measured
coupled friction residual, one symplectic-Euler step. The grid solver computes the robust
reach-avoid value `V(x;μ) = min(g_odd, max_u min_d V(f_cert(x,u)+d))`, where `g_odd` is the signed
distance to the ODD (negative outside → leaving the envelope is failure by construction). The safe
set is `{V ≥ 0}`. The filter admits the task control while the worst-case one-step value stays
`≥ eps`, else applies the value-maximizing control (least-restrictive), rendering `{V ≥ 0}`
forward-invariant.
