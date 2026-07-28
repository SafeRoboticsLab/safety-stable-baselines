# vault — context for Claude Code

Certifiable reach-avoid safety for the Vault robot's balance subsystem, over the reduced 4-state
`x = [v, theta, theta_dot, psi_dot]`. Reach-avoid HJ value function + a least-restrictive CBF-QP
value filter + the SafetySAC RL env + a MuJoCo plant for adversarial RL / ISAACS.

## Scope / boundaries
- This package shares ONLY the 4-state reduced dynamics + the safety method +
  the MuJoCo balance plant. It consumes the pinned model release from the
  sibling `vault-controller` checkout and must NOT grow a dependency on the
  Vault hybrid simulator, Julia/symbolic framework, or full robot model.
- `model_release.py` locates `vault-controller` through
  `VAULT_CONTROLLER_ROOT` (default `../vault-controller`), requires this
  package's `data/MODEL_INPUTS.lock.json` to byte-match the release lock, and
  hash-checks artifacts on first access.
- The reduced dynamics live in `dynamics.py`, which compiles the release's
  single opt6 kernel. `f_cert.py` is the ONE source of the one-step model +
  margins; the env, grid solver, and filter all call it. Don't fork the
  dynamics or vendor another kernel.
- The deployed controller is NOT here — it's the separate private `vault-controller` repo.
  `evaluate.py` imports it and fails loud if absent (no fallback controller, by design).

## Run / regenerate
Run as modules from the repo root: `python -m vault.<name>`.
- `python -m vault.grid`     regenerate the value function -> `data/grid_reachavoid_odd.npz`
- `python -m vault.distill`  conservative `V_mlp` from the grid -> `models/`
- `python -m vault.train`    SafetySAC reach-avoid V + pi_safe
- `python -m vault.evaluate` filter using the controller source checkout
The reduced-dynamics kernel compiles on first artifact use (needs `cc`/`gcc`);
it is cached.

## Conventions
- Python, Google style. State `x=[v,theta,theta_dot,psi_dot]`; control `u=[tau_L,tau_R]` (N·m);
  `mu` = friction. The value function is mu-aware.
- Constants live in `config.py` — import from there, don't hard-code params or ODD bounds.
- Keep modules importable without heavy deps unless used (torch only in distill/filter/train;
  mujoco only in mujoco_plant/evaluate).

## Do NOT
- Do not add hybrid-sim / framework / full-robot-model dependencies.
- Do not duplicate the dynamics — extend `f_cert.py`.
- Do not commit `models/` or compiled `*.so` (see `.gitignore`); they regenerate.
