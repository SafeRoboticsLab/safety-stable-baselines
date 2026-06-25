"""vault — certifiable safety for the 4-state balance robot.

A self-contained package for the reach-avoid Hamilton-Jacobi value function over the
reduced balance state x = [v, theta, theta_dot, psi_dot] and the least-restrictive
value filter built on it.

Modules
-------
dynamics      opt6 reduced 4-state dynamics (f + analytic Jacobians; vendored C kernel)
f_cert        certified one-step model + ODD margins (single source of the dynamics)
env           SafetySAC reach-avoid RL environment (fast, f_cert-based)
mujoco_plant  high-fidelity MuJoCo plant (validation + adversarial RL + ISAACS)
grid          4D grid HJ reach-avoid value iteration (regenerate the value function)
distill       conservative deployable V_mlp distilled from the grid value
train         SafetySAC training (reach-avoid V + pi_safe)
filter        CBF-QP least-restrictive value filter
evaluate      in-the-loop filter evaluation (requires the vault-controller repo)

Submodules are imported lazily (no torch/mujoco import cost unless used).
"""
__all__ = ["config", "dynamics", "f_cert"]
