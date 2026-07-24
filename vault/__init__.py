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

Contact/slam extension (not part of f_cert; needs mujoco_plant(contact_geometry=True))
---------------------------------------------------------------------------------------
contact_margin    SDF-to-terrain (non-wheel/foot) + slam (impact normal velocity) margins
target_margin     the target/"reach" margin (at-a-stop neighborhood) for reach-avoid training
mujoco_env        ContactSafetyEnv: SafetySAC reach-avoid RL env on the real MuJoCo plant;
                  reach_avoid=True also terminates on reaching the target set
train_contact     SafetySAC training for ContactSafetyEnv (avoid-only fallback)
reach_avoid_sac   ReachAvoidSafetySAC: SafetySAC + the target-set Bellman extension
train_reach_avoid ReachAvoidSafetySAC training for ContactSafetyEnv(reach_avoid=True)
reach_avoid_eval  comparative rollout eval: avoid-only vs reach-avoid fallback
calibration_check does the critic's Q(x,pi_safe(x)) actually predict realized rollout safety?
safety_filter     modular fallback / monitor / intervention safety-filter architecture --
                  see its docstring for the constraint-set/safe-set/target-set vocabulary
                  (Hsu, Hu, Fisac 2024, "The Safety Filter", arXiv:2309.05837)
teleop            keyboard/gamepad test rig: deployed controller -> safety_filter -> plant
                  (requires the vault-controller repo)

Submodules are imported lazily (no torch/mujoco import cost unless used).
"""
__all__ = ["config", "dynamics", "f_cert"]
