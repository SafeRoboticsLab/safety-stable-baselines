# Code reference

Auto-generated from source docstrings. See the [API guide](API.md) for the
conceptual contract.

## Backups — the two value operators

::: safety_sb3.backups

## On-policy learners (PPO family)

::: safety_sb3.reach_avoid_ppo
    options:
      members: [ReachAvoidPPO]

## Off-policy learners (SAC family)

::: safety_sb3.reach_avoid_sac
    options:
      members: [ReachAvoidSAC]

::: safety_sb3.isaacs
    options:
      members: [GameplaySAC, IsaacsSAC]

## Discount (gamma) annealing

::: safety_sb3.gamma_anneal
    options:
      members: [StepGammaAnneal, GeometricGammaAnneal, GammaAnnealMixin, make_default_gamma_schedule]

## Evaluation callbacks

::: safety_sb3.eval_callbacks
    options:
      members: [SafeSuccessRateEvalCallback]

## Rollout buffers

::: safety_sb3.safety_buffers
    options:
      members: [SafetyRolloutBuffer, ReachAvoidRolloutBuffer]
