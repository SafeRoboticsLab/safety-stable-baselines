# Code reference

Auto-generated from source docstrings. See the [API guide](API.md) for the
conceptual contract, and the [index](index.md) for the MAP naming law.

The module layout follows MAP: buffers carry the **Mode**, the `*_1p` / `*_2p`
modules carry the **Players**, and the `*_base` modules hold what both player counts
of an **Algorithm** share.

## Backups — the value operators

::: safety_sb3.backups

## On-policy learners (PPO family)

::: safety_sb3.ppo_base
    options:
      members: [AbstractPPO]

::: safety_sb3.ppo_1p
    options:
      members: [AbstractPPO1P, SafetyPPO1P, ReachAvoidPPO1P, CumulativePPO1P]

::: safety_sb3.ppo_2p
    options:
      members: [AbstractPPO2P, SafetyPPO2P, ReachAvoidPPO2P]

## Off-policy learners (SAC family)

::: safety_sb3.sac_base
    options:
      members: [AbstractSAC]

::: safety_sb3.sac_1p
    options:
      members: [AbstractSAC1P, SafetySAC1P, ReachAvoidSAC1P, CumulativeSAC1P]

::: safety_sb3.sac_2p
    options:
      members: [AbstractSAC2P, SafetySAC2P, ReachAvoidSAC2P]

## A2C and DQN

::: safety_sb3.a2c
    options:
      members: [AbstractA2C, SafetyA2C1P, ReachAvoidA2C1P, CumulativeA2C1P]

::: safety_sb3.dqn
    options:
      members: [AbstractDQN, SafetyDQN1P, CumulativeDQN1P]

## The reach-avoid mixin

::: safety_sb3.reach_avoid_mixin

## The two-player league

::: safety_sb3.leaderboard
    options:
      members: [Leaderboard, LeagueEvaluator]

## Policies

::: safety_sb3.policies
    options:
      members: [TwoPlayerSACPolicy]

## Discount (gamma) annealing

::: safety_sb3.gamma_anneal
    options:
      members: [StepGammaAnneal, GeometricGammaAnneal, GammaAnnealMixin, make_default_gamma_schedule]

## Evaluation callbacks

::: safety_sb3.eval_callbacks
    options:
      members: [SafeSuccessRateEvalCallback]

## Buffers

::: safety_sb3.buffers_rollout
    options:
      members: [SafetyRolloutBuffer, ReachAvoidRolloutBuffer, CumulativeRolloutBuffer, rollout_buffer_classes]

::: safety_sb3.buffers_tensor
    options:
      members: [TensorSafetyRolloutBuffer, TensorReachAvoidRolloutBuffer, TensorCumulativeRolloutBuffer]

::: safety_sb3.buffers_replay
    options:
      members: [ReachAvoidReplayBuffer]
