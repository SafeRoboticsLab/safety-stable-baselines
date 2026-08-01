"""safety-stable-baselines: safety / reach-avoid RL on Stable-Baselines3.

Here's the MAP to navigate the codebase — **Mode. Algorithm. Players.**

    M = Mode       Safety | ReachAvoid | Cumulative   (which Bellman operator)
    A = Algorithm  PPO | SAC | A2C | DQN              (which RL method)
    P = Players    1P | 2P                            (single | zero-sum game)

Read any class name straight off the axes::

    SafetyPPO1P       avoid, PPO, single-player
    ReachAvoidSAC2P   reach-avoid, SAC, control-vs-disturbance
    CumulativePPO1P   ordinary reward-maximizing PPO

so the full roster is just the product:

    ======================  ==========  ==============  ==========
    Algorithm               Safety      ReachAvoid      Cumulative
    ======================  ==========  ==============  ==========
    PPO 1P / PPO 2P         yes / yes   yes / yes       yes / --
    SAC 1P / SAC 2P         yes / yes   yes / yes       yes / --
    A2C 1P                  yes         yes             yes
    DQN 1P                  yes         NO (see below)  yes
    ======================  ==========  ==============  ==========

Two gaps are deliberate. **DQN has no reach-avoid variant**: its discrete replay
buffer carries no target margin ``l(s)``, so the operator is not computable
there, and asking for it raises. **Cumulative has no 2P variant**: an
adversarial game whose value is a discounted return is a different research
question, not a mode of these ones.

The Modes are genuinely different value operators — see :mod:`safety_sb3.backups`,
which defines all three in one place. Avoid is NOT expressible as a reach-avoid
instance with a degenerate ``l`` (that module proves it); pick the Mode that
matches your task. Every learner is written against ``backups.target(mode, ...)``
rather than a fixed operator, so a concrete class is a one-line ``_MODE``
specialization of an abstract algorithm, and ordinary reward-maximizing RL is the
third Mode rather than a separate library.

Structure follows Players first, then Mode: ``AbstractPPO`` -> ``AbstractPPO1P``
/ ``AbstractPPO2P`` -> the concretes. The players axis carries ~10x the code of
the mode axis, so putting it in the inheritance chain and the mode in a one-line
attribute (plus, for on-policy reach-avoid, the small
``_ReachAvoidPlumbing`` mixin) is what keeps the leaves one line each.
"""
from . import backups
from .a2c import (AbstractA2C, CumulativeA2C1P, ReachAvoidA2C1P, SafetyA2C1P)
from .buffers_replay import (ReachAvoidReplayBuffer,
                             ReachAvoidReplayBufferSamples)
from .buffers_rollout import (CumulativeRolloutBuffer, ReachAvoidRolloutBuffer,
                              SafetyRolloutBuffer)
from .buffers_tensor import (TensorCumulativeRolloutBuffer,
                             TensorReachAvoidRolloutBuffer,
                             TensorSafetyRolloutBuffer)
from .callbacks import StdCapCallback
from .dqn import AbstractDQN, CumulativeDQN1P, SafetyDQN1P
from .eval_callbacks import SafeSuccessRateEvalCallback
from .gamma_anneal import (GammaAnnealMixin, GeometricGammaAnneal,
                           StepGammaAnneal, make_default_gamma_schedule)
from .leaderboard import Leaderboard, LeagueEvaluator
from .policies import TwoPlayerSACPolicy
from .ppo_1p import (AbstractPPO1P, CumulativePPO1P, ReachAvoidPPO1P,
                     SafetyPPO1P)
from .ppo_2p import AbstractPPO2P, ReachAvoidPPO2P, SafetyPPO2P
from .ppo_base import AbstractPPO
from .sac_1p import (AbstractSAC1P, CumulativeSAC1P, ReachAvoidSAC1P,
                     SafetySAC1P)
from .sac_2p import AbstractSAC2P, ReachAvoidSAC2P, SafetySAC2P
from .sac_base import AbstractSAC
from .tensor_env import TensorVecEnv, TensorVecNormalize

__all__ = [
    "backups",
    # --- callbacks ---
    "StdCapCallback",
    "SafeSuccessRateEvalCallback",
    # --- PPO: Mode x Players ---
    "SafetyPPO1P",
    "ReachAvoidPPO1P",
    "CumulativePPO1P",
    "SafetyPPO2P",
    "ReachAvoidPPO2P",
    # --- SAC: Mode x Players ---
    "SafetySAC1P",
    "ReachAvoidSAC1P",
    "CumulativeSAC1P",
    "SafetySAC2P",
    "ReachAvoidSAC2P",
    # --- A2C / DQN (single-player only) ---
    "SafetyA2C1P",
    "ReachAvoidA2C1P",
    "CumulativeA2C1P",
    "SafetyDQN1P",
    "CumulativeDQN1P",
    # --- the abstract algorithms: the extension points for a new Mode or a new
    #     Player count. The *1P/*2P loops are usable directly with mode=; the
    #     bare AbstractPPO/AbstractSAC own no loop and refuse to run. ---
    "AbstractPPO",
    "AbstractPPO1P",
    "AbstractPPO2P",
    "AbstractSAC",
    "AbstractSAC1P",
    "AbstractSAC2P",
    "AbstractA2C",
    "AbstractDQN",
    # --- envs / buffers / policies ---
    "TensorVecEnv",
    "TensorVecNormalize",
    "SafetyRolloutBuffer",
    "ReachAvoidRolloutBuffer",
    "CumulativeRolloutBuffer",
    "TensorSafetyRolloutBuffer",
    "TensorReachAvoidRolloutBuffer",
    "TensorCumulativeRolloutBuffer",
    "ReachAvoidReplayBuffer",
    "ReachAvoidReplayBufferSamples",
    "TwoPlayerSACPolicy",
    # --- the two-player league ---
    "Leaderboard",
    "LeagueEvaluator",
    # --- gamma annealing (on by default in every learner) ---
    "GammaAnnealMixin",
    "GeometricGammaAnneal",
    "StepGammaAnneal",
    "make_default_gamma_schedule",
]
