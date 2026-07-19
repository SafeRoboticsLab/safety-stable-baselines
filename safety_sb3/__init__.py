"""safety-stable-baselines: safety / reach-avoid RL on Stable-Baselines3.

The learners form a 2x2 over {problem} x {players}::

                 avoid (stay safe)        reach-avoid (reach, staying safe)
    single       SafetySAC/PPO/DQN/A2C    ReachAvoidSAC, ReachAvoidPPO
    two-player   IsaacsSAC, IsaacsPPO     GameplaySAC, GameplayPPO

The two problems take DIFFERENT value operators — see :mod:`safety_sb3.backups`,
which defines both in one place. Avoid is not expressible as a reach-avoid
instance with a degenerate ``l``; pick the row that matches your task.

.. warning::
   **v0.2.0 renamed the two-player reach-avoid classes.** ``IsaacsPPO`` /
   ``IsaacsSAC`` were reach-avoid and are now :class:`GameplayPPO` /
   :class:`GameplaySAC`; the ``Isaacs*`` names now mean the two-player AVOID
   game (ISAACS proper, Hsu et al. 2022 eq. 7), which the library previously
   lacked. Existing code importing ``IsaacsPPO`` will silently get a different
   algorithm. See RELEASE_NOTES.md.
"""
from . import backups
from .safety_sac import SafetySAC
from .safety_dqn import SafetyDQN
from .safety_buffers import SafetyRolloutBuffer
from .safety_ppo import SafetyPPO
from .tensor_env import TensorVecEnv, TensorVecNormalize
from .tensor_buffers import TensorReachAvoidRolloutBuffer, TensorSafetyRolloutBuffer
from .safety_a2c import SafetyA2C
from .reach_avoid_sac import ReachAvoidSAC
from .isaacs_buffers import ReachAvoidReplayBuffer, ReachAvoidReplayBufferSamples
from .isaacs_policy import IsaacsPolicy
from .isaacs import GameplaySAC, IsaacsSAC
from .safety_buffers import ReachAvoidRolloutBuffer
from .reach_avoid_ppo import ReachAvoidPPO
from .isaacs_ppo import GameplayPPO, IsaacsPPO
from .gamma_anneal import (GammaAnnealMixin, GeometricGammaAnneal,
                           StepGammaAnneal, make_default_gamma_schedule)
from .callbacks import StdCapCallback
from .eval_callbacks import SafeSuccessRateEvalCallback

__all__ = [
    "backups",
    "StdCapCallback",
    "SafeSuccessRateEvalCallback",
    # --- single-player, avoid ---
    "SafetySAC",
    "SafetyDQN",
    "SafetyPPO",
    "SafetyA2C",
    # --- single-player, reach-avoid ---
    "ReachAvoidSAC",
    "ReachAvoidPPO",
    # --- two-player, avoid (ISAACS proper) ---
    "IsaacsSAC",
    "IsaacsPPO",
    # --- two-player, reach-avoid (Gameplay Filters) ---
    "GameplaySAC",
    "GameplayPPO",
    # --- envs / buffers ---
    "TensorVecEnv",
    "TensorVecNormalize",
    "TensorSafetyRolloutBuffer",
    "TensorReachAvoidRolloutBuffer",
    "SafetyRolloutBuffer",
    "ReachAvoidRolloutBuffer",
    "ReachAvoidReplayBuffer",
    "ReachAvoidReplayBufferSamples",
    "IsaacsPolicy",
    # --- gamma annealing (on by default in every Safety* algo) ---
    "GammaAnnealMixin",
    "GeometricGammaAnneal",
    "make_default_gamma_schedule",
]
