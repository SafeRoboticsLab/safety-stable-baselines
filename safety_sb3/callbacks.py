"""Callbacks for stable safety-policy fine-tuning.

Empirically, on-policy training against margin-only objectives (safety or
reach-avoid backups) carries no gait-/action-quality gradient, so the usual
PPO entropy dynamics can inflate the action std until a converged motor skill
erodes. Capping the std is a one-line remedy that preserved fine-tuned
capability where uncapped runs lost it. Pair with SB3's native ``target_kl``
early-stop for the actor epochs.
"""
import math

import torch as th
from stable_baselines3.common.callbacks import BaseCallback


class StdCapCallback(BaseCallback):
  """Clamp the policy's action std at the start of every rollout.

  Works with any SB3 on-policy algorithm whose policy exposes a state-
  independent ``log_std`` parameter (the default ``MlpPolicy`` for Box
  action spaces). Policies without ``log_std`` are silently ignored.

  :param max_std: hard upper bound on the (post-clamp) action std.
  """

  def __init__(self, max_std: float, verbose: int = 0):
    super().__init__(verbose)
    if max_std <= 0:
      raise ValueError(f"max_std must be positive, got {max_std}")
    self.max_log_std = math.log(max_std)

  def _on_rollout_start(self) -> None:
    if hasattr(self.model.policy, "log_std"):
      with th.no_grad():
        self.model.policy.log_std.clamp_(max=self.max_log_std)

  def _on_step(self) -> bool:
    return True
