import torch as th
import numpy as np
from stable_baselines3.common.buffers import RolloutBuffer

from . import backups


class SafetyRolloutBuffer(RolloutBuffer):
    """Rollout buffer for the on-policy AVOID backup.

    ``g(s)`` rides on the reward channel. The operator is
    ``V(s) = (1-gamma)*g + gamma*min(g, V(s'))`` with terminal target ``g`` —
    see :mod:`safety_sb3.backups`, which defines it.
    """

    def _target(self, step: int, v_next: np.ndarray,
                not_done: np.ndarray) -> np.ndarray:
        return backups.avoid_target(self.rewards[step], v_next, not_done,
                                    self.gamma)

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Largely follows the original RolloutBuffer's compute_returns_and_advantage_safety method
        from stable_baselines3.
        We replace the usual GAE with the safety Bellman backup.
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()  # type: ignore[assignment]

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                v_next = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                v_next = self.values[step + 1]

            target = self._target(step, v_next, next_non_terminal)
            delta = target - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        self.returns = self.advantages + self.values


class ReachAvoidRolloutBuffer(SafetyRolloutBuffer):
    """Rollout buffer for the on-policy REACH-AVOID backup.

    Stores the per-step target margin ``l(s)`` alongside the safety margin
    ``g(s)`` (which rides on ``rewards``, as in :class:`SafetyRolloutBuffer`).
    ``l_x`` is written by the algorithm's ``collect_rollouts`` (SB3 rollout
    buffers do not receive ``infos``). The operator is

        ``V(s) = (1-gamma)*min(l, g) + gamma*min(g, max(l, V(s')))``

    — Hsu et al. RSS'21 eq. 15 / Gameplay Filters eq. 6a. See
    :mod:`safety_sb3.backups` for why the anchor is ``min(l, g)`` and not
    ``g``, and why avoid is not expressible by degenerating ``l``.

    :param terminal_type: ``"all"`` (default) -> terminal target ``min(l, g)``;
        ``"g"`` -> ``g``. See :func:`safety_sb3.backups.reach_avoid_target`.
    """

    def __init__(self, *args, terminal_type: str = "all", **kwargs):
        super().__init__(*args, **kwargs)
        self.terminal_type = backups.check_terminal_type(terminal_type)

    def reset(self) -> None:
        super().reset()
        self.l_x = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def _target(self, step: int, v_next: np.ndarray,
                not_done: np.ndarray) -> np.ndarray:
        return backups.reach_avoid_target(self.rewards[step], self.l_x[step],
                                          v_next, not_done, self.gamma,
                                          self.terminal_type)
