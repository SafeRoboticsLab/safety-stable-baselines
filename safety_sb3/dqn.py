"""DQN, written without a fixed Bellman backup.

    AbstractDQN
    ├─ SafetyDQN1P        _MODE = AVOID
    └─ CumulativeDQN1P    _MODE = CUMULATIVE

**There is deliberately no ReachAvoidDQN1P.** MAP would predict one, and it is
the single place in the taxonomy where the product does not close: SB3's
discrete-action ``ReplayBuffer`` carries no target margin ``l(s)``, and the
reach-avoid operator cannot be evaluated without it. That is a missing
capability, not a naming gap, so :class:`AbstractDQN` refuses
``mode='reach-avoid'`` loudly at construction rather than silently computing
something else. Use :class:`~safety_sb3.sac_1p.ReachAvoidSAC1P` (continuous
actions) or add an ``l``-carrying discrete replay buffer.

DQN is also single-player only — a two-player discrete game would need a joint
action-value table over both players' actions, which is a different learner.
"""
from __future__ import annotations

import numpy as np
import torch as th
import torch.nn.functional as F

from stable_baselines3.dqn.dqn import DQN

from . import backups
from .gamma_anneal import GammaAnnealMixin


class AbstractDQN(GammaAnnealMixin, DQN):
    """DQN whose TD target is the backup for ``self._MODE``, dispatched through
    :mod:`safety_sb3.backups` like every other learner in the library.

    ``gamma_anneal`` (ON by default) anneals the discount 0.99 -> 0.9999 over the
    first 50% of training (read as ``self.gamma`` in the TD target of
    ``train()``); the numpy off-policy loop applies it via
    ``_update_current_progress_remaining``. See ``gamma_anneal.py``.

    :param mode: backup to converge to; defaults to the class's ``_MODE``.
    """

    _MODE = backups.AVOID

    def __init__(self, *args, mode: str | None = None, gamma_anneal=True,
                 **kwargs):
        self._MODE = backups.check_mode(self._MODE if mode is None else mode)
        if self._MODE == backups.REACH_AVOID:
            raise ValueError(
                f"{type(self).__name__} cannot run mode='reach-avoid': its "
                "replay buffer stores no target margin l(s), so the reach-avoid "
                "operator is not computable here. Use ReachAvoidSAC1P, or add an "
                "l-carrying discrete replay buffer.")
        super().__init__(*args, **kwargs)
        self._setup_gamma_anneal(gamma_anneal)

    def train(self, gradient_steps: int, batch_size: int) -> None:
        """Largely follows the original DQN train method from stable_baselines3.
        The TD target is the backup for ``self._MODE`` (safety_sb3.backups).
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )  # type: ignore[union-attr]

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)

                # 1-step TD target for this learner's mode -- defined in
                # safety_sb3.backups. For AVOID this is the algebraically
                # identical rearrangement of the form this file used to inline:
                #   (1 - g*nt)*gs + g*nt*min(gs, V')
                #   == nt*((1-g)*gs + g*min(gs, V')) + (1-nt)*gs
                # i.e. the full gs is still returned at terminal states.
                gs = replay_data.rewards  # g(s) (a reward, in CUMULATIVE mode)
                not_done = 1.0 - replay_data.dones
                target_q_values = backups.target(
                    self._MODE, gs, next_q_values, not_done, self.gamma)

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


# ----------------------------------------------------------------- the modes

class SafetyDQN1P(AbstractDQN):
    """DQN with the **avoid** backup — ``Q(s,a) = (1-γ)·g + γ·min(g, max_a' Q')``
    (Fisac et al. 2019). ``g(s)`` rides on the reward channel."""

    _MODE = backups.AVOID


class CumulativeDQN1P(AbstractDQN):
    """Ordinary DQN — ``Q(s,a) = r + γ·max_a' Q'``. Not a safety learner; it is
    here so a reward-maximizing discrete baseline shares this library's code."""

    _MODE = backups.CUMULATIVE
