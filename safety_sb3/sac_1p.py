"""Single-player SAC — the **1P** half of the SAC family, and its three modes.

    AbstractSAC                 (sac_base.py — ctor, buffers, entropy, backup)
    └─ AbstractSAC1P            the single-actor update loop
        ├─ SafetySAC1P          _MODE = AVOID
        ├─ ReachAvoidSAC1P      _MODE = REACH_AVOID
        └─ CumulativeSAC1P      _MODE = CUMULATIVE

There is exactly ONE ``train()`` here and the three learners below are one line
each. No reach-avoid mixin is needed on this side of the library: SAC's entire
mode delta is the replay-buffer choice and the TD target, and both already
dispatch on ``_MODE`` inside :class:`~safety_sb3.sac_base.AbstractSAC`.

That de-duplication is load-bearing, not cosmetic. These used to be two
hand-copied ``train()`` bodies, and the copy had silently dropped
``_clamp_entropy_temps()`` — so ``min_alpha``/``max_alpha`` did nothing on the
reach-avoid path. One shared body makes that unrepresentable.
"""
from __future__ import annotations

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.utils import polyak_update

from . import backups
from .sac_base import AbstractSAC


class AbstractSAC1P(AbstractSAC):
  """SAC's single-actor update loop; the backup is chosen by ``_MODE``."""

  def train(self, gradient_steps: int, batch_size: int) -> None:
    """Largely follows the original SAC train method from stable_baselines3.
    The TD target is the one for ``self._MODE`` (:meth:`_bellman_target`), so
    this one body serves every single-player mode in the library.
    """
    # Switch to train mode (this affects batch norm / dropout)
    self.policy.set_training_mode(True)
    # Update optimizers learning rate
    optimizers = [self.actor.optimizer, self.critic.optimizer]
    if self.ent_coef_optimizer is not None:
      optimizers += [self.ent_coef_optimizer]

    # Update learning rate according to lr schedule
    self._update_learning_rate(optimizers)

    ent_coef_losses, ent_coefs = [], []
    actor_losses, critic_losses = [], []

    for gradient_step in range(gradient_steps):
      # Sample replay buffer
      replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

      # We need to sample because `log_std` may have changed between two gradient steps
      if self.use_sde:
        self.actor.reset_noise()

      # Action by the current actor for the sampled state
      actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
      log_prob = log_prob.reshape(-1, 1)

      ent_coef_loss = None
      if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
        # Important: detach the variable from the graph
        # so we don't change it with other losses
        # see https://github.com/rail-berkeley/softlearning/issues/60
        ent_coef = th.exp(self.log_ent_coef.detach())
        ent_coef_loss = -(self.log_ent_coef *
                          (log_prob + self.target_entropy).detach()).mean()
        ent_coef_losses.append(ent_coef_loss.item())
      else:
        ent_coef = self.ent_coef_tensor

      ent_coefs.append(ent_coef.item())

      # Optimize entropy coefficient
      if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
        self.ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optimizer.step()
        self._clamp_entropy_temps()  # min_alpha/max_alpha floor/ceiling

      with th.no_grad():
        # Select action according to policy
        next_actions, next_log_prob = self.actor.action_log_prob(
          replay_data.next_observations
        )
        # Compute the next Q values: min over all critics targets
        next_q_values = th.cat(
          self.critic_target(replay_data.next_observations, next_actions), dim=1
        )
        next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
        # add entropy term
        next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)

        target_q_values = self._bellman_target(replay_data, next_q_values)

      # Get current Q-values estimates for each critic network
      # using action from the replay buffer
      current_q_values = self.critic(replay_data.observations, replay_data.actions)

      # Compute critic loss
      critic_loss = 0.5 * sum(
        F.mse_loss(current_q, target_q_values) for current_q in current_q_values
      )
      assert isinstance(critic_loss, th.Tensor)  # for type checker
      critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

      # Optimize the critic
      self.critic.optimizer.zero_grad()
      critic_loss.backward()
      self.critic.optimizer.step()

      # Compute actor loss -- the actor MAXIMIZES the value of self._MODE.
      # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
      # Min over all critic networks
      q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
      min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
      actor_loss = (ent_coef*log_prob - min_qf_pi).mean()
      actor_losses.append(actor_loss.item())

      # Optimize the actor
      self.actor.optimizer.zero_grad()
      actor_loss.backward()
      self.actor.optimizer.step()

      # Update target networks
      if gradient_step % self.target_update_interval == 0:
        polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
        # Copy running stats, see GH issue #996
        polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

    self._n_updates += gradient_steps

    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
    # record_mean, not record -- see AbstractSAC._record_window_max: the dump
    # cadence is ~49 train() calls, and record() keeps only the last of them.
    self.logger.record_mean("train/ent_coef", np.mean(ent_coefs))
    self.logger.record_mean("train/actor_loss", np.mean(actor_losses))
    self.logger.record_mean("train/critic_loss", np.mean(critic_losses))
    if len(ent_coef_losses) > 0:
      self.logger.record_mean("train/ent_coef_loss", np.mean(ent_coef_losses))
    # Onset detectors: a mean over the window hides a spike that starts inside
    # it. These two are the pair that moved first in the E057 divergence.
    self._record_window_max("train/critic_loss_max", float(max(critic_losses)))
    self._record_window_max("train/ent_coef_max", float(max(ent_coefs)))


# ----------------------------------------------------------------- the modes

class SafetySAC1P(AbstractSAC1P):
  """SAC with the **avoid** (safety) Bellman backup — Fisac et al. 2019::

      V(s) = (1-γ)·g + γ·min(g, V')        terminal: V(s) = g

  ``g(s)`` is the safety margin (it rides on the ``reward`` field); ``V'`` is
  SAC's soft next-state value. ``V(s) >= 0`` iff the state is in the safe set.
  For a task with a *target* to reach use :class:`ReachAvoidSAC1P`; the two
  operators are not interchangeable (see :mod:`safety_sb3.backups`).
  """

  _MODE = backups.AVOID
  _tensor_store_l = False  # no l(s) in an avoid problem


class ReachAvoidSAC1P(AbstractSAC1P):
  """SAC with the **reach-avoid** Bellman backup — Hsu et al. RSS'21::

      non-terminal:  y = (1-γ)·min(l, g) + γ·min(g, max(l, V'))
      terminal:      y = min(l, g)

  ``g(s)`` is the safety/avoid margin (rides on the ``reward`` field) and
  ``l(s)`` is the target/reach margin, supplied by the env via ``info["l_x"]``
  and stored by
  :class:`~safety_sb3.buffers_replay.ReachAvoidReplayBuffer` — which the base
  defaults to and validates off this ``_MODE``. Matches the discounted
  reach-avoid update in ``safe_adaptation_dev``
  (``utils/train.get_bellman_update``, ``mode='reach-avoid'``,
  ``terminal_type='all'``).

  Tensor path (``TensorVecEnv``): the device-resident
  :class:`~safety_sb3.tensor_replay.TensorReplayBuffer` is built with
  ``store_l=True`` (``l`` comes from ``step_tensor``'s ``l_x``, not infos).
  """

  _MODE = backups.REACH_AVOID
  _tensor_store_l = True  # tensor buffer stores l(s)


class CumulativeSAC1P(AbstractSAC1P):
  """Ordinary reward-maximizing SAC — ``y = r + γ·V'``.

  Not a safety learner: the ``reward`` field carries a reward, not a margin, and
  ``V >= 0`` means nothing. It exists so a *nominal* baseline runs through every
  line of the same actor/critic/entropy code as the safety learners — the
  standard control for "is the safety operator doing the work, or is it just
  SAC?".
  """

  _MODE = backups.CUMULATIVE
  _tensor_store_l = False
