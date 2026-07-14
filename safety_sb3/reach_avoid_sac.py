"""Reach-avoid SAC (single-agent) — increment 1 toward ISAACS.

Extends :class:`SafetySAC` (avoid-only ``min(g, V')``) to the **reach-avoid**
Bellman backup::

    non-terminal:  y = (1-γ)·min(l, g) + γ·min(g, max(l, V'))
    terminal:      y = min(l, g)

where ``g(s)`` is the safety/avoid margin (rides on the ``reward`` field) and
``l(s)`` is the target/reach margin (supplied by the env via ``info["l_x"]`` and
stored by :class:`ReachAvoidReplayBuffer`).  ``V'`` is SAC's soft next-state
value (twin-target min minus entropy).  This matches the discounted reach-avoid
update in ``safe_adaptation_dev`` (``utils/train.get_bellman_update``,
``mode='reach-avoid'``, ``terminal_type='all'``).

This is the single-agent (max-player only) base; the ISAACS disturbance actor +
leaderboard are added in later increments on top of this.
"""

from __future__ import annotations

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.utils import polyak_update

from safety_sb3.isaacs_buffers import ReachAvoidReplayBuffer
from safety_sb3.safety_sac import SafetySAC


class ReachAvoidSAC(SafetySAC):
  """SAC with the reach-avoid Bellman backup.

  Tensor path (``TensorVecEnv``): the device-resident
  :class:`~safety_sb3.tensor_replay.TensorReplayBuffer` is built with
  ``store_l=True`` (l comes from ``step_tensor``'s ``l_x`` return, not infos);
  ``train()`` below is unchanged."""

  _tensor_store_l = True  # tensor buffer stores l(s)

  def __init__(self, *args, replay_buffer_class=None, **kwargs) -> None:
    if replay_buffer_class is None:
      replay_buffer_class = ReachAvoidReplayBuffer
    super().__init__(*args, replay_buffer_class=replay_buffer_class, **kwargs)
    # On load (_init_setup_model=False) the buffer is built later; only validate
    # when it already exists. The tensor path builds its own l-carrying buffer.
    if (not self._tensor_path and self.replay_buffer is not None
        and not isinstance(self.replay_buffer, ReachAvoidReplayBuffer)):
      raise TypeError(
        "ReachAvoidSAC needs a ReachAvoidReplayBuffer (it stores l(s)); got "
        f"{type(self.replay_buffer).__name__}."
      )

  def train(self, gradient_steps: int, batch_size: int) -> None:
    self.policy.set_training_mode(True)
    optimizers = [self.actor.optimizer, self.critic.optimizer]
    if self.ent_coef_optimizer is not None:
      optimizers += [self.ent_coef_optimizer]
    self._update_learning_rate(optimizers)

    ent_coef_losses, ent_coefs = [], []
    actor_losses, critic_losses = [], []

    for gradient_step in range(gradient_steps):
      replay_data = self.replay_buffer.sample(
        batch_size, env=self._vec_normalize_env
      )
      if self.use_sde:
        self.actor.reset_noise()

      actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
      log_prob = log_prob.reshape(-1, 1)

      ent_coef_loss = None
      if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
        ent_coef = th.exp(self.log_ent_coef.detach())
        ent_coef_loss = -(
          self.log_ent_coef * (log_prob + self.target_entropy).detach()
        ).mean()
        ent_coef_losses.append(ent_coef_loss.item())
      else:
        ent_coef = self.ent_coef_tensor
      ent_coefs.append(ent_coef.item())

      if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
        self.ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optimizer.step()

      with th.no_grad():
        next_actions, next_log_prob = self.actor.action_log_prob(
          replay_data.next_observations
        )
        next_q_values = th.cat(
          self.critic_target(replay_data.next_observations, next_actions), dim=1
        )
        next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
        next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)

        # --- Reach-avoid Bellman backup ---
        gs = replay_data.rewards  # g(s): safety / avoid margin
        lx = replay_data.l_x  # l(s): target / reach margin
        not_done = 1.0 - replay_data.dones
        v_to_go = th.minimum(gs, th.maximum(lx, next_q_values))  # min(g, max(l, V'))
        terminal_target = th.minimum(lx, gs)  # min(l, g)
        target_q_values = (
          1.0 - self.gamma * not_done
        ) * terminal_target + self.gamma * not_done * v_to_go

      current_q_values = self.critic(
        replay_data.observations, replay_data.actions
      )
      critic_loss = 0.5 * sum(
        F.mse_loss(current_q, target_q_values) for current_q in current_q_values
      )
      critic_losses.append(critic_loss.item())

      self.critic.optimizer.zero_grad()
      critic_loss.backward()
      self.critic.optimizer.step()

      # Control actor MAXIMIZES the reach-avoid value.
      q_values_pi = th.cat(
        self.critic(replay_data.observations, actions_pi), dim=1
      )
      min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
      actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
      actor_losses.append(actor_loss.item())

      self.actor.optimizer.zero_grad()
      actor_loss.backward()
      self.actor.optimizer.step()

      if gradient_step % self.target_update_interval == 0:
        polyak_update(
          self.critic.parameters(), self.critic_target.parameters(), self.tau
        )
        polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

    self._n_updates += gradient_steps
    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
    self.logger.record("train/ent_coef", np.mean(ent_coefs))
    self.logger.record("train/actor_loss", np.mean(actor_losses))
    self.logger.record("train/critic_loss", np.mean(critic_losses))
    if len(ent_coef_losses) > 0:
      self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
