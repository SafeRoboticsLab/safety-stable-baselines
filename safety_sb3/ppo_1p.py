"""Single-player PPO — the **1P** half of the PPO family, and its three modes.

    AbstractPPO                 (ppo_base.py — recipe, buffers, adaptive LR)
    └─ AbstractPPO1P            one actor, one value net, one rollout buffer
        ├─ SafetyPPO1P          _MODE = AVOID
        ├─ ReachAvoidPPO1P      _MODE = REACH_AVOID   (+ _ReachAvoidPlumbing)
        └─ CumulativePPO1P      _MODE = CUMULATIVE

Everything specific to *one* player lives here: the two rollout loops (numpy /
GPU-resident), which are SB3's ``OnPolicyAlgorithm.collect_rollouts`` with the
timeout value-bootstrap gated off and one added call — the buffer is handed each
step's extras so it can keep whatever its own operator needs (see
:mod:`safety_sb3.buffers_rollout`).

The three concrete learners below are one line each. That is the point of the
split: the mode axis costs a line, the players axis costs a loop.
"""

from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from . import backups
from .buffers_tensor import TensorReachAvoidMaskedRolloutBuffer
from .ppo_base import AbstractPPO
from .reach_avoid_mixin import _ReachAvoidPlumbing


class AbstractPPO1P(AbstractPPO):
    """PPO with the ordinary single-actor rollout, backup chosen by ``_MODE``."""

    def _collect_rollouts_tensor(
        self,
        env,
        callback: BaseCallback,
        rollout_buffer,
        n_rollout_steps: int,
    ) -> bool:
        """GPU-resident rollout: policy forward, env step, buffer add and the
        backup all stay on device. Episode stats and env ``metrics()``
        (curriculum levels etc.) are recorded to the logger directly."""
        assert self._last_obs is not None
        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        callback.on_rollout_start()

        dev = env.device
        obs = self._last_obs
        if not th.is_tensor(obs):  # first call after _setup_learn
            obs = th.as_tensor(np.asarray(obs), dtype=th.float32, device=dev)
        episode_starts = th.as_tensor(
            np.asarray(self._last_episode_starts, dtype=np.float32), device=dev
        ) if not th.is_tensor(self._last_episode_starts) else self._last_episode_starts

        low = th.as_tensor(self.action_space.low, dtype=th.float32, device=dev)
        high = th.as_tensor(self.action_space.high, dtype=th.float32, device=dev)

        ep_ret = getattr(self, "_t_ep_ret", None)
        if ep_ret is None or ep_ret.shape[0] != env.num_envs:
            self._t_ep_ret = th.zeros(env.num_envs, device=dev)
            self._t_ep_len = th.zeros(env.num_envs, device=dev)
        fin_ret, fin_len = [], []

        n_steps = 0
        while n_steps < n_rollout_steps:
            with th.no_grad():
                actions, values, log_probs = self.policy(obs)
            clipped = th.clamp(actions, low, high)

            new_obs, rewards, dones, timeouts, l_x = env.step_tensor(clipped)
            self.num_timesteps += env.num_envs
            n_steps += 1

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            rollout_buffer.record_extras(l_x)
            # Opt-in per-step policy_mask (RAS handover training): a hybrid env exposes
            # ``_policy_mask`` (False = a frozen hand-off skill drove this step,
            # exclude it from the policy gradient). No-op for every other env /
            # buffer -- both getattr and hasattr miss -> byte-identical rollout.
            pm = getattr(env, "_policy_mask", None)
            if pm is not None and hasattr(rollout_buffer, "record_policy_mask"):
                rollout_buffer.record_policy_mask(pm)
            rollout_buffer.add(obs, actions, rewards, episode_starts,
                               values.flatten(), log_probs)

            self._t_ep_ret += rewards
            self._t_ep_len += 1.0
            if bool(dones.any()):
                d = dones.bool()
                fin_ret.append(self._t_ep_ret[d])
                fin_len.append(self._t_ep_len[d])
                self._t_ep_ret = th.where(d, th.zeros_like(self._t_ep_ret), self._t_ep_ret)
                self._t_ep_len = th.where(d, th.zeros_like(self._t_ep_len), self._t_ep_len)

            obs = new_obs
            episode_starts = dones.float()

        with th.no_grad():
            last_values = self.policy.predict_values(obs)
        rollout_buffer.compute_returns_and_advantage(
            last_values=last_values.flatten(), dones=dones.float())

        self._last_obs = obs
        self._last_episode_starts = episode_starts

        if fin_ret:
            self.logger.record("rollout/ep_rew_mean", float(th.cat(fin_ret).mean()))
            self.logger.record("rollout/ep_len_mean", float(th.cat(fin_len).mean()))
        for k, v in (env.metrics() or {}).items():
            self.logger.record(f"env/{k}", float(v))

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """SB3 ``OnPolicyAlgorithm.collect_rollouts`` with two safety changes:
        the timeout value-bootstrap is gated by ``self.bootstrap_on_timeout``
        (off by default — the reward is the physical margin g(s)), and the buffer
        is offered each step's ``infos`` before ``add()`` so it can capture the
        extras its operator needs. On the GPU-resident path this dispatches to
        ``_collect_rollouts_tensor``."""
        # Anneal gamma (buffer.gamma) for THIS rollout's GAE, then collect. The
        # tensor path bypasses SB3's _update_current_progress_remaining, so apply
        # it here explicitly (idempotent on the numpy path).
        self._apply_gamma_anneal()
        if self._tensor_path:
            return self._collect_rollouts_tensor(
                env, callback, rollout_buffer, n_rollout_steps)
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(
                        actions, self.action_space.low, self.action_space.high
                    )

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            # Timeout bootstrapping — DISABLED by default for safety (see the
            # module docstring). Stock PPO would corrupt g(s) here.
            if self.bootstrap_on_timeout:
                for idx, done in enumerate(dones):
                    if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                    ):
                        terminal_obs = self.policy.obs_to_tensor(
                            infos[idx]["terminal_observation"]
                        )[0]
                        with th.no_grad():
                            terminal_value = self.policy.predict_values(terminal_obs)[0]
                        rewards[idx] += self.gamma * terminal_value

            rollout_buffer.record_extras(infos)

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True


# ----------------------------------------------------------------- the modes

class SafetyPPO1P(AbstractPPO1P):
    """On-policy **avoid** RL — Fisac et al. 2019::

        V(s) = (1-γ)·g + γ·min(g, V')          terminal: V(s) = g

    ``g(s)`` is the safety margin and rides on the reward channel; ``V(s) >= 0``
    iff the state is in the safe set. There is no target set and no ``l``: this
    learner never reads ``info["l_x"]`` and its buffer has nowhere to put it.
    For a task with a target to reach use :class:`ReachAvoidPPO1P` — the two
    operators are not interchangeable (see :mod:`safety_sb3.backups`).
    """

    _MODE = backups.AVOID


class ReachAvoidPPO1P(_ReachAvoidPlumbing, AbstractPPO1P):
    """On-policy **reach-avoid** RL — Hsu et al. RSS'21 eq. 15::

        V(s) = (1-γ)·min(l, g) + γ·min(g, max(l, V'))

    ``g`` (safety margin) rides on the reward channel; ``l`` (target margin) is
    read from ``info["l_x"]`` on the numpy path and returned directly by
    ``TensorVecEnv.step_tensor`` on the GPU-resident path — in both cases by the
    *buffer*, not by this class.

    Unlike the SAC family this is fully on-policy: it pairs with vectorized envs
    at large ``n_envs`` (the GPU-parallel regime where off-policy reach-avoid
    saturates; validated on a 1280-env MuJoCo quadruped in ``unitree_rl_mjlab``).
    """

    _MODE = backups.REACH_AVOID


class ReachAvoidMaskedPPO1P(_ReachAvoidPlumbing, AbstractPPO1P):
    """Reach-avoid PPO with a per-step **policy_mask** (RAS phase-2 handover training).

    Identical to :class:`ReachAvoidPPO1P` EXCEPT that steps a frozen hand-off
    skill drove (``env._policy_mask == 0``) are excluded from the POLICY gradient
    while still training the VALUE (they stay in the buffer for value/returns/GAE
    -- so the reach-avoid value learns to reach a state the lander actually lands
    from). The mechanism, per the design:

      * value loss + returns + GAE + KL diagnostics: UNCHANGED (over all samples);
      * advantage normalization: over the UNMASKED entries only;
      * policy loss + entropy loss: masked means (``(m*·).sum()/m.sum()``).

    With an all-ones mask this reproduces :class:`ReachAvoidPPO1P` exactly (the
    graceful-degradation property: no handover -> the proven naive-RA update).

    Tensor path only (the RAS env is GPU-resident): forces the masked reach-avoid
    rollout buffer, which is the only buffer that carries ``policy_mask``.
    """

    _MODE = backups.REACH_AVOID

    def __init__(self, *args, **kwargs):
        # Force the masked reach-avoid buffer on the tensor path (it is the only
        # one exposing record_policy_mask + MaskedRolloutBufferSamples). Detect
        # the tensor env exactly as AbstractPPO does, BEFORE super() runs.
        _env = kwargs.get("env", args[1] if len(args) >= 2 else None)
        if (getattr(_env, "is_tensor_env", False)
                and kwargs.get("rollout_buffer_class") is None):
            kwargs["rollout_buffer_class"] = TensorReachAvoidMaskedRolloutBuffer
        super().__init__(*args, **kwargs)

    def train(self) -> None:
        """Stock ``PPO.train()`` with the three masked modifications above, then
        safety_sb3's KL-adaptive-LR tail (kept identical to ``AbstractPPO.train``
        so the recipe matches naive-RA)."""
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate (adaptive on safety_sb3, see AbstractPPO)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                # --- RAS mask: 1 = reach-controlled (trainable), 0 = hand-off ---
                m = rollout_data.policy_mask
                nu = m.sum().clamp_min(1.0)

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage over the UNMASKED entries only (the masked
                # steps' advantages never enter the policy loss; folding them into
                # the mean/std would bias the reach-policy update).
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    mask_bool = m > 0
                    if bool(mask_bool.any()):
                        adv_u = advantages[mask_bool]
                        advantages = (advantages - adv_u.mean()) / (adv_u.std() + 1e-8)
                    else:  # degenerate all-masked minibatch: avoid nan (loss is 0)
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss -- MASKED MEAN (sum over reach steps / #reach)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -(m * th.min(policy_loss_1, policy_loss_2)).sum() / nu

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf)
                # Value loss over ALL samples (the masked/lander steps TRAIN the
                # value -- that is the point of keeping them in the buffer).
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss -- masked identically to the policy loss.
                if entropy is None:
                    entropy_loss = -(m * (-log_prob)).sum() / nu
                else:
                    entropy_loss = -(m * entropy).sum() / nu
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # approx KL / early-stop over ALL samples (SB3 semantics kept).
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        # KL-adaptive-LR tail -- identical to AbstractPPO.train (adjust the LR for
        # the NEXT update from the KL just measured; keeps parity with naive-RA).
        if self.adaptive_lr and self.desired_kl is not None:
            kl = self.logger.name_to_value.get("train/approx_kl")
            if kl is not None and kl > 0.0:
                if kl > self.desired_kl * 2.0:
                    self._adaptive_lr = max(self.lr_min, self._adaptive_lr / self.adaptive_lr_factor)
                elif kl < self.desired_kl / 2.0:
                    self._adaptive_lr = min(self.lr_max, self._adaptive_lr * self.adaptive_lr_factor)
            self.logger.record("train/adaptive_lr", self._adaptive_lr)


class CumulativePPO1P(AbstractPPO1P):
    """Ordinary reward-maximizing PPO — ``V(s) = r + γ·V(s')``.

    Not a safety learner: the reward channel carries a reward, not a margin, and
    ``V >= 0`` means nothing. It is here so a *nominal* baseline runs through
    every line of the same code as the safety learners — the standard control for
    "is the safety operator doing the work, or is it just PPO?".
    """

    _MODE = backups.CUMULATIVE
