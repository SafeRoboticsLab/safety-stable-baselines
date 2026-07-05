"""Safety PPO with the Safety Bellman backup (reward channel = g(s)).

The return computation lives in :class:`SafetyRolloutBuffer`
(``V = min(g, V')``).  This module ports the pieces of the *validated* rsl_rl
``SafetyPPO`` that live OUTSIDE the return computation and turn out to be what
lets it learn hard robot tasks where stock-SB3-PPO-on-g stalls:

* **No timeout bootstrapping** (``bootstrap_on_timeout=False``, the default).
  Stock PPO adds ``gamma * V(terminal)`` to the reward at truncations
  (``on_policy_algorithm.collect_rollouts``).  For every Safety* algorithm the
  reward IS the physical safety margin ``g(s)`` — an absolute quantity — so
  adding a value estimate destroys its semantics and silently corrupts the
  backup the moment episodes start reaching the time limit.  rsl_rl's SafetyPPO
  overrides ``process_env_step`` for exactly this reason; we gate it here.

* **KL-adaptive learning rate** (``adaptive_lr=True`` with ``desired_kl``).
  rsl_rl raises the LR when the update KL sits well below target and lowers it
  when above.  SB3's ``target_kl`` only *early-stops*; it never adjusts the LR,
  so on the low-KL / weak-gradient regime typical of the sparse safety signal
  the policy barely moves.  Enabling this recovers rsl_rl's behavior.

Observation normalization is orthogonal and handled the SB3 way — wrap the env
in ``VecNormalize`` — which mirrors rsl_rl's built-in running obs normalizer.

All knobs are constructor arguments so downstream projects can configure the
algorithm without subclassing.  Defaults are chosen so ``SafetyPPO`` behaves
correctly for safety out of the box (``bootstrap_on_timeout=False``); the LR
schedule stays stock unless ``adaptive_lr`` is set.
"""

from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor, update_learning_rate
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.ppo.ppo import PPO

from .safety_buffers import SafetyRolloutBuffer


def _guard_and_normalize_env(env, normalize_obs: bool):
    """Reject reward-normalizing wrappers and optionally add obs normalization.

    For every Safety* algorithm the reward IS the physical safety margin g(s),
    an absolute quantity. Normalizing it (VecNormalize's ``norm_reward=True``
    default) rescales g by a running std and destroys the Safety Bellman
    backup — the same class of corruption as timeout value-bootstrapping. So
    we hard-error on it. Observation normalization, by contrast, is REQUIRED
    for hard robot tasks (rsl_rl normalizes obs on both actor and critic);
    ``normalize_obs=True`` wraps the env in ``VecNormalize(norm_obs=True,
    norm_reward=False)`` for you (mirrors rsl_rl's built-in running normalizer).
    """
    if isinstance(env, VecNormalize) and env.norm_reward:
        raise ValueError(
            "Safety* algorithms require an UN-normalized reward: the reward is "
            "the physical safety margin g(s), and VecNormalize(norm_reward=True) "
            "rescales it, corrupting the Safety Bellman backup. Re-wrap with "
            "VecNormalize(env, norm_obs=True, norm_reward=False), or pass "
            "normalize_obs=True to the algorithm and drop your VecNormalize."
        )
    if normalize_obs and not isinstance(env, VecNormalize):
        env = VecNormalize(env, norm_obs=True, norm_reward=False)
    return env


class SafetyPPO(PPO):
    """PPO with the Safety Bellman backup, the rsl_rl-parity training recipe.

    :param bootstrap_on_timeout: if False (default) skip PPO's timeout value
        bootstrapping — correct for Safety* algorithms whose reward is g(s).
    :param normalize_obs: wrap the env in VecNormalize(norm_obs=True,
        norm_reward=False) — obs normalization is needed to match rsl_rl on
        hard robot tasks; reward normalization is refused (corrupts g).
    :param adaptive_lr: enable rsl_rl-style KL-adaptive learning rate.
    :param desired_kl: target KL for the adaptive LR controller.
    :param lr_bounds: (min, max) bounds for the adaptive LR.
    :param adaptive_lr_factor: multiplicative step for the adaptive LR.
    """

    def __init__(
        self,
        *args,
        learning_rate=3e-4,
        rollout_buffer_class=None,
        rollout_buffer_kwargs=None,
        bootstrap_on_timeout: bool = False,
        normalize_obs: bool = False,
        adaptive_lr: bool = False,
        desired_kl: float | None = 0.01,
        lr_bounds: tuple[float, float] = (1e-5, 1e-2),
        adaptive_lr_factor: float = 1.5,
        **kwargs,
    ):
        # Default to SafetyRolloutBuffer unless the caller overrides it.
        if rollout_buffer_class is None:
            rollout_buffer_class = SafetyRolloutBuffer

        # Guard the reward-normalization footgun + optionally add obs norm.
        # env is the 2nd positional PPO arg (policy, env, ...) or a kwarg.
        args = list(args)
        if "env" in kwargs:
            kwargs["env"] = _guard_and_normalize_env(kwargs["env"], normalize_obs)
        elif len(args) >= 2:
            args[1] = _guard_and_normalize_env(args[1], normalize_obs)
        args = tuple(args)

        self.bootstrap_on_timeout = bool(bootstrap_on_timeout)
        self.adaptive_lr = bool(adaptive_lr)
        self.desired_kl = desired_kl
        self.lr_min, self.lr_max = float(lr_bounds[0]), float(lr_bounds[1])
        self.adaptive_lr_factor = float(adaptive_lr_factor)
        if self.adaptive_lr:
            if callable(learning_rate):
                raise ValueError(
                    "adaptive_lr=True requires a float initial learning_rate, "
                    f"got a schedule: {learning_rate!r}"
                )
            # Mutable current LR, adjusted each update from the measured KL.
            self._adaptive_lr = float(learning_rate)

        super().__init__(
            *args,
            learning_rate=learning_rate,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            **kwargs,
        )

    def _setup_model(self) -> None:
        # Builds policy, optimizer, and rollout buffer.
        super()._setup_model()
        assert isinstance(self.rollout_buffer, SafetyRolloutBuffer), (
            "SafetyPPO requires SafetyRolloutBuffer. "
            "Pass `rollout_buffer_class=SafetyRolloutBuffer` (and "
            "rollout_buffer_kwargs if needed)."
        )

    # ---------------------------------------------------------- adaptive LR
    def _update_learning_rate(self, optimizers) -> None:
        """Set the optimizer LR.  With ``adaptive_lr`` use the KL-controlled
        value instead of SB3's progress schedule (called at each ``train()``)."""
        if not self.adaptive_lr:
            super()._update_learning_rate(optimizers)
            return
        self.logger.record("train/learning_rate", self._adaptive_lr)
        update_learning_rate(optimizers, self._adaptive_lr)

    def train(self) -> None:
        super().train()
        # Adjust the LR for the NEXT update from the KL just measured, mirroring
        # rsl_rl: KL >> target -> shrink LR; KL << target -> grow LR.
        if self.adaptive_lr and self.desired_kl is not None:
            kl = self.logger.name_to_value.get("train/approx_kl")
            if kl is not None and kl > 0.0:
                if kl > self.desired_kl * 2.0:
                    self._adaptive_lr = max(
                        self.lr_min, self._adaptive_lr / self.adaptive_lr_factor
                    )
                elif kl < self.desired_kl / 2.0:
                    self._adaptive_lr = min(
                        self.lr_max, self._adaptive_lr * self.adaptive_lr_factor
                    )
            self.logger.record("train/adaptive_lr", self._adaptive_lr)

    # ------------------------------------------------------- rollout hooks
    def _record_step_extras(self, rollout_buffer: RolloutBuffer, infos: list) -> None:
        """Hook: stash per-step extras (e.g. ``l_x``) at ``rollout_buffer.pos``
        just before ``add()``.  No-op for avoid-only SafetyPPO; reach-avoid and
        friends override it.  (Rollout buffers do not receive ``infos`` in
        stock SB3, so extras must be captured here.)"""

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """SB3 ``OnPolicyAlgorithm.collect_rollouts`` with two safety changes:
        the timeout value-bootstrap is gated by ``self.bootstrap_on_timeout``
        (off by default — the reward is the physical margin g(s)), and a
        ``_record_step_extras`` hook runs before each ``add()``."""
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

            # Timeout bootstrapping — DISABLED by default for safety (see class
            # docstring).  Stock PPO would corrupt g(s) here.
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

            self._record_step_extras(rollout_buffer, infos)

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
