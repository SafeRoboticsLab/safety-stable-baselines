"""Reach-avoid PPO (increment 1 of the on-policy ISAACS family).

On-policy counterpart of :class:`ReachAvoidSAC`: the avoid-only
:class:`SafetyPPO` backup ``V = min(g, V')`` is replaced with the reach-avoid
Bellman backup

    V(s) = min( g(s), max( l(s), gamma * V(s') ) )

where ``g(s) >= 0`` defines the safe set (rides on the ``reward`` channel, as
in all Safety* algorithms here) and ``l(s) >= 0`` the target set (supplied by
the env via ``info["l_x"]`` — the same contract as :class:`ReachAvoidSAC`).
``V(s) > 0`` iff a policy exists that reaches ``{l >= 0}`` without ever
leaving ``{g >= 0}``.

Unlike the SAC family this is fully on-policy: it pairs with vectorized envs
at large ``n_envs`` (the GPU-parallel regime where off-policy ISAACS saturates;
validated on a 1280-env MuJoCo quadruped in ``unitree_rl_mjlab``).

References:
  K.-C. Hsu, V. Rubies-Royo, C. Tomlin, J. F. Fisac, "Safety and Liveness
  Guarantees through Reach-Avoid Reinforcement Learning", RSS 2021.
"""

from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from .safety_buffers import ReachAvoidRolloutBuffer
from .safety_ppo import SafetyPPO


class ReachAvoidPPO(SafetyPPO):
  """PPO with the reach-avoid Bellman backup (g = reward, l = info["l_x"])."""

  def __init__(self, *args, rollout_buffer_class=None, **kwargs):
    if rollout_buffer_class is None:
      rollout_buffer_class = ReachAvoidRolloutBuffer
    super().__init__(*args, rollout_buffer_class=rollout_buffer_class, **kwargs)

  def _setup_model(self) -> None:
    super()._setup_model()
    assert isinstance(self.rollout_buffer, ReachAvoidRolloutBuffer), (
      "ReachAvoidPPO requires ReachAvoidRolloutBuffer."
    )

  def collect_rollouts(
    self,
    env: VecEnv,
    callback: BaseCallback,
    rollout_buffer: RolloutBuffer,
    n_rollout_steps: int,
  ) -> bool:
    """SB3's OnPolicyAlgorithm.collect_rollouts with one insertion: the target
    margin ``info["l_x"]`` is written into the buffer alongside each step
    (rollout buffers do not receive ``infos`` in stock SB3)."""
    assert self._last_obs is not None, "No previous observation was provided"
    self.policy.set_training_mode(False)

    n_steps = 0
    rollout_buffer.reset()
    if self.use_sde:
      self.policy.reset_noise(env.num_envs)

    callback.on_rollout_start()

    while n_steps < n_rollout_steps:
      if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
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

      # --- reach-avoid insertion: capture l(s) at the position add() will use.
      assert isinstance(rollout_buffer, ReachAvoidRolloutBuffer)
      rollout_buffer.l_x[rollout_buffer.pos] = np.array(
        [float(info.get("l_x", 0.0)) for info in infos], dtype=np.float32
      )

      rollout_buffer.add(
        self._last_obs, actions, rewards,
        self._last_episode_starts, values, log_probs,
      )
      self._last_obs = new_obs
      self._last_episode_starts = dones

    with th.no_grad():
      values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

    rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

    callback.update_locals(locals())
    callback.on_rollout_end()
    return True
