import numpy as np
import torch as th
import torch.nn.functional as F

from stable_baselines3.sac.sac import SAC
from stable_baselines3.common.type_aliases import RolloutReturn, TrainFreq
from stable_baselines3.common.utils import polyak_update, should_collect_more_steps


class SafetySAC(SAC):
    """We subclass SAC to reuse the actor, entropy regularization, replay buffer, etc.

    GPU-resident path: pass a ``TensorVecEnv`` (detected via ``is_tensor_env``,
    optionally auto-wrapped in ``TensorVecNormalize`` with ``normalize_obs=True``)
    and collection + replay stay on device end-to-end
    (:class:`safety_sb3.tensor_replay.TensorReplayBuffer`); ``train()`` is
    unchanged — the tensor buffer returns the same named tuples. NOTE on the
    tensor path set ``gradient_steps`` explicitly (>=1): the SB3 convention
    ``gradient_steps=-1`` ("as many as env steps") would mean num_envs updates
    per vector step.
    """

    _tensor_store_l = False  # ReachAvoidSAC flips this (buffer stores l_x)

    def __init__(self, *args, normalize_obs: bool = False, **kwargs):
        from .safety_ppo import _guard_and_normalize_env
        if "env" in kwargs:
            kwargs["env"] = _guard_and_normalize_env(kwargs["env"], normalize_obs)
        elif len(args) >= 2:
            args = list(args)
            args[1] = _guard_and_normalize_env(args[1], normalize_obs)
        _env = kwargs.get("env", args[1] if len(args) >= 2 else None)
        self._tensor_path = bool(getattr(_env, "is_tensor_env", False))
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        if not self._tensor_path:
            super()._setup_model()
            return
        from .tensor_replay import TensorReplayBuffer
        # Keep the numpy buffer SB3 allocates in _setup_model negligible, then
        # replace it with the device-resident buffer at the real size.
        real_size = self.buffer_size
        self.buffer_size = self.env.num_envs
        super()._setup_model()
        self.buffer_size = real_size
        self.replay_buffer = TensorReplayBuffer(
            real_size,
            obs_dim=int(np.prod(self.observation_space.shape)),
            act_dim=int(np.prod(self.action_space.shape)),
            n_envs=self.env.num_envs,
            device=str(self.device),
            store_l=self._tensor_store_l,
        )

    # --- tensor collection ---------------------------------------------------

    def collect_rollouts(self, env, callback, train_freq: TrainFreq,
                         replay_buffer, action_noise=None, learning_starts=0,
                         log_interval=None) -> RolloutReturn:
        if self._tensor_path:
            return self._collect_rollouts_tensor(
                env, callback, train_freq, replay_buffer,
                learning_starts=learning_starts, log_interval=log_interval)
        return super().collect_rollouts(
            env, callback, train_freq, replay_buffer, action_noise=action_noise,
            learning_starts=learning_starts, log_interval=log_interval)

    def _collect_rollouts_tensor(self, env, callback, train_freq: TrainFreq,
                                 replay_buffer, learning_starts=0,
                                 log_interval=None) -> RolloutReturn:
        """Torch-native off-policy collection: actor forward, env.step_tensor,
        replay add — all on device. Mirrors OffPolicyAlgorithm.collect_rollouts
        step/episode accounting (one vector step counts 1 toward train_freq)."""
        self.policy.set_training_mode(False)
        callback.on_rollout_start()
        dev = env.device

        obs = self._last_obs
        if not th.is_tensor(obs):  # first call after _setup_learn
            obs = th.as_tensor(np.asarray(obs), dtype=th.float32, device=dev)

        low = th.as_tensor(self.action_space.low, dtype=th.float32, device=dev)
        high = th.as_tensor(self.action_space.high, dtype=th.float32, device=dev)

        if getattr(self, "_t_ep_ret", None) is None \
                or self._t_ep_ret.shape[0] != env.num_envs:
            self._t_ep_ret = th.zeros(env.num_envs, device=dev)
            self._t_ep_len = th.zeros(env.num_envs, device=dev)
            self._tensor_last_dump = 0
        fin_ret, fin_len = [], []

        num_collected_steps, num_collected_episodes = 0, 0
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps,
                                        num_collected_episodes):
            if self.num_timesteps < learning_starts:
                actions = low + (high - low) * th.rand(
                    (env.num_envs, low.shape[0]), device=dev)
            else:
                with th.no_grad():
                    actions = self.actor(obs, deterministic=False)
            actions = th.clamp(actions, low, high)

            new_obs, g, dones, timeouts, l_x = env.step_tensor(actions)
            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            callback.update_locals(locals())
            if not callback.on_step():
                continue_training = False
                break

            replay_buffer.add_batch(
                obs, new_obs, actions, g, dones, timeouts,
                l_x=l_x if self._tensor_store_l else None)

            self._t_ep_ret += g
            self._t_ep_len += 1.0
            n_done = int(dones.sum())
            if n_done:
                d = dones.bool()
                fin_ret.append(self._t_ep_ret[d])
                fin_len.append(self._t_ep_len[d])
                self._t_ep_ret = th.where(d, th.zeros_like(self._t_ep_ret),
                                          self._t_ep_ret)
                self._t_ep_len = th.where(d, th.zeros_like(self._t_ep_len),
                                          self._t_ep_len)
                num_collected_episodes += n_done
                self._episode_num += n_done
            obs = new_obs

        self._last_obs = obs
        if fin_ret:
            self.logger.record("rollout/ep_rew_mean", float(th.cat(fin_ret).mean()))
            self.logger.record("rollout/ep_len_mean", float(th.cat(fin_len).mean()))
        # periodic dump (numpy path dumps on episode boundaries; with thousands
        # of parallel envs use an env-step cadence instead)
        if self.num_timesteps - self._tensor_last_dump >= 50_000:
            self._tensor_last_dump = self.num_timesteps
            for k, v in (env.metrics() or {}).items():
                self.logger.record(f"env/{k}", float(v))
            self._dump_logs()

        callback.on_rollout_end()
        return RolloutReturn(num_collected_steps * env.num_envs,
                             num_collected_episodes, continue_training)

    def train(self, gradient_steps: int, batch_size: int) -> None:
        """Largely follows the original SAC train method from stable_baselines3.
        We use the safety Bellman backup.
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

                # Safety Bellman: td error + entropy term
                gs = replay_data.rewards  # immediate safety margin values g(s) from env rewards
                not_done = 1.0 - replay_data.dones
                v_to_go = th.minimum(gs, next_q_values)
                target_q_values = (
                    1.0 - self.gamma * not_done
                ) * gs + self.gamma * not_done * v_to_go  # ensures that the full gs is returned at terminal states

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

            # Compute actor loss
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
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
