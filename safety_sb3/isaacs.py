"""Two-player (adversarial) safety RL on SB3 — the SAC family.

Control actor (max-player) and disturbance actor (min-player) share one twin
critic over the full concatenated action ``Q(s, [a_ctrl, a_dstb])``.  Each actor
has its own entropy coefficient / target entropy (ctrl: ``-ctrl_dim``, dstb:
``-dstb_dim``).  Uses :class:`IsaacsPolicy` (two sub-space actors + full critic)
and the :class:`ReachAvoidReplayBuffer` (stores ``l(s)``).

Two classes, one per problem — the machinery is shared, only the backup differs
(see :mod:`safety_sb3.backups`):

* :class:`IsaacsSAC`  — two-player **avoid** game.  ISAACS proper
  (Hsu et al. 2022, eq. 7): no target set, anchor ``g``.
* :class:`GameplaySAC` — two-player **reach-avoid** game.  Gameplay Filters
  (Hsu et al. 2024, eq. 6a), which extends ISAACS to reach-avoid: anchor
  ``min(l, g)``.

Soft max-min value used in the critic target::

    V'  = min(Q1', Q2')  - α_ctrl·logπ_ctrl(s')  + α_dstb·logπ_dstb(s')
    y   = backups.target(mode, g, V', ...)       (avoid | reach-avoid)

(The ctrl entropy raises the value as a max-player bonus; the dstb entropy raises
the min as a min-player softening — same convention as the base ``SafetySAC``
adding ctrl entropy to ``next_q``; see the README TODO on this choice.)

.. warning::
   Before v0.2.0 ``IsaacsSAC`` was the *reach-avoid* game (now
   :class:`GameplaySAC`) and there was no two-player avoid class. See
   RELEASE_NOTES.md.
"""

from __future__ import annotations

import copy

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.utils import polyak_update

from safety_sb3 import backups
from safety_sb3.isaacs_policy import IsaacsPolicy
from safety_sb3.leaderboard import Leaderboard
from safety_sb3.reach_avoid_sac import ReachAvoidSAC


class GameplaySAC(ReachAvoidSAC):
  """Two-player REACH-AVOID game — Gameplay Filters (eq. 6a).

  Anchor ``min(l, g)``; needs a target margin ``l``. For a two-player *avoid*
  game (no target set) use :class:`IsaacsSAC`.
  """

  policy_aliases = {"MlpPolicy": IsaacsPolicy, "MultiInputPolicy": IsaacsPolicy}
  _MODE = backups.REACH_AVOID

  def __init__(
    self,
    policy,
    env,
    *,
    ctrl_action_dim: int | None = None,  # None only on load (restored from policy_kwargs)
    ctrl_update_period: int = 1,
    dstb_update_period: int = 1,
    # --- leaderboard (increment 3) ---
    use_leaderboard: bool = False,
    leaderboard_eval_env=None,
    save_top_k_ctrl: int = 5,
    save_top_k_dstb: int = 5,
    softmax_rationality: float = 3.0,
    leaderboard_freq: int = 10_000,
    n_eval_episodes: int = 10,
    leaderboard_dir: str = "isaacs_leaderboard",
    **kwargs,
  ) -> None:
    self.ctrl_action_dim = None if ctrl_action_dim is None else int(ctrl_action_dim)
    self.ctrl_update_period = int(ctrl_update_period)
    self.dstb_update_period = int(dstb_update_period)
    self.use_leaderboard = bool(use_leaderboard)
    self._lb_eval_env = leaderboard_eval_env
    self._lb_cfg = dict(
      save_top_k_ctrl=save_top_k_ctrl,
      save_top_k_dstb=save_top_k_dstb,
      softmax_rationality=softmax_rationality,
      model_dir=leaderboard_dir,
    )
    self.leaderboard_freq = int(leaderboard_freq)
    self.n_eval_episodes = int(n_eval_episodes)
    self._next_lb_step = self.leaderboard_freq
    self._rollout_dstb = None  # scratch dstb actor for the current rollout (None = current)
    policy_kwargs = dict(kwargs.pop("policy_kwargs", None) or {})
    if self.ctrl_action_dim is not None:
      policy_kwargs["ctrl_action_dim"] = self.ctrl_action_dim
    super().__init__(policy, env, policy_kwargs=policy_kwargs, **kwargs)

  def _setup_model(self) -> None:
    # On load, ctrl_action_dim is restored into policy_kwargs before _setup_model.
    if self.ctrl_action_dim is None:
      self.ctrl_action_dim = self.policy_kwargs.get("ctrl_action_dim")
    assert self.ctrl_action_dim is not None, (
      "ctrl_action_dim is required (the number of leading control action dims)."
    )
    super()._setup_model()  # sets ctrl entropy (target = -full_dim) and aliases
    # SB3 only defines ent_coef_tensor for FIXED ent_coef; default it for "auto".
    if not hasattr(self, "ent_coef_tensor"):
      self.ent_coef_tensor = None
    self.dstb_actor = self.policy.dstb_actor
    dstb_dim = self.policy.dstb_action_dim

    # Per-actor target entropies (SB3 set ctrl to -full_dim).
    self.target_entropy = float(-self.ctrl_action_dim)
    self.dstb_target_entropy = float(-dstb_dim)

    # Disturbance entropy coefficient (mirror SB3's ctrl setup).
    self.dstb_log_ent_coef = None
    self.dstb_ent_coef_optimizer = None
    self.dstb_ent_coef_tensor = None
    if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
      init_value = 1.0
      if "_" in self.ent_coef:
        init_value = float(self.ent_coef.split("_")[1])
      self.dstb_log_ent_coef = th.log(
        th.ones(1, device=self.device) * init_value
      ).requires_grad_(True)
      self.dstb_ent_coef_optimizer = th.optim.Adam(
        [self.dstb_log_ent_coef], lr=self.lr_schedule(1)
      )
    else:
      self.dstb_ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    if self.use_leaderboard:
      self._leaderboard = Leaderboard(seed=self.seed or 0, **self._lb_cfg)
      self._scratch_ctrl = copy.deepcopy(self.policy.actor)
      self._scratch_dstb = copy.deepcopy(self.policy.dstb_actor)

  # --- leaderboard: roll out against a sampled past disturbance ---
  def _resample_rollout_dstb(self) -> None:
    step = self._leaderboard.sample_dstb_step()
    if step is None:
      self._rollout_dstb = None  # use the current dstb actor
    else:
      self._leaderboard.load_actor(self._scratch_dstb, "dstb", step)
      self._scratch_dstb.set_training_mode(False)
      self._rollout_dstb = self._scratch_dstb

  def collect_rollouts(self, env, callback, train_freq, replay_buffer, action_noise=None, learning_starts=0, log_interval=None):
    if self.use_leaderboard and self.num_timesteps >= learning_starts:
      self._resample_rollout_dstb()
    out = super().collect_rollouts(
      env, callback, train_freq, replay_buffer,
      action_noise=action_noise, learning_starts=learning_starts, log_interval=log_interval,
    )
    if (
      self.use_leaderboard
      and self._lb_eval_env is not None
      and self.num_timesteps >= self._next_lb_step
    ):
      self._leaderboard_step()
      self._next_lb_step += self.leaderboard_freq
    return out

  _LB_MAX_STEPS = 400

  @th.no_grad()
  def _eval_pair(self, ctrl_actor, dstb_actor) -> float:
    """Reach-avoid success rate of ``ctrl`` vs ``dstb`` (None = dummy/no-dstb).

    Uses a parallel ``VecEnv`` eval env when available (one batch = ``num_envs``
    first-episodes, much cheaper for large leaderboards); otherwise falls back to
    the single-env loop.
    """
    env = self._lb_eval_env
    dn = self.policy.dstb_action_dim
    if hasattr(env, "num_envs") and hasattr(env, "step_async"):
      return self._eval_pair_vec(env, ctrl_actor, dstb_actor, dn)

    succ = 0
    for _ in range(self.n_eval_episodes):
      obs, _ = env.reset()
      done = safe = reached = False
      safe = True
      while not done:
        ot = th.as_tensor(np.asarray(obs), dtype=th.float32, device=self.device).reshape(1, -1)
        c = ctrl_actor(ot, deterministic=True).cpu().numpy()[0]
        d = (
          np.zeros(dn, np.float32)
          if dstb_actor is None
          else dstb_actor(ot, deterministic=True).cpu().numpy()[0]
        )
        scaled = np.concatenate([c, d]).astype(np.float32)  # [-1, 1]
        action = self.policy.unscale_action(scaled[None])[0]
        obs, g, term, trunc, info = env.step(action)
        done = bool(term or trunc)
        if g < 0:
          safe = False
        if float(info.get("l_x", -1.0)) >= 0:
          reached = True
      succ += int(safe and reached)
    return succ / max(self.n_eval_episodes, 1)

  def _eval_pair_vec(self, env, ctrl_actor, dstb_actor, dn) -> float:
    """Parallel reach-avoid success over ``num_envs`` * ``n_eval_episodes`` first-episodes."""
    n_env = env.num_envs
    total_succ, total = 0, 0
    for _ in range(max(1, self.n_eval_episodes)):
      obs = env.reset()
      ep_safe = np.ones(n_env, dtype=bool)
      ep_reached = np.zeros(n_env, dtype=bool)
      done_once = np.zeros(n_env, dtype=bool)
      for _ in range(self._LB_MAX_STEPS):
        ot = th.as_tensor(np.asarray(obs), dtype=th.float32, device=self.device)
        c = ctrl_actor(ot, deterministic=True).cpu().numpy()
        d = (
          np.zeros((n_env, dn), np.float32)
          if dstb_actor is None
          else dstb_actor(ot, deterministic=True).cpu().numpy()
        )
        env.step_async(np.concatenate([c, d], axis=1).astype(np.float32))
        obs, g, dones, infos = env.step_wait()
        active = ~done_once
        ep_safe &= ~(active & (np.asarray(g) < 0))
        lx = np.array([i.get("l_x", -1.0) for i in infos], dtype=np.float32)
        ep_reached |= active & (lx >= 0)
        done_once |= active & np.asarray(dones, dtype=bool)
        if done_once.all():
          break
      total_succ += int((ep_safe & ep_reached).sum())
      total += n_env
    return total_succ / max(total, 1)

  def _leaderboard_step(self) -> None:
    lb = self._leaderboard
    step = self.num_timesteps
    nc, nd, kc, kd = len(lb.ctrl_steps), len(lb.dstb_steps), lb.kc, lb.kd
    ctrl_cur, dstb_cur = self.policy.actor, self.policy.dstb_actor
    # current ctrl (row kc) vs each opponent
    for j in range(nd):
      lb.load_actor(self._scratch_dstb, "dstb", lb.dstb_steps[j])
      lb.set_score(kc, j, self._eval_pair(ctrl_cur, self._scratch_dstb))
    lb.set_score(kc, kd, self._eval_pair(ctrl_cur, dstb_cur))  # current dstb
    lb.set_score(kc, kd + 1, self._eval_pair(ctrl_cur, None))  # dummy
    # each saved ctrl vs current dstb (col kd)
    for i in range(nc):
      lb.load_actor(self._scratch_ctrl, "ctrl", lb.ctrl_steps[i])
      lb.set_score(i, kd, self._eval_pair(self._scratch_ctrl, dstb_cur))
    lb.prune(step, ctrl_cur, dstb_cur)
    self.logger.record("leaderboard/n_ctrl", len(lb.ctrl_steps))
    self.logger.record("leaderboard/n_dstb", len(lb.dstb_steps))

  # --- rollout: sample ctrl + dstb, concatenate ---
  def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
    if self.num_timesteps < learning_starts and not (
      self.use_sde and self.use_sde_at_warmup
    ):
      unscaled = np.array([self.action_space.sample() for _ in range(n_envs)])
      scaled = self.policy.scale_action(unscaled)
    else:
      obs_tensor, _ = self.policy.obs_to_tensor(self._last_obs)
      dstb_net = (
        self._rollout_dstb if self._rollout_dstb is not None else self.policy.dstb_actor
      )
      with th.no_grad():
        ctrl = self.policy.actor(obs_tensor, deterministic=False)
        dstb = dstb_net(obs_tensor, deterministic=False)
      scaled = th.cat([ctrl, dstb], dim=1).cpu().numpy()  # actors output in [-1, 1]

    if action_noise is not None:
      scaled = np.clip(scaled + action_noise(), -1, 1)
    buffer_action = scaled
    action = self.policy.unscale_action(scaled)
    return action, buffer_action

  # --- entropy-coefficient helper ---
  def _alpha(self, log_coef, optimizer, coef_tensor, target_entropy, log_prob):
    if log_coef is not None and optimizer is not None:
      coef = th.exp(log_coef.detach())
      loss = -(log_coef * (log_prob + target_entropy).detach()).mean()
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    else:
      coef = coef_tensor
    return coef

  def train(self, gradient_steps: int, batch_size: int) -> None:
    self.policy.set_training_mode(True)
    opts = [self.actor.optimizer, self.dstb_actor.optimizer, self.critic.optimizer]
    if self.ent_coef_optimizer is not None:
      opts.append(self.ent_coef_optimizer)
    if self.dstb_ent_coef_optimizer is not None:
      opts.append(self.dstb_ent_coef_optimizer)
    self._update_learning_rate(opts)

    ctrl_losses, dstb_losses, critic_losses = [], [], []
    for step in range(gradient_steps):
      rd = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

      ctrl_pi, ctrl_logp = self.actor.action_log_prob(rd.observations)
      ctrl_logp = ctrl_logp.reshape(-1, 1)
      dstb_pi, dstb_logp = self.dstb_actor.action_log_prob(rd.observations)
      dstb_logp = dstb_logp.reshape(-1, 1)

      ctrl_ent = self._alpha(
        self.log_ent_coef, self.ent_coef_optimizer, self.ent_coef_tensor,
        self.target_entropy, ctrl_logp,
      )
      dstb_ent = self._alpha(
        self.dstb_log_ent_coef, self.dstb_ent_coef_optimizer,
        self.dstb_ent_coef_tensor, self.dstb_target_entropy, dstb_logp,
      )

      # --- critic update (reach-avoid, soft max-min next value) ---
      with th.no_grad():
        next_ctrl, next_ctrl_logp = self.actor.action_log_prob(rd.next_observations)
        next_dstb, next_dstb_logp = self.dstb_actor.action_log_prob(
          rd.next_observations
        )
        next_action = th.cat([next_ctrl, next_dstb], dim=1)
        next_q = th.cat(self.critic_target(rd.next_observations, next_action), dim=1)
        next_q, _ = th.min(next_q, dim=1, keepdim=True)
        next_q = (
          next_q
          - ctrl_ent * next_ctrl_logp.reshape(-1, 1)
          + dstb_ent * next_dstb_logp.reshape(-1, 1)
        )
        # Backup for THIS class's problem -- see safety_sb3.backups.
        # GameplaySAC -> reach-avoid (eq. 6a); IsaacsSAC -> avoid (eq. 7).
        gs = rd.rewards
        not_done = 1.0 - rd.dones
        target_q = backups.target(
          self._MODE, gs, next_q, not_done, self.gamma,
          l=getattr(rd, "l_x", None), terminal_type=self.terminal_type)

      current_q = self.critic(rd.observations, rd.actions)
      critic_loss = 0.5 * sum(F.mse_loss(cq, target_q) for cq in current_q)
      critic_losses.append(critic_loss.item())
      self.critic.optimizer.zero_grad()
      critic_loss.backward()
      self.critic.optimizer.step()

      # --- ctrl actor update (MAX): maximize Q(s, [a_ctrl, detach a_dstb]) ---
      if step % self.ctrl_update_period == 0:
        with th.no_grad():
          dstb_aux = self.dstb_actor(rd.observations, deterministic=False)
        q_pi = th.cat(
          self.critic(rd.observations, th.cat([ctrl_pi, dstb_aux], dim=1)), dim=1
        )
        min_q, _ = th.min(q_pi, dim=1, keepdim=True)
        ctrl_loss = (ctrl_ent * ctrl_logp - min_q).mean()
        ctrl_losses.append(ctrl_loss.item())
        self.actor.optimizer.zero_grad()
        ctrl_loss.backward()
        self.actor.optimizer.step()

      # --- dstb actor update (MIN): minimize Q(s, [detach a_ctrl, a_dstb]) ---
      if step % self.dstb_update_period == 0:
        with th.no_grad():
          ctrl_aux = self.actor(rd.observations, deterministic=False)
        q_pi = th.cat(
          self.critic(rd.observations, th.cat([ctrl_aux, dstb_pi], dim=1)), dim=1
        )
        min_q, _ = th.min(q_pi, dim=1, keepdim=True)
        dstb_loss = (dstb_ent * dstb_logp + min_q).mean()
        dstb_losses.append(dstb_loss.item())
        self.dstb_actor.optimizer.zero_grad()
        dstb_loss.backward()
        self.dstb_actor.optimizer.step()

      if step % self.target_update_interval == 0:
        polyak_update(
          self.critic.parameters(), self.critic_target.parameters(), self.tau
        )
        polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

    self._n_updates += gradient_steps
    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
    self.logger.record("train/critic_loss", np.mean(critic_losses))
    if ctrl_losses:
      self.logger.record("train/ctrl_actor_loss", np.mean(ctrl_losses))
    if dstb_losses:
      self.logger.record("train/dstb_actor_loss", np.mean(dstb_losses))

  def _excluded_save_params(self):
    # Leaderboard runtime objects are rebuilt by _setup_model on load; the eval
    # env in particular is unpicklable (e.g. mjlab holds a mujoco MjSpec).
    return super()._excluded_save_params() + [
      "dstb_actor",
      "_lb_eval_env",
      "_leaderboard",
      "_scratch_ctrl",
      "_scratch_dstb",
      "_rollout_dstb",
    ]

  def _get_torch_save_params(self):
    state_dicts, others = super()._get_torch_save_params()
    state_dicts += ["dstb_actor.optimizer"]
    if self.dstb_ent_coef_optimizer is not None:
      state_dicts += ["dstb_ent_coef_optimizer"]
      others += ["dstb_log_ent_coef"]
    return state_dicts, others


class IsaacsSAC(GameplaySAC):
  """Two-player AVOID game — ISAACS proper (Hsu et al. 2022, eq. 7).

  ``V(s) = (1-γ)·g + γ·max_ctrl min_dstb min(g, V')``: the robust-invariance
  value under a worst-case disturbance. No target set, no ``l`` — the paper has
  neither.

  This is the class to use for an adversarial *avoid* task (e.g. "stay standing
  against a worst-case force"). Do NOT emulate it by giving
  :class:`GameplaySAC` a degenerate ``l``: no ``l`` reduces the reach-avoid
  operator to avoid (:mod:`safety_sb3.backups` proves the two conditions are
  contradictory), and the constant-``l`` trick that appeared to work before
  v0.2.0 relied on a bug in the reach-avoid anchor.

  ``l`` is ignored if present, so an ``l``-carrying replay buffer is harmless.
  """

  _MODE = backups.AVOID
