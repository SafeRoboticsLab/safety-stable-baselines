"""Two-player (adversarial) SAC — the **2P** half of the SAC family.

    AbstractSAC                 (sac_base.py)
    └─ AbstractSAC2P            two actors, two entropy temps, ONE joint critic
        ├─ SafetySAC2P          _MODE = AVOID          (ISAACS proper)
        └─ ReachAvoidSAC2P      _MODE = REACH_AVOID    (Gameplay Filters)

Control actor (max-player) and disturbance actor (min-player) share one twin
critic over the full concatenated action ``Q(s, [a_ctrl, a_dstb])``. Each actor
has its own entropy coefficient / target entropy (ctrl: ``-ctrl_dim``, dstb:
``-dstb_dim``). Uses :class:`~safety_sb3.policies.TwoPlayerSACPolicy` (two
sub-space actors + one full critic).

PURE Hamilton-Jacobi value in the critic target — NO entropy::

    V'  = min(Q1', Q2')                          (no soft term)
    y   = backups.target(mode, g, V', ...)       (avoid | reach-avoid)

The per-actor temperatures α_ctrl, α_dstb appear ONLY in the ctrl/dstb actor
losses (exploration; annealed away over training), never in the critic target.
This is ISAACS eq. 8a exactly (Hsu et al. 2023, ``entropy_motives=0``), and it is
what both reference codebases do (base_block.py adds entropy to the target only
``if self.mode == 'performance'``). A soft term in the target would tax the game
value whose zero level set is the online safety certificate, warping {V>=0}.
(An earlier version of this file put ``- α_ctrl·logπ_ctrl + α_dstb·logπ_dstb`` in
V' and mislabelled it "as ISAACS formulates it"; that was a port error, now
corrected — see the note at the target computation.)

**This is the minimax game as ISAACS formulates it**, and it is NOT the same
object as :mod:`safety_sb3.ppo_2p` even though both hold two actors. Because
SAC's critic takes the action, this one ``Q`` *is* the game value and both actors
differentiate through it; nothing needs scheduling, and replay makes the learner
indifferent to whose policy produced the data. The on-policy two-player learner
has two state-only ``V(s)`` nets, two rollout buffers and a phase machine, and is
an alternating best-response approximation. See :mod:`safety_sb3.ppo_2p` for the
full comparison and for why there is deliberately no shared
``AbstractTwoPlayer`` base.
"""

from __future__ import annotations

import copy

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.utils import polyak_update, update_learning_rate

from . import backups
from .leaderboard import Leaderboard, LeagueEvaluator
from .policies import TwoPlayerSACPolicy
from .sac_base import AbstractSAC


class AbstractSAC2P(AbstractSAC):
  """Two-player off-policy game; the backup is chosen by ``_MODE``."""

  policy_aliases = {"MlpPolicy": TwoPlayerSACPolicy,
                    "MultiInputPolicy": TwoPlayerSACPolicy}

  def __init__(
    self,
    policy,
    env,
    *,
    ctrl_action_dim: int | None = None,  # None only on load (restored from policy_kwargs)
    ctrl_update_period: int = 1,
    dstb_update_period: int = 1,
    # --- per-network / per-actor learning rates ---
    # Each is None -> falls back to the shared ``learning_rate``.
    # ``dstb_learning_rate`` is the dstb ACTOR lr; ``ent_coef_lr`` /
    # ``dstb_ent_coef_lr`` are the ctrl / dstb ENTROPY (alpha) optimizer lrs.
    # The ctrl actor always uses the shared ``learning_rate``.
    critic_learning_rate: float | None = None,
    dstb_learning_rate: float | None = None,
    ent_coef_lr: float | None = None,
    dstb_ent_coef_lr: float | None = None,
    # --- optional StepLR decay for the ctrl/dstb/critic optimizers ---
    # OFF by default (constant lr). When on, each lr decays by ``lr_decay`` every
    # ``lr_period`` env-steps toward ``lr_end`` (reference StepLR). The entropy
    # (alpha) lrs stay constant -- an alpha-lr schedule is a follow-up.
    lr_schedule: bool = False,
    lr_period: int = 1_000_000,
    lr_decay: float = 0.1,
    lr_end: float = 0.0,
    # --- league ---
    use_leaderboard: bool = False,
    leaderboard_eval_env=None,
    save_top_k_ctrl: int = 5,
    save_top_k_dstb: int = 5,
    softmax_rationality: float = 3.0,
    leaderboard_freq: int = 10_000,
    n_eval_episodes: int = 10,
    leaderboard_dir: str = "sac_2p_leaderboard",
    **kwargs,
  ) -> None:
    self.ctrl_action_dim = None if ctrl_action_dim is None else int(ctrl_action_dim)
    self.ctrl_update_period = int(ctrl_update_period)
    self.dstb_update_period = int(dstb_update_period)
    # Raw per-network lr args (None = fall back to shared learning_rate); the
    # numeric lrs are resolved in _setup_model once self.lr_schedule exists.
    self._critic_lr_arg = critic_learning_rate
    self._dstb_lr_arg = dstb_learning_rate
    self._ent_coef_lr_arg = ent_coef_lr
    self._dstb_ent_coef_lr_arg = dstb_ent_coef_lr
    # StepLR decay config for the ctrl/dstb/critic optimizers.
    self._lr_schedule_on = bool(lr_schedule)
    self._lr_period = max(1, int(lr_period))
    self._lr_decay = float(lr_decay)
    self._lr_end = float(lr_end)
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
    # Resolve the per-network / per-actor lrs now that self.lr_schedule exists.
    # None -> shared learning_rate. ``_ctrl_lr`` is the shared value; the ctrl
    # entropy lr (``_ent_coef_lr``) is read by AbstractSAC._reset_entropy_temp.
    shared_lr = float(self.lr_schedule(1))
    self._ctrl_lr = shared_lr
    self._critic_lr = shared_lr if self._critic_lr_arg is None else float(self._critic_lr_arg)
    self._dstb_lr = shared_lr if self._dstb_lr_arg is None else float(self._dstb_lr_arg)
    self._ent_coef_lr = shared_lr if self._ent_coef_lr_arg is None else float(self._ent_coef_lr_arg)
    self._dstb_ent_coef_lr = (
      shared_lr if self._dstb_ent_coef_lr_arg is None else float(self._dstb_ent_coef_lr_arg))
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
        [self.dstb_log_ent_coef], lr=self._dstb_ent_coef_lr
      )
    else:
      self.dstb_ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    # Snapshot the dstb init entropy temperature so a gamma jump can reset it
    # too (the ctrl init is snapshotted in AbstractSAC._setup_entropy_bounds).
    self._init_dstb_log_ent_coef = (
      None if self.dstb_log_ent_coef is None
      else self.dstb_log_ent_coef.detach().clone())

    # TwoPlayerSACPolicy._build built actor/dstb_actor/critic at the shared lr,
    # and SB3 built the ctrl ent_coef optimizer at the shared lr; stamp the
    # dedicated lrs so they hold even before the first train() step (the dstb ent
    # optimizer was already built at its dedicated lr above). The ctrl actor
    # keeps the shared lr.
    update_learning_rate(self.critic.optimizer, self._critic_lr)
    update_learning_rate(self.dstb_actor.optimizer, self._dstb_lr)
    if self.ent_coef_optimizer is not None:
      update_learning_rate(self.ent_coef_optimizer, self._ent_coef_lr)

    if self.use_leaderboard:
      self._leaderboard = Leaderboard(seed=self.seed or 0, **self._lb_cfg)
      self._scratch_ctrl = copy.deepcopy(self.policy.actor)
      self._scratch_dstb = copy.deepcopy(self.policy.dstb_actor)
      # The league's estimator. Everything about scoring lives in
      # leaderboard.py; this class only decides WHEN to refresh the board.
      self._league_eval = LeagueEvaluator(
        env=self._lb_eval_env,
        device=self.device,
        n_eval_episodes=self.n_eval_episodes,
        dstb_action_dim=self.policy.dstb_action_dim,
        unscale_action=self.policy.unscale_action,
        success=self._league_success,
        obs_normalizer=lambda: (self.env if hasattr(self.env, "normalize_obs")
                                else None),
      )

  # --- the league's win condition -------------------------------------------
  # The avoid game has no target set, so survival IS the win. ReachAvoidSAC2P
  # overrides this with the extra reach requirement. Passing the rule to the
  # evaluator (rather than a mode flag) is what lets both modes share one
  # scorer without either of them carrying the other's machinery.

  @staticmethod
  def _league_success(safe, reached):
    return safe

  # --- league: roll out against a sampled past disturbance ---
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

  def _leaderboard_step(self) -> None:
    """Refresh the live frontier of the board, then admit/evict.

    Only the current ctrl's row and the current dstb's column are recomputed —
    archived-vs-archived cells are carried over by :meth:`Leaderboard.prune`, so
    each refresh costs ``nc + nd + 2`` evaluations rather than a full matrix.
    """
    lb = self._leaderboard
    ev = self._league_eval
    step = self.num_timesteps
    nc, nd, kc, kd = len(lb.ctrl_steps), len(lb.dstb_steps), lb.kc, lb.kd
    ctrl_cur, dstb_cur = self.policy.actor, self.policy.dstb_actor
    # current ctrl (row kc) vs each opponent
    for j in range(nd):
      lb.load_actor(self._scratch_dstb, "dstb", lb.dstb_steps[j])
      lb.set_score(kc, j, ev.score(ctrl_cur, self._scratch_dstb))
    lb.set_score(kc, kd, ev.score(ctrl_cur, dstb_cur))  # current dstb
    lb.set_score(kc, kd + 1, ev.score(ctrl_cur, None))  # dummy
    # each saved ctrl vs current dstb (col kd)
    for i in range(nc):
      lb.load_actor(self._scratch_ctrl, "ctrl", lb.ctrl_steps[i])
      lb.set_score(i, kd, ev.score(self._scratch_ctrl, dstb_cur))
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

  # --- tensor-path rollout: same ctrl+dstb concatenation, on device ---
  def _tensor_policy_actions(self, obs: th.Tensor) -> th.Tensor:
    """Two-player tensor collect: sample ctrl + dstb and concatenate to the
    env's ``ctrl_dim + dstb_dim`` action. The GPU-resident analog of
    :meth:`_sample_action` (numpy path); without it the base single-player
    collect samples ctrl only and mismatches the env's action space. The dstb
    is the league-sampled opponent when active, else the current dstb actor
    (mirrors the numpy path). Actors output in [-1, 1] and the env action space
    is [-1, 1], so no unscaling is needed — the collect loop clamps to bounds.
    """
    dstb_net = (
      self._rollout_dstb if getattr(self, "_rollout_dstb", None) is not None
      else self.policy.dstb_actor
    )
    with th.no_grad():
      ctrl = self.policy.actor(obs, deterministic=False)
      dstb = dstb_net(obs, deterministic=False)
    return th.cat([ctrl, dstb], dim=1)

  # --- entropy-coefficient helper ---
  def _alpha(self, log_coef, optimizer, coef_tensor, target_entropy, log_prob):
    if log_coef is not None and optimizer is not None:
      coef = th.exp(log_coef.detach())
      loss = -(log_coef * (log_prob + target_entropy).detach()).mean()
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      # min_alpha/max_alpha floor/ceiling -- both actors share bounds.
      if self._log_min_alpha is not None or self._log_max_alpha is not None:
        with th.no_grad():
          log_coef.clamp_(min=self._log_min_alpha, max=self._log_max_alpha)
    else:
      coef = coef_tensor
    return coef

  def _reset_entropy_temp(self) -> None:
    """A gamma jump resets BOTH actors' entropy temperature (reference resets
    ctrl AND dstb alpha on every jump)."""
    super()._reset_entropy_temp()  # ctrl
    dlec = getattr(self, "dstb_log_ent_coef", None)
    if dlec is None or getattr(self, "_init_dstb_log_ent_coef", None) is None:
      return
    with th.no_grad():
      dlec.data.copy_(self._init_dstb_log_ent_coef)
    if getattr(self, "dstb_ent_coef_optimizer", None) is not None:
      dstb_lr = getattr(self, "_dstb_ent_coef_lr", None)
      if dstb_lr is None:
        dstb_lr = float(self.lr_schedule(1))
      self.dstb_ent_coef_optimizer = th.optim.Adam([dlec], lr=dstb_lr)

  # --- per-network learning-rate control ------------------------------------
  def _steplr_value(self, base_lr: float) -> float:
    """Reference StepLR: ``base * lr_decay ** (env_steps // lr_period)``,
    floored at ``lr_end``."""
    num_decay = int(self.num_timesteps) // self._lr_period
    return max(base_lr * (self._lr_decay ** num_decay), self._lr_end)

  def _dedicated_lr_specs(self):
    """Optimizers with their own lr, as ``{id(opt): (base_lr, steplr_on)}``.

    The critic and dstb actor always carry a dedicated (possibly StepLR-decayed)
    lr. The ctrl actor is only intercepted when StepLR is on -- otherwise it
    falls through to SB3's shared-schedule ``_update_learning_rate`` (unchanged
    behaviour). The entropy (alpha) optimizers hold a constant dedicated lr; an
    alpha-lr StepLR is a follow-up."""
    specs = {
      id(self.critic.optimizer): (self._critic_lr, self._lr_schedule_on),
      id(self.dstb_actor.optimizer): (self._dstb_lr, self._lr_schedule_on),
    }
    if self._lr_schedule_on:
      specs[id(self.actor.optimizer)] = (self._ctrl_lr, True)
    if self.ent_coef_optimizer is not None:
      specs[id(self.ent_coef_optimizer)] = (self._ent_coef_lr, False)
    if self.dstb_ent_coef_optimizer is not None:
      specs[id(self.dstb_ent_coef_optimizer)] = (self._dstb_ent_coef_lr, False)
    return specs

  def _update_learning_rate(self, optimizers) -> None:
    """Stop the blanket SB3 overwrite from collapsing every optimizer onto the
    single shared ``lr_schedule``. Optimizers with a dedicated lr keep it (or its
    StepLR-decayed value); anything else routes through SB3's default update so
    the shared ctrl-actor schedule is preserved."""
    if not isinstance(optimizers, list):
      optimizers = [optimizers]
    specs = self._dedicated_lr_specs()
    shared = []
    for opt in optimizers:
      spec = specs.get(id(opt))
      if spec is None:
        shared.append(opt)
        continue
      base_lr, steplr_on = spec
      update_learning_rate(opt, self._steplr_value(base_lr) if steplr_on else base_lr)
    if shared:
      super()._update_learning_rate(shared)

  def train(self, gradient_steps: int, batch_size: int) -> None:
    self.policy.set_training_mode(True)
    opts = [self.actor.optimizer, self.dstb_actor.optimizer, self.critic.optimizer]
    if self.ent_coef_optimizer is not None:
      opts.append(self.ent_coef_optimizer)
    if self.dstb_ent_coef_optimizer is not None:
      opts.append(self.dstb_ent_coef_optimizer)
    self._update_learning_rate(opts)

    # Accumulate on device; one sync per train(). See AbstractSAC1P.train for
    # the measurement and why this is a prerequisite for CUDA-graph capture.
    _z = th.zeros((), device=self.device)
    critic_loss_sum, ctrl_loss_sum, dstb_loss_sum = _z.clone(), _z.clone(), _z.clone()
    critic_loss_max = th.full((), -float("inf"), device=self.device)
    n_critic = n_ctrl = n_dstb = 0
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

      # --- critic update (soft max-min next value) ---
      with th.no_grad():
        # log-probs of the next actions are intentionally discarded: entropy is
        # not in the safety critic target (see the pure-HJ backup below).
        next_ctrl, _ = self.actor.action_log_prob(rd.next_observations)
        next_dstb, _ = self.dstb_actor.action_log_prob(rd.next_observations)
        next_action = th.cat([next_ctrl, next_dstb], dim=1)
        next_q = th.cat(self.critic_target(rd.next_observations, next_action), dim=1)
        next_q, _ = th.min(next_q, dim=1, keepdim=True)
        # PURE Isaacs/HJ backup -- NO entropy in the critic target. ISAACS
        # (Hsu et al. 2023) passes entropy_motives=0 for the safety value
        # (eq. 8a: y = (1-g)g' + g*min{g', Q'}); the per-actor temperatures live
        # ONLY in the ctrl/dstb ACTOR losses below (eq. 8b). AbstractSAC2P is
        # never CUMULATIVE (there is no two-player cumulative game -- see the MAP
        # law in registry.algo_name), so the soft term is dropped
        # unconditionally, not gated. The previous
        # `- ctrl_ent*logp + dstb_ent*logp` taxed the game value whose zero
        # level set is the online safety certificate.
        # Backup for THIS learner's mode -- see safety_sb3.backups.
        # ReachAvoidSAC2P -> reach-avoid (eq. 6a); SafetySAC2P -> avoid (eq. 7).
        target_q = self._bellman_target(rd, next_q)

      current_q = self.critic(rd.observations, rd.actions)
      critic_loss = 0.5 * sum(F.mse_loss(cq, target_q) for cq in current_q)
      _cl = critic_loss.detach()
      critic_loss_sum += _cl
      critic_loss_max = th.maximum(critic_loss_max, _cl)
      n_critic += 1
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
        ctrl_loss_sum += ctrl_loss.detach(); n_ctrl += 1
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
        dstb_loss_sum += dstb_loss.detach(); n_dstb += 1
        self.dstb_actor.optimizer.zero_grad()
        dstb_loss.backward()
        self.dstb_actor.optimizer.step()

      if step % self.target_update_interval == 0:
        self._polyak(
          self.critic.parameters(), self.critic_target.parameters(), self.tau
        )
        self._polyak(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

    self._n_updates += gradient_steps
    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
    # record_mean + window max -- see AbstractSAC._record_window_max.
    if n_critic:
      self.logger.record_mean("train/critic_loss", (critic_loss_sum / n_critic).item())
      self._record_window_max("train/critic_loss_max", critic_loss_max.item())
    if n_ctrl:
      self.logger.record_mean("train/ctrl_actor_loss", (ctrl_loss_sum / n_ctrl).item())
    if n_dstb:
      self.logger.record_mean("train/dstb_actor_loss", (dstb_loss_sum / n_dstb).item())
    # Per-actor entropy temperature (alpha) + gamma; ctrl_ent/dstb_ent hold the
    # last step's values.
    self.logger.record("train/ent_coef_ctrl", float(ctrl_ent.mean()))
    self.logger.record("train/ent_coef_dstb", float(dstb_ent.mean()))
    self.logger.record("train/gamma", float(self.gamma))

  def _excluded_save_params(self):
    # League runtime objects are rebuilt by _setup_model on load; the eval env in
    # particular is unpicklable (e.g. mjlab holds a mujoco MjSpec).
    return super()._excluded_save_params() + [
      "dstb_actor",
      "_lb_eval_env",
      "_leaderboard",
      "_league_eval",
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


# ----------------------------------------------------------------- the modes

class SafetySAC2P(AbstractSAC2P):
  """Two-player off-policy **avoid** game — ISAACS proper (Hsu et al. 2022,
  eq. 7)::

      V(s) = (1-γ)·g + γ·max_ctrl min_dstb min(g, V')

  the robust-invariance value under a worst-case disturbance. No target set, no
  ``l`` — the paper has neither.

  This is the class for an adversarial *avoid* task ("stay standing against a
  worst-case force"). Do NOT emulate it by giving :class:`ReachAvoidSAC2P` a
  degenerate ``l``: no ``l`` reduces the reach-avoid operator to avoid
  (:mod:`safety_sb3.backups` proves the two conditions are contradictory).
  ``l`` is ignored if present, so an ``l``-carrying replay buffer is harmless.
  """

  _MODE = backups.AVOID


class ReachAvoidSAC2P(AbstractSAC2P):
  """Two-player off-policy **reach-avoid** game — Gameplay Filters (Hsu et al.
  2024, eq. 6a), which extends ISAACS to reach-avoid: anchor ``min(l, g)``.

  Needs a target margin ``l``; the base defaults the replay buffer to the
  ``l``-carrying one off this ``_MODE``. For a two-player *avoid* game (no
  target set) use :class:`SafetySAC2P`.
  """

  _MODE = backups.REACH_AVOID

  @staticmethod
  def _league_success(safe, reached):
    """Reach-avoid: the controller must have reached the target AND stayed safe."""
    return safe & reached
