"""IsaacsPPO — on-policy ISAACS (two-player zero-sum reach-avoid).

The ISAACS game (Hsu, Nguyen, Fisac, L4DC 2023) with a PPO learner instead of
SAC: the control player MAXIMIZES the reach-avoid value, the disturbance
player MINIMIZES it, and a leaderboard of archived opponents damps cycling.
The game, value definition, and leaderboard semantics match
:class:`IsaacsSAC`; the learner differs:

* both players are PPO policies over SUB-spaces of the env's concatenated
  action space (same env contract as :class:`IsaacsSAC`: one
  ``Box(ctrl_dim + dstb_dim)`` action, split by the env; ``g`` on the reward
  channel; ``l`` via ``info["l_x"]``);
* the reach-avoid targets are computed on-policy per rollout
  (:class:`ReachAvoidRolloutBuffer`); the min player trains on the SAME
  targets with a NEGATED advantage (negation commutes with advantage
  normalization) — the zero-sum property;
* training alternates in phases (dstb pretrain, then K dstb / M ctrl rollout
  cycles); the frozen player acts stochastically;
* with ``n_envs > 1``, leaderboard opponents are assigned to env SLICES so a
  single on-policy batch contains rollouts against the whole archived
  population (validated at 1280 GPU-parallel envs in ``unitree_rl_mjlab``;
  off-policy ISAACS saturates at ~32-64 envs).

``self.policy`` is always the CONTROL policy — ``predict()``, ``save()``, and
downstream filter wrappers see the deployable controller, as in
:class:`IsaacsPolicy`.
"""

from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from .leaderboard import Leaderboard
from .reach_avoid_ppo import ReachAvoidPPO
from .safety_buffers import ReachAvoidRolloutBuffer

# Slice opponent codes (match Leaderboard.sample_dstb_slices).
_ZERO, _RANDOM, _CURRENT = -3, -2, -1


class IsaacsPPO(ReachAvoidPPO):
  """Two-player on-policy reach-avoid game with leaderboard opponents."""

  def __init__(
    self,
    policy,
    env,
    # Sentinel default keeps SB3 ``load()`` working: load() constructs the
    # class BARE (cls(policy, env=None, _init_setup_model=False)) then restores
    # __dict__, so the saved ctrl_action_dim replaces the sentinel before
    # _setup_model() ever runs. Direct construction must pass a real value
    # (validated in _split_action_space).
    ctrl_action_dim: int = -1,
    dstb_pretrain_rollouts: int = 20,
    ctrl_rollouts_per_cycle: int = 4,
    dstb_rollouts_per_cycle: int = 1,
    use_leaderboard: bool = False,
    leaderboard_dir: str = "isaacs_ppo_leaderboard",
    save_top_k_ctrl: int = 5,
    save_top_k_dstb: int = 5,
    softmax_rationality: float = 3.0,
    leaderboard_freq_cycles: int = 1,
    num_slices: int = 8,
    dstb_learning_rate: float | None = None,
    dstb_ent_coef: float | None = None,
    **kwargs,
  ) -> None:
    # Per-player optimization: the two players sit in different optimization
    # regimes (ctrl is usually warm-started and precise; dstb is fresh and
    # exploratory). The reference setup used ctrl lr 5e-4 / dstb 3e-4 and
    # ctrl ent 5e-4 / dstb 2e-3. None -> inherit the ctrl value.
    self._dstb_lr = None if dstb_learning_rate is None else float(dstb_learning_rate)
    self._dstb_ent = None if dstb_ent_coef is None else float(dstb_ent_coef)
    self.ctrl_action_dim = int(ctrl_action_dim)
    self._dstb_pretrain = int(dstb_pretrain_rollouts)
    self._ctrl_per_cycle = int(ctrl_rollouts_per_cycle)
    self._dstb_per_cycle = int(dstb_rollouts_per_cycle)
    self._use_lb = bool(use_leaderboard)
    self._lb_dir = leaderboard_dir
    self._lb_kc = int(save_top_k_ctrl)
    self._lb_kd = int(save_top_k_dstb)
    self._lb_rationality = float(softmax_rationality)
    self._lb_freq_cycles = int(leaderboard_freq_cycles)
    self._n_slices = int(num_slices)
    self._rollouts_done = 0
    super().__init__(policy, env, **kwargs)

  # --- spaces / model setup --------------------------------------------------

  def _split_action_space(self):
    full = self.action_space
    assert isinstance(full, spaces.Box) and len(full.shape) == 1
    c = self.ctrl_action_dim
    assert c > 0, (
      "ctrl_action_dim not set — pass it to IsaacsPPO(...) (the -1 default "
      "exists only so SB3 load() can construct the class bare).")
    ctrl = spaces.Box(low=full.low[:c], high=full.high[:c], dtype=full.dtype)
    dstb = spaces.Box(low=full.low[c:], high=full.high[c:], dtype=full.dtype)
    return ctrl, dstb

  def _setup_model(self) -> None:
    # Build the CTRL policy/buffer over the ctrl SUB-space by temporarily
    # narrowing the action space seen by the base setup.
    full_space = self.action_space
    ctrl_space, dstb_space = self._split_action_space()
    self.action_space = ctrl_space
    super()._setup_model()
    self.action_space = full_space
    self._full_action_space = full_space
    self._ctrl_space, self._dstb_space = ctrl_space, dstb_space

    # Min player: same policy class/kwargs over the dstb sub-space.
    self.dstb_policy = self.policy_class(
      self.observation_space, dstb_space, self.lr_schedule,
      use_sde=self.use_sde, **self.policy_kwargs,
    ).to(self.device)
    dstb_buf_cls = (self.tensor_rollout_buffer_class if self._tensor_path
                    else ReachAvoidRolloutBuffer)
    self.dstb_rollout_buffer = dstb_buf_cls(
      self.n_steps, self.observation_space, dstb_space,
      device=self.device, gamma=self.gamma, gae_lambda=self.gae_lambda,
      n_envs=self.n_envs,
    )

    self._leaderboard: Leaderboard | None = None
    if self._use_lb:
      self._leaderboard = Leaderboard(
        self._lb_kc, self._lb_kd, self._lb_rationality, self._lb_dir
      )
      self._scratch_dstb = [
        self.policy_class(
          self.observation_space, dstb_space, self.lr_schedule,
          use_sde=self.use_sde, **self.policy_kwargs,
        ).to(self.device)
        for _ in range(self._lb_kd)
      ]
    self._slice_opps: list[int] = [_ZERO, _RANDOM] + [_CURRENT] * max(
      self._n_slices - 2, 0
    )
    n = self.n_envs
    self._slice_of_env = (np.arange(n) * self._n_slices // max(n, 1)).clip(
      max=self._n_slices - 1
    )
    self._ever_l = np.zeros(n, dtype=bool)
    self._ever_gneg = np.zeros(n, dtype=bool)

    # Per-player KL-adaptive LR state: a SHARED controller cross-contaminates
    # (a dstb phase's KL would move the LR the next ctrl phase trains with,
    # and vice versa). Each player keeps its own scalar; train() swaps the
    # active one into SafetyPPO's controller and stores it back after.
    if self.adaptive_lr:
      self._adaptive_lr_ctrl = float(self._adaptive_lr)
      self._adaptive_lr_dstb = float(self._dstb_lr if self._dstb_lr is not None
                                     else self._adaptive_lr)
    if self._dstb_lr is not None:
      for grp in self.dstb_policy.optimizer.param_groups:
        grp["lr"] = self._dstb_lr

  # --- phase machine -----------------------------------------------------------

  def _phase(self) -> str:
    if self._rollouts_done < self._dstb_pretrain:
      return "dstb"
    k = (self._rollouts_done - self._dstb_pretrain) % (
      self._dstb_per_cycle + self._ctrl_per_cycle
    )
    return "dstb" if k < self._dstb_per_cycle else "ctrl"

  def _cycle_len(self) -> int:
    return self._dstb_per_cycle + self._ctrl_per_cycle

  # --- opponents ---------------------------------------------------------------

  def _resample_slice_opponents(self) -> None:
    if self._leaderboard is None:
      self._slice_opps = [_ZERO, _RANDOM] + [_CURRENT] * max(self._n_slices - 2, 0)
      return
    self._slice_opps = self._leaderboard.sample_dstb_slices(self._n_slices)
    self._loaded_scratch: dict[int, th.nn.Module] = {}
    for opp in {o for o in self._slice_opps if o >= 0}:
      scratch = self._scratch_dstb[len(self._loaded_scratch) % len(self._scratch_dstb)]
      self._leaderboard.load_actor(
        scratch, "dstb", self._leaderboard.dstb_steps[opp]
      )
      scratch.to(self.device)
      self._loaded_scratch[opp] = scratch

  def _dstb_actions_for_ctrl_phase(self, obs_tensor) -> np.ndarray:
    """Per-slice opponent disturbance (stochastic), in dstb sub-space units."""
    n = self.n_envs
    out = np.zeros((n, self._dstb_space.shape[0]), dtype=np.float32)
    cur_cache = None
    for si, opp in enumerate(self._slice_opps):
      mask = self._slice_of_env == si
      if not mask.any() or opp == _ZERO:
        continue
      if opp == _RANDOM:
        out[mask] = np.random.uniform(
          self._dstb_space.low, self._dstb_space.high,
          size=(int(mask.sum()), self._dstb_space.shape[0]),
        )
      elif opp == _CURRENT:
        if cur_cache is None:
          with th.no_grad():
            a, _, _ = self.dstb_policy(obs_tensor)
          cur_cache = a.cpu().numpy()
        out[mask] = cur_cache[mask]
      else:
        with th.no_grad():
          a, _, _ = self._loaded_scratch[opp](obs_tensor)
        out[mask] = a.cpu().numpy()[mask]
    return out

  def _dstb_actions_tensor(self, obs: th.Tensor) -> th.Tensor:
    """Per-slice opponent disturbance, torch end-to-end (ctrl phase)."""
    n = self.n_envs
    dev = obs.device
    d = self._dstb_space.shape[0]
    lo = th.as_tensor(self._dstb_space.low, dtype=th.float32, device=dev)
    hi = th.as_tensor(self._dstb_space.high, dtype=th.float32, device=dev)
    out = th.zeros(n, d, device=dev)
    slice_of_env = getattr(self, "_slice_of_env_t", None)
    if slice_of_env is None or slice_of_env.shape[0] != n:
      self._slice_of_env_t = th.as_tensor(self._slice_of_env, device=dev)
      slice_of_env = self._slice_of_env_t
    cur = None
    for si, opp in enumerate(self._slice_opps):
      mask = slice_of_env == si
      if not bool(mask.any()) or opp == _ZERO:
        continue
      if opp == _RANDOM:
        out[mask] = lo + (hi - lo) * th.rand(int(mask.sum()), d, device=dev)
      elif opp == _CURRENT:
        if cur is None:
          with th.no_grad():
            cur, _, _ = self.dstb_policy(obs)
        out[mask] = cur[mask]
      else:
        with th.no_grad():
          a, _, _ = self._loaded_scratch[opp](obs)
        out[mask] = a[mask]
    return out

  def _collect_rollouts_tensor(self, env, callback, rollout_buffer,
                               n_rollout_steps: int) -> bool:
    """Torch twin of the two-player rollout: both actions each step, env gets
    the concatenation, only the ACTIVE player's data is stored."""
    del rollout_buffer
    phase = self._phase()
    active = self.dstb_policy if phase == "dstb" else self.policy
    passive = self.policy if phase == "dstb" else self.dstb_policy
    buf = self.dstb_rollout_buffer if phase == "dstb" else self.rollout_buffer

    assert self._last_obs is not None
    active.set_training_mode(False)
    passive.set_training_mode(False)
    if phase == "ctrl":
      self._resample_slice_opponents()

    dev = env.device
    obs = self._last_obs
    if not th.is_tensor(obs):
      obs = th.as_tensor(np.asarray(obs), dtype=th.float32, device=dev)
    episode_starts = (self._last_episode_starts
                      if th.is_tensor(self._last_episode_starts)
                      else th.as_tensor(np.asarray(
                        self._last_episode_starts, dtype=np.float32), device=dev))
    c_lo = th.as_tensor(self._ctrl_space.low, dtype=th.float32, device=dev)
    c_hi = th.as_tensor(self._ctrl_space.high, dtype=th.float32, device=dev)
    d_lo = th.as_tensor(self._dstb_space.low, dtype=th.float32, device=dev)
    d_hi = th.as_tensor(self._dstb_space.high, dtype=th.float32, device=dev)
    ever_l = getattr(self, "_ever_l_t", None)
    if ever_l is None or ever_l.shape[0] != env.num_envs:
      self._ever_l_t = th.zeros(env.num_envs, dtype=th.bool, device=dev)
      self._ever_gneg_t = th.zeros(env.num_envs, dtype=th.bool, device=dev)
    slice_of_env = getattr(self, "_slice_of_env_t", None)
    if slice_of_env is None:
      self._slice_of_env_t = th.as_tensor(self._slice_of_env, device=dev)

    buf.reset()
    callback.on_rollout_start()
    n_steps = 0
    while n_steps < n_rollout_steps:
      with th.no_grad():
        actions, values, log_probs = active(obs)
      if phase == "dstb":
        with th.no_grad():
          ctrl_a, _, _ = passive(obs)
        dstb_a = actions
      else:
        ctrl_a = actions
        dstb_a = self._dstb_actions_tensor(obs)
      full = th.cat([th.clamp(ctrl_a, c_lo, c_hi),
                     th.clamp(dstb_a, d_lo, d_hi)], dim=1)

      new_obs, rewards, dones, timeouts, l_x = env.step_tensor(full)
      self.num_timesteps += env.num_envs
      n_steps += 1
      callback.update_locals(locals())
      if not callback.on_step():
        return False

      buf.l_x[buf.pos] = l_x.reshape(buf.n_envs)
      buf.add(obs, actions, rewards, episode_starts,
              values.flatten(), log_probs)

      self._ever_l_t |= l_x >= 0.0
      self._ever_gneg_t |= rewards < 0.0
      if self._leaderboard is not None and bool(dones.any()):
        d_b = dones.bool()
        succ = d_b & timeouts.bool() & self._ever_l_t & ~self._ever_gneg_t
        b = self._leaderboard.board
        if phase == "ctrl":
          for si, opp in enumerate(self._slice_opps):
            m = d_b & (self._slice_of_env_t == si)
            if not bool(m.any()):
              continue
            col = (b.shape[1] - 1 if opp == _ZERO
                   else b.shape[1] - 2 if opp in (_RANDOM, _CURRENT) else opp)
            self._leaderboard.ema_score(-1, col, float(succ[m].float().mean()))
        else:
          self._leaderboard.ema_score(
            -1, b.shape[1] - 2, float(succ[d_b].float().mean()))
        self._ever_l_t = th.where(d_b, th.zeros_like(self._ever_l_t), self._ever_l_t)
        self._ever_gneg_t = th.where(d_b, th.zeros_like(self._ever_gneg_t),
                                     self._ever_gneg_t)

      obs = new_obs
      episode_starts = dones.float()

    with th.no_grad():
      last_values = active.predict_values(obs)
    buf.compute_returns_and_advantage(last_values=last_values.flatten(),
                                      dones=dones.float())
    self._last_obs = obs
    self._last_episode_starts = episode_starts
    for k, v in (env.metrics() or {}).items():
      self.logger.record(f"env/{k}", float(v))
    callback.update_locals(locals())
    callback.on_rollout_end()
    return True

  # --- rollout collection --------------------------------------------------------

  def collect_rollouts(
    self,
    env: VecEnv,
    callback: BaseCallback,
    rollout_buffer: RolloutBuffer,
    n_rollout_steps: int,
  ) -> bool:
    """Two-player rollout: both actions computed each step, env receives the
    concatenation, only the ACTIVE player's data is stored (in its buffer)."""
    if self._tensor_path:
      return self._collect_rollouts_tensor(env, callback, rollout_buffer,
                                           n_rollout_steps)
    del rollout_buffer  # phase decides the buffer
    phase = self._phase()
    active_policy = self.dstb_policy if phase == "dstb" else self.policy
    passive_policy = self.policy if phase == "dstb" else self.dstb_policy
    buf: ReachAvoidRolloutBuffer = (
      self.dstb_rollout_buffer if phase == "dstb" else self.rollout_buffer
    )

    assert self._last_obs is not None
    active_policy.set_training_mode(False)
    passive_policy.set_training_mode(False)
    if phase == "ctrl":
      self._resample_slice_opponents()

    n_steps = 0
    buf.reset()
    callback.on_rollout_start()

    while n_steps < n_rollout_steps:
      with th.no_grad():
        obs_tensor = obs_as_tensor(self._last_obs, self.device)
        actions, values, log_probs = active_policy(obs_tensor)
      actions = actions.cpu().numpy()

      if phase == "dstb":
        with th.no_grad():
          ctrl_a, _, _ = passive_policy(obs_tensor)  # frozen ctrl, stochastic
        ctrl_np = ctrl_a.cpu().numpy()
        dstb_np = actions
      else:
        ctrl_np = actions
        dstb_np = self._dstb_actions_for_ctrl_phase(obs_tensor)

      ctrl_clipped = np.clip(ctrl_np, self._ctrl_space.low, self._ctrl_space.high)
      dstb_clipped = np.clip(dstb_np, self._dstb_space.low, self._dstb_space.high)
      full_actions = np.concatenate([ctrl_clipped, dstb_clipped], axis=1)

      new_obs, rewards, dones, infos = env.step(full_actions)
      self.num_timesteps += env.num_envs

      callback.update_locals(locals())
      if not callback.on_step():
        return False
      self._update_info_buffer(infos, dones)
      n_steps += 1

      # Timeout bootstrap (ACTIVE player's value fn) — DISABLED by default:
      # the reward is the physical margin g(s) (see SafetyPPO docstring).
      if self.bootstrap_on_timeout:
        for idx, done in enumerate(dones):
          if (
            done
            and infos[idx].get("terminal_observation") is not None
            and infos[idx].get("TimeLimit.truncated", False)
          ):
            terminal_obs = active_policy.obs_to_tensor(
              infos[idx]["terminal_observation"]
            )[0]
            with th.no_grad():
              terminal_value = active_policy.predict_values(terminal_obs)[0]
            rewards[idx] += self.gamma * terminal_value

      l_now = np.array(
        [float(info.get("l_x", 0.0)) for info in infos], dtype=np.float32
      )
      buf.l_x[buf.pos] = l_now
      buf.add(
        self._last_obs, actions, rewards,
        self._last_episode_starts, values, log_probs,
      )

      # reach-avoid episode flags -> leaderboard board (training outcomes)
      self._ever_l |= l_now >= 0.0
      self._ever_gneg |= rewards < 0.0
      if self._leaderboard is not None and dones.any():
        timeouts = np.array(
          [bool(info.get("TimeLimit.truncated", False)) for info in infos]
        )
        succ = dones & timeouts & self._ever_l & ~self._ever_gneg
        b = self._leaderboard.board
        if phase == "ctrl":
          for si, opp in enumerate(self._slice_opps):
            m = dones & (self._slice_of_env == si)
            if not m.any():
              continue
            col = (b.shape[1] - 1 if opp == _ZERO
                   else b.shape[1] - 2 if opp in (_RANDOM, _CURRENT) else opp)
            self._leaderboard.ema_score(-1, col, float(succ[m].mean()))
        else:
          self._leaderboard.ema_score(
            -1, b.shape[1] - 2, float(succ[dones].mean())
          )
        self._ever_l[dones] = False
        self._ever_gneg[dones] = False

      self._last_obs = new_obs
      self._last_episode_starts = dones

    with th.no_grad():
      values = active_policy.predict_values(obs_as_tensor(new_obs, self.device))
    buf.compute_returns_and_advantage(last_values=values, dones=dones)

    callback.update_locals(locals())
    callback.on_rollout_end()
    return True

  # --- updates -----------------------------------------------------------------

  def _update_learning_rate(self, optimizers) -> None:
    """Phase-aware: after the base update (adaptive scalar or ctrl schedule),
    re-apply the dstb player's own fixed LR in dstb phases (the base call
    would otherwise apply the ctrl schedule to the dstb optimizer)."""
    super()._update_learning_rate(optimizers)
    if (not self.adaptive_lr and self._dstb_lr is not None
        and self._phase() == "dstb"):
      from stable_baselines3.common.utils import update_learning_rate
      update_learning_rate(
        optimizers if not isinstance(optimizers, list) else optimizers[0],
        self._dstb_lr)
      self.logger.record("isaacs/lr_dstb", self._dstb_lr)

  def train(self) -> None:
    phase = self._phase()
    if phase == "dstb":
      # Min player: SAME reach-avoid targets, NEGATED advantage; PPO.train()
      # runs against the swapped-in dstb policy/buffer with the DSTB player's
      # own LR (adaptive state) and entropy coefficient.
      self.dstb_rollout_buffer.advantages *= -1.0
      ctrl_policy, ctrl_buf = self.policy, self.rollout_buffer
      ctrl_ent = self.ent_coef
      self.policy, self.rollout_buffer = self.dstb_policy, self.dstb_rollout_buffer
      if self._dstb_ent is not None:
        self.ent_coef = self._dstb_ent
      if self.adaptive_lr:
        self._adaptive_lr = self._adaptive_lr_dstb
      try:
        super().train()
      finally:
        if self.adaptive_lr:
          self._adaptive_lr_dstb = float(self._adaptive_lr)
          self.logger.record("isaacs/lr_dstb", self._adaptive_lr_dstb)
        self.policy, self.rollout_buffer = ctrl_policy, ctrl_buf
        self.ent_coef = ctrl_ent
    else:
      if self.adaptive_lr:
        self._adaptive_lr = self._adaptive_lr_ctrl
      super().train()
      if self.adaptive_lr:
        self._adaptive_lr_ctrl = float(self._adaptive_lr)
        self.logger.record("isaacs/lr_ctrl", self._adaptive_lr_ctrl)

    self._rollouts_done += 1
    self.logger.record("isaacs/phase_is_dstb", float(phase == "dstb"))
    self.logger.record("isaacs/rollouts_done", self._rollouts_done)

    # snapshot + prune every N full cycles (after pretrain)
    if (
      self._leaderboard is not None
      and self._rollouts_done > self._dstb_pretrain
      and (self._rollouts_done - self._dstb_pretrain)
      % (self._cycle_len() * self._lb_freq_cycles) == 0
    ):
      self._leaderboard.prune(
        self.num_timesteps, self.policy, self.dstb_policy
      )
      self.logger.record(
        "isaacs/archived_dstb", float(len(self._leaderboard.dstb_steps))
      )

  # --- persistence ---------------------------------------------------------------

  def _excluded_save_params(self):
    return super()._excluded_save_params() + [
      "_leaderboard", "_scratch_dstb", "_loaded_scratch",
    ]

  def _get_torch_save_params(self):
    state_dicts, tensors = super()._get_torch_save_params()
    state_dicts = state_dicts + ["dstb_policy", "dstb_policy.optimizer"]
    return state_dicts, tensors
