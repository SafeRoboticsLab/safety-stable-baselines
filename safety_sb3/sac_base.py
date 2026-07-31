"""SAC, written without a fixed Bellman backup — the shared half of the family.

:class:`AbstractSAC` is everything the SAC-family learners share regardless of
how many players are in the game: the constructor and mode check, replay-buffer
selection and validation, the entropy-temperature bounds / reset, gamma
annealing, the GPU-resident collection loop, and the TD target itself. It
deliberately owns **no** ``train()`` — that is the axis its subclasses split on:

    :class:`~safety_sb3.sac_1p.AbstractSAC1P` — one actor, one entropy temp.
    :class:`~safety_sb3.sac_2p.AbstractSAC2P` — two actors over disjoint action
        sub-spaces, two entropy temps, ONE twin critic over the joint action.

Which value function a learner converges to is a *parameter*, not a class: the
update loops form the soft next-state value ``V'`` and hand it to
:func:`safety_sb3.backups.target` together with ``self._MODE``.

    ``_MODE = backups.AVOID``        -> SafetySAC1P / SafetySAC2P
    ``_MODE = backups.REACH_AVOID``  -> ReachAvoidSAC1P / ReachAvoidSAC2P
    ``_MODE = backups.CUMULATIVE``   -> CumulativeSAC1P (ordinary SAC)

Each of those is a one-line specialization; the mode is also selectable per
instance via the ``mode=`` constructor argument. This follows Bertsekas's
abstract-DP framing (define the algorithm against an abstract operator, then
choose the operator) and it is why the reach-avoid learner cannot drift from the
avoid learner again: before this class they were two hand-copied ``train()``
bodies, and the copy had already lost the ``min_alpha``/``max_alpha`` clamp.

Note that SAC needs no reach-avoid mixin, unlike the on-policy family: its whole
mode delta is the buffer choice and the ``l``-carrying flag below, both of which
already dispatch on ``_MODE`` right here.
"""
from __future__ import annotations

import numpy as np
import torch as th

from stable_baselines3.common.type_aliases import RolloutReturn, TrainFreq
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.sac.sac import SAC

from . import backups
from .buffers_replay import ReachAvoidReplayBuffer
from .env_checks import guard_and_normalize_env
from .gamma_anneal import GammaAnnealMixin


class AbstractSAC(GammaAnnealMixin, SAC):
  """SAC whose Bellman backup is chosen by ``_MODE`` (see the module docstring).

  Subclasses SB3's SAC to reuse the actor, entropy regularization, replay
  buffer, etc. Abstract: it has no ``train()``.

  GPU-resident path: pass a ``TensorVecEnv`` (detected via ``is_tensor_env``,
  optionally auto-wrapped in ``TensorVecNormalize`` with ``normalize_obs=True``)
  and collection + replay stay on device end-to-end
  (:class:`safety_sb3.tensor_replay.TensorReplayBuffer`); ``train()`` is
  unchanged -- the tensor buffer returns the same named tuples. NOTE on the
  tensor path set ``gradient_steps`` explicitly (>=1): the SB3 convention
  ``gradient_steps=-1`` ("as many as env steps") would mean num_envs updates
  per vector step.

  :param mode: backup to converge to, one of :data:`safety_sb3.backups.MODES`.
      Defaults to the class's ``_MODE``, so subclasses need only set that.
  :param terminal_type: terminal target for the reach-avoid backup; ignored by
      the other modes. See :func:`safety_sb3.backups.reach_avoid_target`.
  """

  #: default backup; subclasses set this, ``mode=`` overrides it per instance
  _MODE = backups.AVOID
  _tensor_store_l = False  # reach-avoid flips this (buffer stores l_x)

  def __init__(self, *args, mode: str | None = None,
               replay_buffer_class=None, terminal_type: str = "all",
               normalize_obs: bool = False, gamma_anneal=True,
               min_alpha: float | None = 1e-3, max_alpha: float | None = None,
               **kwargs):
    # The mode must be known before super().__init__ -> _setup_model, which
    # picks the buffer (an l-carrying one for reach-avoid).
    self._MODE = backups.check_mode(self._MODE if mode is None else mode)
    self.terminal_type = backups.check_terminal_type(terminal_type)
    if self._MODE == backups.REACH_AVOID:
      if replay_buffer_class is None:
        replay_buffer_class = ReachAvoidReplayBuffer
      self._tensor_store_l = True  # tensor buffer stores l(s)
    if "env" in kwargs:
      kwargs["env"] = guard_and_normalize_env(kwargs["env"], normalize_obs)
    elif len(args) >= 2:
      args = list(args)
      args[1] = guard_and_normalize_env(args[1], normalize_obs)
    _env = kwargs.get("env", args[1] if len(args) >= 2 else None)
    self._tensor_path = bool(getattr(_env, "is_tensor_env", False))
    # Entropy-temperature (alpha) FLOOR/ceiling (reference: min_alpha=1e-3).
    # SB3's auto-entropy is unbounded; a floor stops alpha collapsing to ~0
    # (deterministic, no exploration) esp. after a gamma-jump alpha reset.
    # Set before super().__init__ -> _setup_model reads them.
    self._min_alpha = None if min_alpha is None else float(min_alpha)
    self._max_alpha = None if max_alpha is None else float(max_alpha)
    super().__init__(*args, replay_buffer_class=replay_buffer_class, **kwargs)
    # Discount-factor annealing (ON by default): the REFERENCE-FAITHFUL
    # discrete-jump schedule (gamma 0.99 -> 0.999 @20% -> 0.9999 @40%, hold);
    # each jump resets alpha via _on_gamma_jump (Q-scale shift). self.gamma is
    # read in the TD target of train(). See gamma_anneal.py.
    self._setup_gamma_anneal(gamma_anneal)
    self._check_replay_buffer()

  def _check_replay_buffer(self) -> None:
    """Reach-avoid needs a buffer that stores ``l(s)``.

    On load (``_init_setup_model=False``) the buffer is built later; only
    validate when it already exists. The tensor path builds its own
    l-carrying buffer.
    """
    if self._MODE != backups.REACH_AVOID or self._tensor_path:
      return
    if self.replay_buffer is not None and not isinstance(
        self.replay_buffer, ReachAvoidReplayBuffer):
      raise TypeError(
        f"{type(self).__name__} is in mode={self._MODE!r} and needs a "
        "ReachAvoidReplayBuffer (it stores l(s)); got "
        f"{type(self.replay_buffer).__name__}."
      )

  def _setup_model(self) -> None:
    if not self._tensor_path:
      super()._setup_model()
      self._setup_entropy_bounds()
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
    self._setup_entropy_bounds()

  # --- entropy-temperature (alpha) bounds + reset (issues 6 & 7) -----------
  def _setup_entropy_bounds(self) -> None:
    """Compute the log-space alpha clamp bounds and snapshot the init
    log_ent_coef (so a gamma jump can reset alpha to it)."""
    import math
    # A NON-POSITIVE bound means "no bound": log() is undefined there, and a
    # floor of 0 is exactly the request to let alpha decay freely. Treating it
    # as None rather than raising matters because `--min-alpha 0` is the
    # natural way to ASK for an unfloored run, and a math domain error at
    # construction kills the run before it starts (E056 lost two cells this
    # way, leaving the min_alpha hypothesis untested).
    self._log_min_alpha = (None if not self._min_alpha or self._min_alpha <= 0
                           else math.log(self._min_alpha))
    self._log_max_alpha = (None if not self._max_alpha or self._max_alpha <= 0
                           else math.log(self._max_alpha))
    lec = getattr(self, "log_ent_coef", None)   # None for a FIXED ent_coef
    self._init_log_ent_coef = None if lec is None else lec.detach().clone()

  # --- logging over the DUMP WINDOW, not the last call ----------------------
  # On the tensor path _dump_logs() fires every 50k env-steps, which at 1024
  # envs is ~49 train() calls. ``Logger.record`` OVERWRITES ("if called many
  # times, last value will be used"), so a per-train() record() reaches wandb as
  # a 1-in-49 sample and ~96% of the window is discarded unseen. That is how
  # E057 could go critic_loss 0.0014 -> 1.95e+07 inside a single window with
  # only the endpoints visible.
  #
  # ``record_mean`` fixes the average. It does NOT fix an onset spike, which a
  # 49-call mean dilutes -- hence the running MAX below, reset on dump.

  def _record_window_max(self, key: str, value: float) -> None:
    """Running max of ``value`` since the last dump, recorded under ``key``."""
    store = getattr(self, "_win_max_store", None)
    if store is None:
      store = self._win_max_store = {}
    cur = store.get(key)
    store[key] = value if cur is None else max(cur, value)
    self.logger.record(key, store[key])

  def _reset_window_max(self) -> None:
    """Start a fresh window. Called immediately after a dump."""
    self._win_max_store = {}

  def _clamp_entropy_temps(self) -> None:
    """Clamp the learned entropy temperature into [min_alpha, max_alpha].
    Called after each entropy-coefficient optimizer step in train()."""
    if self._log_min_alpha is None and self._log_max_alpha is None:
      return
    lec = getattr(self, "log_ent_coef", None)
    if lec is not None:
      with th.no_grad():
        lec.clamp_(min=self._log_min_alpha, max=self._log_max_alpha)

  def _entropy_optimizer_lr(self) -> float:
    """LR for the (ctrl) entropy-coefficient optimizer. Falls back to the
    shared ``learning_rate`` when no dedicated ``ent_coef_lr`` was set
    (the two-player learners set ``self._ent_coef_lr``); keeps a rebuilt
    optimizer on the configured entropy lr rather than the shared one."""
    lr = getattr(self, "_ent_coef_lr", None)
    return float(self.lr_schedule(1)) if lr is None else float(lr)

  def _reset_entropy_temp(self) -> None:
    """Reset the learned entropy temperature to its init value and rebuild
    its optimizer (clears Adam moments) -- the reference alpha reset on a
    gamma jump. No-op for a fixed ent_coef."""
    lec = getattr(self, "log_ent_coef", None)
    if lec is None or self._init_log_ent_coef is None:
      return
    with th.no_grad():
      lec.data.copy_(self._init_log_ent_coef)
    if getattr(self, "ent_coef_optimizer", None) is not None:
      self.ent_coef_optimizer = th.optim.Adam(
        [lec], lr=self._entropy_optimizer_lr())

  def _on_gamma_jump(self, old_gamma: float, new_gamma: float) -> None:
    """A discrete gamma jump shifts the Q-scale, so the tuned entropy
    temperature is stale -> reset it (reference: reset alpha on every jump)."""
    self._reset_entropy_temp()
    logger = getattr(self, "logger", None)
    if logger is not None:
      logger.record("train/alpha_reset_gamma", float(new_gamma))

  # --- tensor collection ---------------------------------------------------

  def collect_rollouts(self, env, callback, train_freq: TrainFreq,
                       replay_buffer, action_noise=None, learning_starts=0,
                       log_interval=None) -> RolloutReturn:
    # Anneal self.gamma before the train() that follows this collection reads
    # it in the TD target. The tensor path bypasses SB3's
    # _update_current_progress_remaining, so apply it here (idempotent).
    self._apply_gamma_anneal()
    if self._tensor_path:
      return self._collect_rollouts_tensor(
        env, callback, train_freq, replay_buffer,
        learning_starts=learning_starts, log_interval=log_interval)
    return super().collect_rollouts(
      env, callback, train_freq, replay_buffer, action_noise=action_noise,
      learning_starts=learning_starts, log_interval=log_interval)

  def _tensor_policy_actions(self, obs: "th.Tensor") -> "th.Tensor":
    """Post-warmup policy actions for the tensor collect: on device, in the
    env's action range, shape ``(num_envs, action_dim)``.

    Single-player: the ctrl actor. The two-player learners override this to
    sample BOTH players and concatenate ``[a_ctrl, a_dstb]`` so the composed
    action matches the env's ``ctrl_dim + dstb_dim`` action space (the
    numpy-path analog is :meth:`~safety_sb3.sac_2p.AbstractSAC2P._sample_action`).
    """
    with th.no_grad():
      return self.actor(obs, deterministic=False)

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
        actions = self._tensor_policy_actions(obs)
      actions = th.clamp(actions, low, high)

      new_obs, g, dones, timeouts, l_x = env.step_tensor(actions)
      # An env may EXECUTE something other than what it was handed: a safety
      # filter replacing an uncertified proposal with its fallback's action is
      # the case this exists for. The buffer must hold the transition that
      # ACTUALLY happened -- storing the proposal against the resulting
      # (reward, next state) would fit the critic to a transition that never
      # occurred. Off-policy learning makes the substitution exactly correct
      # with no importance correction, which is why filtered training is run
      # with SAC. An env that does not override leaves `actions` untouched.
      executed = getattr(env, "executed_action", None)
      if executed is not None:
        actions = executed
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
      self._reset_window_max()   # the window just closed; start the next one

    callback.on_rollout_end()
    return RolloutReturn(num_collected_steps * env.num_envs,
                         num_collected_episodes, continue_training)

  def train(self, gradient_steps: int, batch_size: int) -> None:
    """Refuse to fall through to SB3's ``SAC.train``.

    This class has no update loop by design — that is the Players axis. Without
    this guard, instantiating it directly would inherit stock SAC's train() and
    silently converge to the CUMULATIVE fixed point while ``_MODE`` claimed
    otherwise: exactly the "different fixed points under one name" bug the
    library exists to prevent.
    """
    raise NotImplementedError(
      f"{type(self).__name__} has no update loop. Use a concrete learner "
      "(SafetySAC1P, ReachAvoidSAC2P, ...), or AbstractSAC1P / AbstractSAC2P "
      "with mode= if you want the loop without a named mode.")

  def _bellman_target(self, replay_data, v_next: "th.Tensor") -> "th.Tensor":
    """The backup for THIS instance's mode -- see :mod:`safety_sb3.backups`.

    ``replay_data.rewards`` carries the safety margin ``g(s)`` in the safety
    modes and the reward ``r`` in ``CUMULATIVE``; ``l_x`` is present only on an
    l-carrying buffer (reach-avoid) and is ignored by the other modes.
    """
    return backups.target(
      self._MODE, replay_data.rewards, v_next, 1.0 - replay_data.dones,
      self.gamma, l=getattr(replay_data, "l_x", None),
      terminal_type=self.terminal_type)
