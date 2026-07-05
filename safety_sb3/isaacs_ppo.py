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
    ctrl_action_dim: int,
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
    **kwargs,
  ) -> None:
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
    self.dstb_rollout_buffer = ReachAvoidRolloutBuffer(
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

  def train(self) -> None:
    phase = self._phase()
    if phase == "dstb":
      # Min player: SAME reach-avoid targets, NEGATED advantage; PPO.train()
      # runs against the swapped-in dstb policy/buffer.
      self.dstb_rollout_buffer.advantages *= -1.0
      ctrl_policy, ctrl_buf = self.policy, self.rollout_buffer
      self.policy, self.rollout_buffer = self.dstb_policy, self.dstb_rollout_buffer
      try:
        super().train()
      finally:
        self.policy, self.rollout_buffer = ctrl_policy, ctrl_buf
    else:
      super().train()

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
