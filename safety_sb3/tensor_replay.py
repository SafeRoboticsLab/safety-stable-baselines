"""GPU-resident replay buffers for the off-policy (SAC) safety family.

The numpy SAC path pays a device->host bounce per step plus host->device
sampling transfers per gradient step. This buffer keeps the whole replay on
device: `add_batch` writes one vectorized transition (num_envs rows) per env
step; `sample` gathers uniformly on device and returns the SAME named tuples
the numpy buffers return — so the SAC-family `train()` runs
UNCHANGED on top of it.

Semantics:
  * `rewards` field carries the safety margin g(s).
  * effective dones at sample time depend on `bootstrap_on_timeout`:
      True  -> `dones * (1 - timeouts)`  (SB3's convention: a timeout-truncated
               transition bootstraps its stored next_obs). CORRECT for CUMULATIVE
               (an arbitrary time cutoff on an ordinary return should bootstrap).
      False -> `dones`  (a timeout is TERMINAL, valued min(l,g) by the backup).
               CORRECT for the MARGIN modes (safety / reach-avoid): a timeout is
               the finite-horizon cutoff of the reach-avoid value (Gameplay
               Filters eq.5b). Matches the reference (safe_adaptation_dev
               `agent/replay_memory.py`: non_final = ~done, and `done` includes
               timeout) and safety_sb3's own PPO (`bootstrap_on_timeout=False`).
               `AbstractSAC` sets this default from `_MODE`.
  * `l_x` is stored when `store_l=True` (reach-avoid).

Two documented deviations from SB3-with-VecNormalize:
  * Observations are stored as COLLECTED (i.e. already normalized by
    TensorVecNormalize) and are NOT re-normalized with current stats at sample
    time (rsl_rl-style). With a drifting normalizer old samples are slightly
    stale; freeze the normalizer after warmup if this matters.
  * On mjlab-style auto-resetting envs the stored next_obs of a done step is the
    RESET obs, so bootstrapping a timeout targets V(reset spawn). With a hostile
    reset distribution (e.g. a reverse curriculum whose reset IS the failure
    frontier) that corrupts the value of every survivor. `bootstrap_on_timeout=
    False` (the default for the margin modes) never bootstraps a timeout, so the
    wart cannot bite there; it remains only for CUMULATIVE, where resets are
    benign.

Memory: 2 * slots * n_envs * obs_dim * 4 bytes dominates
(slots = buffer_size // n_envs). Size `buffer_size` accordingly for
large-observation tasks.
"""

from __future__ import annotations

import torch as th
from stable_baselines3.common.buffers import ReplayBufferSamples

from safety_sb3.buffers_replay import ReachAvoidReplayBufferSamples


class TensorReplayBuffer:
  """Device-resident circular replay for TensorVecEnv collection."""

  def __init__(self, buffer_size: int, obs_dim: int, act_dim: int,
               n_envs: int, device: str = "cuda:0", store_l: bool = False,
               bootstrap_on_timeout: bool = True):
    self.slots = max(int(buffer_size) // int(n_envs), 1)
    self.n_envs = int(n_envs)
    self.device = device
    self.store_l = bool(store_l)
    # True: timeout transitions bootstrap (SB3 convention; correct for CUMULATIVE).
    # False: a timeout is terminal -> min(l,g) (correct for the margin modes).
    self.bootstrap_on_timeout = bool(bootstrap_on_timeout)
    S, N = self.slots, self.n_envs
    self.observations = th.zeros(S, N, obs_dim, device=device)
    self.next_observations = th.zeros(S, N, obs_dim, device=device)
    self.actions = th.zeros(S, N, act_dim, device=device)
    self.rewards = th.zeros(S, N, device=device)          # g(s)
    self.dones = th.zeros(S, N, device=device)
    self.timeouts = th.zeros(S, N, device=device)
    self.l_x = th.zeros(S, N, device=device) if store_l else None
    self.pos = 0
    self.full = False

  def add_batch(self, obs: th.Tensor, next_obs: th.Tensor, actions: th.Tensor,
                g: th.Tensor, dones: th.Tensor, timeouts: th.Tensor,
                l_x: th.Tensor | None = None) -> None:
    p = self.pos
    self.observations[p] = obs
    self.next_observations[p] = next_obs
    self.actions[p] = actions
    self.rewards[p] = g
    self.dones[p] = dones.float()
    self.timeouts[p] = timeouts.float()
    if self.store_l:
      assert l_x is not None, "store_l=True buffer needs l_x each add"
      self.l_x[p] = l_x
    self.pos += 1
    if self.pos == self.slots:
      self.pos = 0
      self.full = True

  def size(self) -> int:
    return (self.slots if self.full else self.pos) * self.n_envs

  def sample(self, batch_size: int, env=None):
    """`env` accepted for SB3 signature-compat; obs are stored normalized."""
    upper = self.slots if self.full else self.pos
    s = th.randint(0, upper, (batch_size,), device=self.device)
    e = th.randint(0, self.n_envs, (batch_size,), device=self.device)
    if self.bootstrap_on_timeout:
      dones_eff = (self.dones[s, e] * (1.0 - self.timeouts[s, e])).reshape(-1, 1)
    else:                                    # a timeout is TERMINAL (margin modes)
      dones_eff = self.dones[s, e].reshape(-1, 1)
    if self.store_l:
      return ReachAvoidReplayBufferSamples(
        observations=self.observations[s, e],
        actions=self.actions[s, e],
        next_observations=self.next_observations[s, e],
        dones=dones_eff,
        rewards=self.rewards[s, e].reshape(-1, 1),
        l_x=self.l_x[s, e].reshape(-1, 1),
      )
    return ReplayBufferSamples(
      observations=self.observations[s, e],
      actions=self.actions[s, e],
      next_observations=self.next_observations[s, e],
      dones=dones_eff,
      rewards=self.rewards[s, e].reshape(-1, 1),
    )
