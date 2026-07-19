"""Periodic SAFE-RATE / SUCCESS-RATE evaluation for the tensor training stack.

A standalone SB3 :class:`~stable_baselines3.common.callbacks.BaseCallback` that,
every ``eval_freq`` env-steps, rolls out the current policy on a GPU-resident
eval env and records three scalars to the SB3 logger (which syncs to wandb):

* ``eval/safe_rate``    — fraction of episodes that NEVER entered the failure
  set (never saw ``g < 0``).
* ``eval/success_rate`` — for **reach-avoid** tasks: fraction that reached the
  target (ever ``l_x >= 0``) AND stayed safe; for **avoid-only** tasks it equals
  ``safe_rate`` (the reach term does not apply).
* ``eval/ep_len_mean``  — mean length of the tracked first episodes.

The definitions mirror the reference audit target ``safe_adaptation_dev``
(``utils/eval.py`` ``evaluate_zero_sum``: ``safe_rate = mean(results != -1)``,
``ra_rate = mean(results == 1)``; result codes success=1 / failure=-1 /
timeout=0) and the leaderboard eval in :mod:`safety_sb3.isaacs`
(``_eval_pair_vec``: safe = never ``g < 0``, reached = ever ``l_x >= 0``).

Rollout accounting is parallel: the eval env runs ``num_envs`` episodes at once,
we track each env's FIRST episode (safe until its first ``g < 0``, reached if it
ever hit ``l_x >= 0`` before that env's first ``done``), and accumulate
``ceil(n_rollouts / num_envs)`` batches so at least ``n_rollouts`` first-episodes
are scored. Everything stays on device — the eval env speaks the tensor API
``obs, g, dones, timeouts, l_x = env.step_tensor(actions)``.
"""

from __future__ import annotations

import math
import warnings

import torch as th
from stable_baselines3.common.callbacks import BaseCallback


class SafeSuccessRateEvalCallback(BaseCallback):
  """Periodic safe-rate / success-rate eval on a :class:`TensorVecEnv`.

  :param eval_env: a GPU-resident tensor env (``safety_sb3.tensor_env``
    ``TensorVecEnv``, optionally wrapped in ``TensorVecNormalize``). Must expose
    ``num_envs``, ``reset() -> obs`` and
    ``step_tensor(actions) -> (obs, g, dones, timeouts, l_x)`` (all torch
    tensors on ``eval_env.device``). Should be SEPARATE from the training env.
  :param n_rollouts: minimum number of first-episodes to score per eval. The
    callback runs ``ceil(n_rollouts / num_envs)`` parallel batches.
  :param eval_freq: run an eval every ``eval_freq`` env-steps (compared against
    ``self.num_timesteps``).
  :param reach_avoid: ``True`` for a reach-avoid task (success = reached AND
    safe); ``False`` for avoid-only (success == safe_rate).
  :param max_ep_steps: hard cap on rollout length (guards against episodes that
    never terminate). A capped, never-violated episode counts as safe.
  :param deterministic: use deterministic policy actions during eval.
  """

  def __init__(
    self,
    eval_env,
    n_rollouts: int = 100,
    eval_freq: int = 1_000_000,
    reach_avoid: bool = True,
    max_ep_steps: int = 400,
    deterministic: bool = True,
    verbose: int = 0,
  ) -> None:
    super().__init__(verbose)
    self.eval_env = eval_env
    self.n_rollouts = int(n_rollouts)
    self.eval_freq = int(eval_freq)
    self.reach_avoid = bool(reach_avoid)
    self.max_ep_steps = int(max_ep_steps)
    self.deterministic = bool(deterministic)
    # Fire the first eval once `eval_freq` steps have elapsed.
    self._next_eval = self.eval_freq

  # --- SB3 hook ------------------------------------------------------------
  def _on_step(self) -> bool:
    if self.eval_freq > 0 and self.num_timesteps >= self._next_eval:
      # Advance past any missed windows so a batched step-increment (tensor
      # path adds `num_envs` per step) does not re-fire immediately.
      while self.num_timesteps >= self._next_eval:
        self._next_eval += self.eval_freq
      self._safe_eval()
    return True

  def _safe_eval(self) -> None:
    """Run one eval, never letting a bad env/policy shape crash training."""
    try:
      safe_rate, success_rate, ep_len_mean, n = self._run_eval()
    except Exception as exc:  # noqa: BLE001 -- eval must never kill training
      warnings.warn(
        f"[SafeSuccessRateEvalCallback] eval at step {self.num_timesteps} "
        f"skipped: {type(exc).__name__}: {exc}"
      )
      return
    self.logger.record("eval/safe_rate", safe_rate)
    self.logger.record("eval/success_rate", success_rate)
    self.logger.record("eval/ep_len_mean", ep_len_mean)
    if self.verbose:
      tag = "reach-avoid" if self.reach_avoid else "avoid-only"
      print(
        f"[eval @ {self.num_timesteps}] ({tag}, n={n}) "
        f"safe_rate={safe_rate:.3f} success_rate={success_rate:.3f} "
        f"ep_len_mean={ep_len_mean:.1f}"
      )

  # --- rollout accounting --------------------------------------------------
  @th.no_grad()
  def _run_eval(self):
    env = self.eval_env
    n_env = int(getattr(env, "num_envs"))
    device = getattr(env, "device", self.model.device)
    policy = self.model.policy
    two_player = hasattr(policy, "dstb_actor")
    n_batches = max(1, math.ceil(self.n_rollouts / max(n_env, 1)))

    # Freeze the observation normalizer (if any) and switch the policy to eval
    # mode; both are restored in `finally`.
    prev_training = getattr(env, "training", None)
    if prev_training is not None:
      env.training = False
    prev_policy_training = getattr(policy, "training", None)
    policy.set_training_mode(False)

    safe_hits = 0
    succ_hits = 0
    len_sum = 0.0
    total = 0
    try:
      for _ in range(n_batches):
        ep_safe, ep_reached, ep_len = self._rollout_batch(
          env, policy, two_player, n_env, device
        )
        succ = ep_safe & ep_reached if self.reach_avoid else ep_safe
        safe_hits += int(ep_safe.sum().item())
        succ_hits += int(succ.sum().item())
        len_sum += float(ep_len.sum().item())
        total += n_env
    finally:
      if prev_training is not None:
        env.training = prev_training
      if prev_policy_training:
        policy.set_training_mode(True)

    denom = max(total, 1)
    return safe_hits / denom, succ_hits / denom, len_sum / denom, total

  def _rollout_batch(self, env, policy, two_player, n_env, device):
    """One parallel batch of `num_envs` first-episodes.

    Returns per-env (ep_safe, ep_reached, ep_len) for THIS env's first episode;
    an env stops being tracked at its first `done` (the env auto-resets, but we
    ignore everything after the first termination)."""
    obs = env.reset()
    ep_safe = th.ones(n_env, dtype=th.bool, device=device)
    ep_reached = th.zeros(n_env, dtype=th.bool, device=device)
    done_once = th.zeros(n_env, dtype=th.bool, device=device)
    ep_len = th.zeros(n_env, dtype=th.float32, device=device)

    for _ in range(self.max_ep_steps):
      actions = self._policy_actions(policy, obs, two_player)
      obs, g, dones, _timeouts, l_x = env.step_tensor(actions)
      active = ~done_once  # envs still on their first episode this step
      g = th.as_tensor(g, device=device).reshape(n_env)
      l_x = th.as_tensor(l_x, device=device).reshape(n_env)
      dones = th.as_tensor(dones, device=device).reshape(n_env).bool()
      ep_safe &= ~(active & (g < 0))
      ep_reached |= active & (l_x >= 0)
      ep_len += active.float()  # counts the terminating step too
      done_once |= active & dones
      if bool(done_once.all()):
        break
    return ep_safe, ep_reached, ep_len

  def _policy_actions(self, policy, obs, two_player):
    """Deterministic policy action for the tensor env.

    Single-player: the control actor's output. Two-player (has ``dstb_actor``):
    the control + disturbance actors composed into the env's
    ``ctrl_dim + dstb_dim`` action, mirroring ``isaacs._tensor_policy_actions``.
    Actors output in ``[-1, 1]`` and the tensor env's action space is
    ``[-1, 1]`` (the env clamps), so no unscaling is needed on this path. The
    eval env already returns normalized observations when it is a
    ``TensorVecNormalize`` wrapper, so ``obs`` is fed to the actor as-is."""
    ctrl = policy.actor(obs, deterministic=self.deterministic)
    if not two_player:
      return ctrl
    dstb = policy.dstb_actor(obs, deterministic=self.deterministic)
    return th.cat([ctrl, dstb], dim=1)


# ---------------------------------------------------------------------------
# Self-test: a 1-D double-integrator toy (g = 1 - |x|, l = 0.2 - |x - 0.5|),
# a short ReachAvoidSAC / GameplaySAC learn() with the callback at a small
# eval_freq, asserting eval/safe_rate and eval/success_rate land in [0, 1].
# ---------------------------------------------------------------------------
if __name__ == "__main__":
  import os
  import sys

  import torch as _th
  from gymnasium import spaces

  sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

  from safety_sb3 import GameplaySAC, ReachAvoidSAC  # noqa: E402
  from safety_sb3.tensor_env import TensorVecEnv  # noqa: E402

  DEV = "cpu"
  DT = 0.05
  TIMEOUT = 200

  class DoubleIntegratorEnv(TensorVecEnv):
    """1-D double integrator; avoid g = 1 - |x|, reach l = 0.2 - |x - 0.5|."""

    def __init__(self, num_envs=64, device=DEV, act_dim=1):
      obs_space = spaces.Box(-10, 10, shape=(2,), dtype="float32")
      act_space = spaces.Box(-1, 1, shape=(act_dim,), dtype="float32")
      super().__init__(num_envs, obs_space, act_space, device)
      self.act_dim = act_dim
      self.x = _th.zeros(num_envs, device=device)
      self.v = _th.zeros(num_envs, device=device)
      self.t = _th.zeros(num_envs, device=device)

    def _obs(self):
      return _th.stack([self.x, self.v], dim=1)

    def _reset_ids(self, ids):
      n = int(ids.sum())
      self.x[ids] = _th.rand(n, device=self.device) * 1.6 - 0.8
      self.v[ids] = _th.rand(n, device=self.device) * 1.0 - 0.5
      self.t[ids] = 0.0

    def reset(self):
      self._reset_ids(_th.ones(self.num_envs, dtype=_th.bool, device=self.device))
      return self._obs()

    def step_tensor(self, actions):
      a_c = actions[:, 0].clamp(-1, 1)
      a_d = actions[:, 1].clamp(-1, 1) if self.act_dim > 1 else 0.0
      self.x = self.x + self.v * DT
      self.v = self.v + (3.0 * a_c + 1.5 * a_d) * DT
      self.t = self.t + 1
      g = 1.0 - self.x.abs()
      l = 0.2 - (self.x - 0.5).abs()
      terminated = g < 0
      truncated = (self.t >= TIMEOUT) & ~terminated
      dones = terminated | truncated
      if bool(dones.any()):
        self._reset_ids(dones)
      return self._obs(), g, dones, truncated, l

  def _check(cb, reach_avoid):
    log = cb.model.logger.name_to_value
    sr, uc, el = log.get("eval/safe_rate"), log.get("eval/success_rate"), log.get("eval/ep_len_mean")
    assert sr is not None and uc is not None and el is not None, (
      f"missing eval scalars: {log}")
    assert 0.0 <= sr <= 1.0, f"safe_rate out of [0,1]: {sr}"
    assert 0.0 <= uc <= 1.0, f"success_rate out of [0,1]: {uc}"
    if not reach_avoid:
      assert abs(sr - uc) < 1e-9, f"avoid-only: success must == safe ({sr} vs {uc})"
    else:
      assert uc <= sr + 1e-9, f"reach-avoid: success ({uc}) must be <= safe ({sr})"
    assert el > 0, f"ep_len_mean must be positive: {el}"
    print(f"  -> safe_rate={sr:.3f} success_rate={uc:.3f} ep_len_mean={el:.1f} [ok]")

  print("[1] single-player ReachAvoidSAC + reach-avoid eval callback")
  env = DoubleIntegratorEnv(num_envs=64)
  eval_env = DoubleIntegratorEnv(num_envs=32)
  cb = SafeSuccessRateEvalCallback(
    eval_env, n_rollouts=50, eval_freq=3_000, reach_avoid=True,
    max_ep_steps=250, verbose=1)
  model = ReachAvoidSAC(
    "MlpPolicy", env, buffer_size=20_000, batch_size=256, learning_starts=500,
    train_freq=1, gradient_steps=4, gamma=0.95, learning_rate=3e-4,
    policy_kwargs=dict(net_arch=[64, 64]), verbose=0, device=DEV, seed=0,
    gamma_anneal=False)
  model.learn(total_timesteps=8_000, log_interval=None, callback=cb)
  _check(cb, reach_avoid=True)

  print("[2] single-player, avoid-only flag (success must == safe_rate)")
  eval_env2 = DoubleIntegratorEnv(num_envs=32)
  cb2 = SafeSuccessRateEvalCallback(
    eval_env2, n_rollouts=40, eval_freq=3_000, reach_avoid=False,
    max_ep_steps=250, verbose=1)
  model.learn(total_timesteps=4_000, log_interval=None, callback=cb2,
              reset_num_timesteps=False)
  _check(cb2, reach_avoid=False)

  print("[3] two-player GameplaySAC (composes ctrl+dstb) + reach-avoid eval")
  tp_env = DoubleIntegratorEnv(num_envs=64, act_dim=2)
  tp_eval = DoubleIntegratorEnv(num_envs=32, act_dim=2)
  cb3 = SafeSuccessRateEvalCallback(
    tp_eval, n_rollouts=40, eval_freq=3_000, reach_avoid=True,
    max_ep_steps=250, verbose=1)
  tp_model = GameplaySAC(
    "MlpPolicy", tp_env, ctrl_action_dim=1, buffer_size=20_000, batch_size=256,
    learning_starts=500, train_freq=1, gradient_steps=4, gamma=0.95,
    learning_rate=3e-4, policy_kwargs=dict(net_arch=[64, 64]), verbose=0,
    device=DEV, seed=0, gamma_anneal=False)
  tp_model.learn(total_timesteps=8_000, log_interval=None, callback=cb3)
  _check(cb3, reach_avoid=True)

  print("SafeSuccessRateEvalCallback SELF-TEST PASSED")
