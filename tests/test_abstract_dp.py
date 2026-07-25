"""Abstract-DP structure: one update loop, the backup chosen by a mode.

Pins the properties the SAC/DQN de-duplication is supposed to buy:

  * every learner's TD target comes from ``safety_sb3.backups`` -- including
    ``SafetyDQN1P``, which used to inline its own (algebraically identical) avoid
    backup, and the cumulative (standard RL) operator, which makes plain
    sum-of-rewards RL a mode of this library rather than a fork of it;
  * the single-player SAC learners are ``_MODE`` specializations of ONE
    ``AbstractSAC1P.train()``. The hand-copied reach-avoid ``train()`` they
    replace had silently dropped the ``min_alpha``/``max_alpha`` clamp -- see
    ``test_reach_avoid_sac_clamps_alpha``, the regression guard for that drift;
  * the mode is selectable per instance (``mode=``), not only per class.

  python -m pytest tests/test_abstract_dp.py
"""
import gymnasium as gym
import numpy as np
import torch as th

from safety_sb3 import (AbstractSAC1P, CumulativeSAC1P, ReachAvoidSAC1P,
                        SafetyDQN1P, SafetySAC1P, backups)
from safety_sb3.buffers_replay import ReachAvoidReplayBuffer
from safety_sb3.sac_base import AbstractSAC as _AbstractSAC
from safety_sb3.sac_1p import AbstractSAC1P as _AbstractSAC1P
from safety_sb3.sac_2p import AbstractSAC2P as _AbstractSAC2P

DEV = "cpu"


# --- envs -------------------------------------------------------------------

class _Toy(gym.Env):
  """1-D point: g = 1 - |x| (avoid), l = 0.2 - |x - 0.5| (target), r = -|x|."""
  observation_space = gym.spaces.Box(-3, 3, (2,), dtype=np.float32)
  action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

  def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    self.x = float(self.np_random.uniform(-0.5, 0.5))
    self.t = 0
    return self._obs(), {"l_x": self._l()}

  def _obs(self):
    return np.array([self.x, float(self.t) / 40.0], np.float32)

  def _l(self):
    return float(0.2 - abs(self.x - 0.5))

  def step(self, action):
    self.x += 0.1 * float(np.asarray(action).reshape(-1)[0])
    self.t += 1
    g = 1.0 - abs(self.x)
    return (self._obs(), float(g), bool(g < 0.0), self.t >= 40,
            {"l_x": self._l()})


class _ToyReward(_Toy):
  """Same dynamics, but ``reward`` is a REWARD (cumulative mode)."""

  def step(self, action):
    obs, _, term, trunc, info = super().step(action)
    return obs, float(1.0 - abs(self.x)), term, trunc, info


class _ToyDiscrete(gym.Env):
  observation_space = gym.spaces.Box(-3, 3, (2,), dtype=np.float32)
  action_space = gym.spaces.Discrete(3)

  def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    self.x, self.t = float(self.np_random.uniform(-0.5, 0.5)), 0
    return np.array([self.x, 0.0], np.float32), {}

  def step(self, action):
    self.x += 0.1 * (int(action) - 1)
    self.t += 1
    g = 1.0 - abs(self.x)
    return (np.array([self.x, float(action) - 1.0], np.float32), float(g),
            bool(g < 0.0), self.t >= 40, {})


def _sac(cls, env=None, steps=200, **kw):
  env = _Toy() if env is None else env
  env.reset(seed=0)
  m = cls("MlpPolicy", env, seed=0, device=DEV, learning_starts=32,
          batch_size=16, train_freq=1, gradient_steps=1, buffer_size=1000,
          policy_kwargs=dict(net_arch=[16, 16]), **kw)
  m.learn(total_timesteps=steps, log_interval=1000)
  return m


# --- the cumulative operator ------------------------------------------------

def test_cumulative_is_a_mode():
  assert backups.CUMULATIVE in backups.MODES
  assert backups.CUMULATIVE not in backups.SAFETY_MODES
  r = np.array([1.0, -2.0, 0.5])
  v = np.array([3.0, 3.0, 3.0])
  nt = np.array([1.0, 1.0, 0.0])
  want = np.array([1.0 + 0.9 * 3.0, -2.0 + 0.9 * 3.0, 0.5])  # terminal -> r
  got = backups.cumulative_target(r, v, nt, 0.9)
  assert np.allclose(got, want), (got, want)
  # dispatcher agrees, needs no l, and is torch/numpy agnostic
  assert np.allclose(backups.target(backups.CUMULATIVE, r, v, nt, 0.9), want)
  t = backups.target(backups.CUMULATIVE, th.as_tensor(r), th.as_tensor(v),
                     th.as_tensor(nt), 0.9)
  assert np.allclose(t.numpy(), want)


def test_cumulative_is_not_a_safety_operator():
  """Sanity that the modes are genuinely different fixed points: with a
  positive constant margin/reward the avoid value saturates at g while the
  cumulative value accumulates."""
  g = np.array([1.0])
  v = np.array([5.0])
  nt = np.array([1.0])
  assert backups.avoid_target(g, v, nt, 0.9) == 1.0            # min(g, V') = g
  assert backups.cumulative_target(g, v, nt, 0.9) == 1.0 + 4.5


# --- DQN now routes through backups -----------------------------------------

def test_dqn_avoid_backup_matches_the_old_inline_algebra():
  """SafetyDQN1P used to inline ``(1 - g*nt)*gs + g*nt*min(gs, V')``."""
  rng = np.random.default_rng(0)
  gs, vn = rng.normal(size=256), rng.normal(size=256)
  nt = (rng.random(256) > 0.3).astype(float)
  for gamma in (0.0, 0.5, 0.99, 1.0):
    old = (1.0 - gamma * nt) * gs + gamma * nt * np.minimum(gs, vn)
    new = backups.target(backups.AVOID, gs, vn, nt, gamma)
    assert np.allclose(old, new, atol=1e-12), gamma


def test_dqn_modes():
  assert SafetyDQN1P._MODE == backups.AVOID
  m = SafetyDQN1P("MlpPolicy", _ToyDiscrete(), seed=0, device=DEV,
                learning_starts=32, batch_size=16, train_freq=1,
                buffer_size=1000, policy_kwargs=dict(net_arch=[16, 16]),
                mode=backups.CUMULATIVE)
  assert m._MODE == backups.CUMULATIVE
  m.learn(total_timesteps=200, log_interval=1000)
  try:
    SafetyDQN1P("MlpPolicy", _ToyDiscrete(), mode=backups.REACH_AVOID, device=DEV)
    raise AssertionError("expected ValueError (no l in the DQN buffer)")
  except ValueError:
    pass


# --- one train(), specialized by _MODE --------------------------------------

def test_sac_family_shares_one_train():
  """The de-duplication itself: no concrete SAC learner owns a train() body.

  Every single-player mode gets ``AbstractSAC1P.train`` verbatim. The mode is a
  class attribute, never a fork of the loop.
  """
  for cls in (SafetySAC1P, ReachAvoidSAC1P, CumulativeSAC1P):
    assert issubclass(cls, _AbstractSAC1P), cls
    assert cls.train is _AbstractSAC1P.train, f"{cls.__name__} re-implements train()"
  assert SafetySAC1P._MODE == backups.AVOID
  assert ReachAvoidSAC1P._MODE == backups.REACH_AVOID
  assert CumulativeSAC1P._MODE == backups.CUMULATIVE


def test_train_lives_on_the_players_axis_not_the_shared_base():
  """v0.4.0 structure: AbstractSAC holds everything the two player counts
  SHARE and deliberately no train(); the 1P and 2P loops are siblings under it,
  not one derived from the other."""
  # the shared base defines train() only to REFUSE -- never to compute
  import pytest
  m = object.__new__(_AbstractSAC)
  with pytest.raises(NotImplementedError, match="no update loop"):
    m.train(1, 1)
  for loop in (_AbstractSAC1P, _AbstractSAC2P):
    assert issubclass(loop, _AbstractSAC), loop
    assert "train" in vars(loop), loop
  assert not issubclass(_AbstractSAC2P, _AbstractSAC1P)
  assert not issubclass(_AbstractSAC1P, _AbstractSAC2P)


def test_bellman_target_dispatches_on_mode():
  """_bellman_target is exactly backups.target for the instance's mode."""
  m = _sac(ReachAvoidSAC1P)
  rd = m.replay_buffer.sample(8)
  vn = th.randn(8, 1)
  assert th.allclose(
    m._bellman_target(rd, vn),
    backups.reach_avoid_target(rd.rewards, rd.l_x, vn, 1.0 - rd.dones,
                               m.gamma, m.terminal_type))
  ms = _sac(SafetySAC1P)
  rds = ms.replay_buffer.sample(8)
  assert th.allclose(
    ms._bellman_target(rds, vn),
    backups.avoid_target(rds.rewards, vn, 1.0 - rds.dones, ms.gamma))


def test_reach_avoid_sac_clamps_alpha():
  """REGRESSION: the hand-copied ReachAvoidSAC1P.train() omitted
  _clamp_entropy_temps(), so min_alpha/max_alpha silently did not apply on the
  reach-avoid path. One shared train() makes that unrepresentable.

  1500 steps is deliberate: alpha starts at 1.0 and decays, so a short run
  passes even unclamped (pre-refactor it reached 0.947 by step 300, still
  inside the window, but 0.763 -- clearly below the floor -- by step 1500).
  """
  for cls in (SafetySAC1P, ReachAvoidSAC1P):
    m = _sac(cls, min_alpha=0.9, max_alpha=0.95, steps=1500)
    alpha = float(th.exp(m.log_ent_coef.detach()))
    assert 0.9 - 1e-6 <= alpha <= 0.95 + 1e-6, (cls.__name__, alpha)


# --- mode as a constructor argument -----------------------------------------

def test_mode_selectable_at_init():
  # avoid class asked for reach-avoid -> gets the l-carrying buffer + RA target
  m = _sac(SafetySAC1P, mode=backups.REACH_AVOID)
  assert m._MODE == backups.REACH_AVOID
  assert isinstance(m.replay_buffer, ReachAvoidReplayBuffer)
  assert m._tensor_store_l
  rd = m.replay_buffer.sample(8)
  vn = th.randn(8, 1)
  assert th.allclose(
    m._bellman_target(rd, vn),
    backups.reach_avoid_target(rd.rewards, rd.l_x, vn, 1.0 - rd.dones,
                               m.gamma, m.terminal_type))
  # class default still wins when nothing is passed
  assert _sac(SafetySAC1P)._MODE == backups.AVOID
  try:
    _sac(SafetySAC1P, mode="bogus")
    raise AssertionError("expected ValueError")
  except ValueError:
    pass


def test_cumulative_rollout_buffer_is_sb3_gae():
  """The on-policy side unifies too: the safety buffers' backup is a mode, and
  in CUMULATIVE mode the loop reduces exactly to stock SB3 GAE."""
  from stable_baselines3.common.buffers import RolloutBuffer

  from safety_sb3.buffers_rollout import SafetyRolloutBuffer

  T, gamma, lam = 12, 0.95, 0.9
  rng = np.random.default_rng(0)
  rewards = rng.normal(size=T).astype(np.float32)
  values = rng.normal(size=T).astype(np.float32)
  starts = (rng.random(T) < 0.25).astype(np.float32)
  last_v, last_done = th.tensor([0.3]), np.array([0.0], np.float32)

  def fill(buf):
    buf.reset()
    for t in range(T):
      buf.add(np.zeros((1, 2), np.float32), np.zeros((1, 1), np.float32),
              np.array([rewards[t]]), np.array([starts[t]]),
              th.tensor([[values[t]]]), th.zeros(1))
    buf.compute_returns_and_advantage(last_v, last_done)
    return buf.advantages.copy(), buf.returns.copy()

  kw = dict(observation_space=_Toy.observation_space,
            action_space=_Toy.action_space, device=DEV, gamma=gamma,
            gae_lambda=lam, n_envs=1)
  a_ours, r_ours = fill(SafetyRolloutBuffer(T, mode=backups.CUMULATIVE, **kw))
  a_sb3, r_sb3 = fill(RolloutBuffer(T, **kw))
  assert np.allclose(a_ours, a_sb3, atol=1e-6), (a_ours, a_sb3)
  assert np.allclose(r_ours, r_sb3, atol=1e-6)
  # and the default (avoid) mode is NOT GAE -- the buffers really do differ
  a_avoid, _ = fill(SafetyRolloutBuffer(T, **kw))
  assert not np.allclose(a_avoid, a_sb3, atol=1e-3)


def test_cumulative_sac_trains():
  """Plain reward-maximizing SAC, same code path, no safety semantics."""
  m = _sac(AbstractSAC1P, env=_ToyReward(), mode=backups.CUMULATIVE, steps=600)
  assert m._MODE == backups.CUMULATIVE
  assert not isinstance(m.replay_buffer, ReachAvoidReplayBuffer)
  rd = m.replay_buffer.sample(8)
  vn = th.randn(8, 1)
  assert th.allclose(m._bellman_target(rd, vn),
                     rd.rewards + m.gamma * (1.0 - rd.dones) * vn)
  # a discounted return over a bounded-positive reward must exceed the
  # one-step reward the avoid/reach-avoid operators would cap the value at
  obs = th.zeros(1, 2)
  with th.no_grad():
    q = float(th.cat(m.critic(obs, m.actor(obs)), dim=1).min())
  assert q > 1.0, q


if __name__ == "__main__":
  import sys
  mod = sys.modules[__name__]
  for name in [n for n in dir(mod) if n.startswith("test_")]:
    getattr(mod, name)()
    print(f"[ok] {name}")
