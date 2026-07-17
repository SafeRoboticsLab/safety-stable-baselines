"""Unit tests for the safety / reach-avoid Bellman backups.

These target the exact recursions in safety_sb3/safety_buffers.py:

  avoid:        v_to_go = min(g, V')            anchor = g
  reach-avoid:  v_to_go = min(g, max(l, V'))    anchor = min(l, g)
  blend:        target  = nt * ((1 - gamma)*anchor + gamma*v_to_go)
                          + (1 - nt) * terminal

and the places where the campaign found real bugs: terminal anchoring,
timeout bootstrap, reach banking, and the anchor itself — the two problems
take DIFFERENT anchors, and using the avoid anchor (g) for reach-avoid makes
safe loitering a win (see test_reach_avoid_safe_loiter_is_negative).

The torch buffers in tensor_buffers.py are twins of these and are checked
for exact parity, since the tensor path is what runs at scale.
"""
import gymnasium as gym
import numpy as np
import torch as th

from safety_sb3 import backups
from safety_sb3.safety_buffers import ReachAvoidRolloutBuffer, SafetyRolloutBuffer
from safety_sb3.tensor_buffers import TensorReachAvoidRolloutBuffer

OBS = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
ACT = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)


def _mk(buf_cls, T, gamma=0.9, lam=1.0, **kw):
  buf = buf_cls(T, OBS, ACT, device="cpu", gae_lambda=lam, gamma=gamma,
                n_envs=1, **kw)
  buf.reset()
  for _ in range(T):
    buf.add(np.zeros((1, 2), np.float32), np.zeros((1, 1), np.float32),
            np.zeros(1, np.float32), np.zeros(1, np.float32),
            th.zeros(1), th.zeros(1))
  return buf


def test_avoid_terminal_anchor_is_g():
  """A terminating step must return exactly g — not a bootstrap."""
  buf = _mk(SafetyRolloutBuffer, T=1)
  buf.rewards[0] = np.array([[-2.5]], np.float32).reshape(1)
  buf.values[0] = np.array([7.0], np.float32)     # critic garbage
  buf.compute_returns_and_advantage(th.tensor([[99.0]]), np.array([True]))
  assert np.allclose(buf.returns[0], -2.5), buf.returns[0]


def test_avoid_bootstrap_on_timeout():
  """A non-terminal final step blends g with the bootstrap value."""
  gamma = 0.9
  buf = _mk(SafetyRolloutBuffer, T=1, gamma=gamma)
  g, vboot = 2.0, 0.5
  buf.rewards[0] = np.full(1, g, np.float32)
  buf.values[0] = np.zeros(1, np.float32)
  buf.compute_returns_and_advantage(th.tensor([[vboot]]), np.array([False]))
  expect = (1 - gamma) * g + gamma * min(g, vboot)
  assert np.allclose(buf.returns[0], expect, atol=1e-5), buf.returns[0]


def test_avoid_true_values_are_a_fixed_point():
  """A critic at the safety-Bellman fixed point yields zero advantages,
  and that fixed point carries the worst future g back to the start."""
  gamma = 0.999
  T = 4
  buf = _mk(SafetyRolloutBuffer, T=T, gamma=gamma)
  gs = np.array([3.0, 2.0, -1.0, 2.0], np.float32)
  buf.rewards[:] = gs.reshape(T, 1)
  # V(s) = (1-gamma) g + gamma min(g, V(s')), terminal V = g
  vs = np.empty(T, np.float32)
  vs[T - 1] = gs[T - 1]
  for t in range(T - 2, -1, -1):
    vs[t] = (1 - gamma) * gs[t] + gamma * min(gs[t], vs[t + 1])
  buf.values[:] = vs.reshape(T, 1)
  buf.compute_returns_and_advantage(th.tensor([[5.0]]), np.array([True]))
  assert np.allclose(buf.advantages, 0.0, atol=1e-5), buf.advantages
  assert buf.returns[0] < 0, buf.returns[0]       # dragged down by the -1


def test_reach_avoid_anchor_is_min_l_g():
  """The (1-gamma) blend anchor is min(l, g) — Gameplay Filters eq. 6a.

  The anchor is the terminate-now payoff; for reach-avoid that needs BOTH
  in-target and safe. Using g here is the avoid problem's anchor.
  """
  gamma = 0.9
  g, l, vboot = 2.0, -3.0, 1.0
  buf = _mk(ReachAvoidRolloutBuffer, T=1, gamma=gamma)
  buf.rewards[0] = np.full(1, g, np.float32)
  buf.l_x[0] = np.full(1, l, np.float32)
  buf.values[0] = np.zeros(1, np.float32)
  buf.compute_returns_and_advantage(th.tensor([[vboot]]), np.array([False]))
  expect = (1 - gamma) * min(l, g) + gamma * min(g, max(l, vboot))
  assert np.allclose(buf.returns[0], expect, atol=1e-5), buf.returns[0]
  wrong = (1 - gamma) * g + gamma * min(g, max(l, vboot))   # the g anchor
  assert not np.allclose(buf.returns[0], wrong, atol=1e-3)


def test_reach_avoid_safe_loiter_is_negative():
  """Safe forever but never on target must be a LOSS.

  The reach-avoid value of a trajectory is max_t min(l_t, min_{s<=t} g_s);
  for one that stays safe (g > 0) and never reaches (l < 0) that equals
  max_t l_t < 0. So a critic holding V = l is the fixed point and must see
  zero advantages. The g-anchored backup fixes at V = g > 0 instead —
  scoring 'never reach' as a win, i.e. the loiter optimum.
  """
  gamma, T = 0.99, 4
  g, l = 2.0, -3.0
  buf = _mk(ReachAvoidRolloutBuffer, T=T, gamma=gamma)
  buf.rewards[:] = np.full((T, 1), g, np.float32)
  buf.l_x[:] = np.full((T, 1), l, np.float32)
  buf.values[:] = np.full((T, 1), l, np.float32)      # V = l everywhere
  buf.compute_returns_and_advantage(th.tensor([[l]]), np.array([False]))
  assert np.allclose(buf.advantages, 0.0, atol=1e-5), buf.advantages
  assert np.allclose(buf.returns, l, atol=1e-5), buf.returns


def test_reach_avoid_terminal_type():
  """terminal_type picks the terminal target: min(l, g) (default) or g."""
  g, l = 2.0, -3.0
  for tt, expect in (("all", min(l, g)), ("g", g)):
    buf = _mk(ReachAvoidRolloutBuffer, T=1, terminal_type=tt)
    buf.rewards[0] = np.full(1, g, np.float32)
    buf.l_x[0] = np.full(1, l, np.float32)
    buf.values[0] = np.zeros(1, np.float32)
    buf.compute_returns_and_advantage(th.tensor([[99.0]]), np.array([True]))
    assert np.allclose(buf.returns[0], expect, atol=1e-5), (tt, buf.returns[0])


def test_reach_avoid_banks_l():
  """Reaching the target (l > 0) banks value even if the continuation is bad."""
  gamma = 0.99
  g, l, vboot = 2.0, 1.5, -3.0
  buf = _mk(ReachAvoidRolloutBuffer, T=1, gamma=gamma)
  buf.rewards[0] = np.full(1, g, np.float32)
  buf.l_x[0] = np.full(1, l, np.float32)   # on target
  buf.values[0] = np.zeros(1, np.float32)
  buf.compute_returns_and_advantage(th.tensor([[vboot]]), np.array([False]))
  ra = buf.returns[0].item()
  # avoid twin on identical g sees only the bad future
  buf2 = _mk(SafetyRolloutBuffer, T=1, gamma=gamma)
  buf2.rewards[0] = np.full(1, g, np.float32)
  buf2.values[0] = np.zeros(1, np.float32)
  buf2.compute_returns_and_advantage(th.tensor([[vboot]]), np.array([False]))
  assert ra > 0 > buf2.returns[0].item(), (ra, buf2.returns[0])


def test_reach_avoid_g_still_caps():
  """Banking cannot exceed the safety margin: min(g, max(l, V'))."""
  buf = _mk(ReachAvoidRolloutBuffer, T=1, gamma=0.9)
  buf.rewards[0] = np.full(1, 0.3, np.float32)    # tight g
  buf.l_x[0] = np.full(1, 3.0, np.float32)        # huge bank
  buf.values[0] = np.zeros(1, np.float32)
  buf.compute_returns_and_advantage(th.tensor([[0.0]]), np.array([False]))
  assert buf.returns[0] <= 0.3 + 1e-5, buf.returns[0]


def test_tensor_reach_avoid_matches_numpy():
  """The torch RA buffer must be the numpy buffer's exact twin.

  The tensor path is what runs at scale, so a divergence here is a
  divergence in every large run.
  """
  gamma, T = 0.99, 4
  gs = np.array([3.0, 2.0, -1.0, 2.0], np.float32)
  ls = np.array([-2.0, 1.0, -0.5, -3.0], np.float32)
  vs = np.array([0.4, -0.2, 0.7, 0.1], np.float32)
  boot, done = 0.25, False

  npb = _mk(ReachAvoidRolloutBuffer, T=T, gamma=gamma)
  npb.rewards[:] = gs.reshape(T, 1)
  npb.l_x[:] = ls.reshape(T, 1)
  npb.values[:] = vs.reshape(T, 1)
  npb.compute_returns_and_advantage(th.tensor([[boot]]), np.array([done]))

  tb = TensorReachAvoidRolloutBuffer(T, OBS, ACT, device="cpu", gae_lambda=1.0,
                                     gamma=gamma, n_envs=1)
  tb.reset()
  for t in range(T):
    tb.add(th.zeros(1, 2), th.zeros(1, 1), th.tensor([gs[t]]),
           th.zeros(1), th.tensor([vs[t]]), th.zeros(1))
  tb.l_x[:] = th.tensor(ls).reshape(T, 1)
  tb.compute_returns_and_advantage(th.tensor([boot]), th.tensor([float(done)]))

  assert np.allclose(npb.returns.ravel(), tb.returns.ravel(), atol=1e-5), (
    npb.returns.ravel(), tb.returns.ravel())


# --- the operators themselves (safety_sb3.backups) --------------------------

def test_backups_numpy_and_torch_agree():
  """One definition, two array types — they must not drift apart.

  v0.1.0's bug survived because the backup was re-implemented at four call
  sites and they diverged; this pins the shared definition to both backends.
  """
  gamma = 0.99
  g = np.array([2.0, -1.0, 0.5], np.float32)
  l = np.array([-3.0, 1.0, 0.2], np.float32)
  v = np.array([0.4, 0.9, -2.0], np.float32)
  nd = np.array([1.0, 0.0, 1.0], np.float32)

  a = backups.avoid_target(g, v, nd, gamma)
  b = backups.avoid_target(th.tensor(g), th.tensor(v), th.tensor(nd), gamma)
  assert np.allclose(a, b.numpy(), atol=1e-6), (a, b)

  a = backups.reach_avoid_target(g, l, v, nd, gamma)
  b = backups.reach_avoid_target(th.tensor(g), th.tensor(l), th.tensor(v),
                                 th.tensor(nd), gamma)
  assert np.allclose(a, b.numpy(), atol=1e-6), (a, b)


def test_constant_l_cannot_express_avoid():
  """No constant l reduces the reach-avoid operator to the avoid operator.

  Executable form of the argument in safety_sb3.backups: the reduction needs
  min(l,g)=g (=> l >= g) AND max(l,V')=V' (=> l <= V'), and V' <= g, so it
  demands l >= g >= V' >= l. Each constant buys one half and loses the other.
  """
  gamma = 0.99
  g = np.float32(2.0)     # permanently safe
  nd = np.float32(1.0)    # never terminates

  # l_neg buys the recursion, destroys the anchor: V == l, whatever g is.
  # => {V >= 0} is EMPTY even though the state is safe forever.
  l_neg = np.float32(-10.0)
  assert np.isclose(
    backups.reach_avoid_target(g, l_neg, l_neg, nd, gamma), l_neg)

  # l_pos buys the anchor, destroys the recursion: V == g even when the
  # continuation is catastrophic -- max(l, V') clips the future away.
  l_pos, doomed = np.float32(10.0), np.float32(-5.0)
  assert np.isclose(
    backups.reach_avoid_target(g, l_pos, doomed, nd, gamma), g)

  # ...whereas the AVOID operator does propagate that doomed future.
  assert backups.avoid_target(g, doomed, nd, gamma) < 0


def test_mode_dispatch_and_validation():
  gamma = 0.99
  g, l, v, nd = (np.float32(x) for x in (2.0, -3.0, 0.5, 1.0))
  assert np.isclose(backups.target(backups.AVOID, g, v, nd, gamma),
                    backups.avoid_target(g, v, nd, gamma))
  assert np.isclose(backups.target(backups.REACH_AVOID, g, v, nd, gamma, l=l),
                    backups.reach_avoid_target(g, l, v, nd, gamma))
  for bad in (lambda: backups.target("nonsense", g, v, nd, gamma),
              lambda: backups.target(backups.REACH_AVOID, g, v, nd, gamma),
              lambda: backups.reach_avoid_target(g, l, v, nd, gamma,
                                                 terminal_type="bogus")):
    try:
      bad()
      raise AssertionError("expected ValueError")
    except ValueError:
      pass
