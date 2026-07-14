"""Unit tests for the safety / reach-avoid Bellman backups.

These target the exact recursions in safety_sb3/safety_buffers.py:

  avoid:        v_to_go = min(g, V')
  reach-avoid:  v_to_go = min(g, max(l, V'))
  blend:        target  = (1 - gamma*nt) * g + gamma*nt * v_to_go

and the places where the campaign found real bugs: terminal anchoring,
timeout bootstrap, and reach banking.
"""
import gymnasium as gym
import numpy as np
import torch as th

from safety_sb3.safety_buffers import ReachAvoidRolloutBuffer, SafetyRolloutBuffer

OBS = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
ACT = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)


def _mk(buf_cls, T, gamma=0.9, lam=1.0):
  buf = buf_cls(T, OBS, ACT, device="cpu", gae_lambda=lam, gamma=gamma,
                n_envs=1)
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


def test_reach_avoid_banks_l():
  """Reaching the target (l > 0) banks value even if the continuation is bad."""
  gamma = 0.99
  buf = _mk(ReachAvoidRolloutBuffer, T=2, gamma=gamma)
  buf.rewards[:] = np.array([[2.0], [2.0]], np.float32)   # g > 0 throughout
  buf.l_x[:] = np.array([[1.5], [-1.0]], np.float32)      # l > 0 at step 0
  buf.values[:] = 0.0
  # continuation is terrible: bootstrap very negative but non-terminal chain
  buf.compute_returns_and_advantage(th.tensor([[-3.0]]), np.array([True]))
  ra = buf.returns[0].item()
  # avoid twin on identical g sees only the bad future
  buf2 = _mk(SafetyRolloutBuffer, T=2, gamma=gamma)
  buf2.rewards[:] = np.array([[2.0], [2.0]], np.float32)
  buf2.values[:] = 0.0
  buf2.compute_returns_and_advantage(th.tensor([[-3.0]]), np.array([True]))
  assert ra > buf2.returns[0].item() + 0.5, (ra, buf2.returns[0])


def test_reach_avoid_g_still_caps():
  """Banking cannot exceed the safety margin: min(g, max(l, V'))."""
  buf = _mk(ReachAvoidRolloutBuffer, T=1, gamma=0.9)
  buf.rewards[0] = np.full(1, 0.3, np.float32)    # tight g
  buf.l_x[0] = np.full(1, 3.0, np.float32)        # huge bank
  buf.values[0] = np.zeros(1, np.float32)
  buf.compute_returns_and_advantage(th.tensor([[0.0]]), np.array([False]))
  assert buf.returns[0] <= 0.3 + 1e-5, buf.returns[0]
