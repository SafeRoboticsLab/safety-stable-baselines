"""ReachAvoidMaskedPPO1P — the RAS policy_mask learner (reach-avoid-stay handover training).

Two properties pin the correctness of the masked update:

  (a) IDENTITY — with an all-ones policy_mask the masked ``train()`` must produce
      the SAME parameter update as stock ``ReachAvoidPPO1P`` on the same buffer.
      This is the graceful-degradation guarantee: when the handover gate never
      fires, RAS is byte-for-byte the proven naive-RA update.

  (b) MASK SEMANTICS — with an all-zeros policy_mask the policy pathway receives
      ZERO gradient (the masked/lander steps must never move the reach policy)
      while the value pathway still receives a NONZERO gradient (those steps DO
      train the value — the whole point of keeping them in the buffer).

Both run on the GPU-resident (tensor) path — the only path the masked buffer
supports — using the tiny bicycle5d tensor env just to construct valid models;
the rollout buffers are then filled with fixed synthetic data so the two learners
see identical inputs (no rollout RNG in the comparison).
"""
from __future__ import annotations

import os
import sys

import torch as th

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3.common.utils import configure_logger  # noqa: E402

from safety_sb3 import ReachAvoidMaskedPPO1P, ReachAvoidPPO1P  # noqa: E402
from safety_sb3.buffers_tensor import (  # noqa: E402
  TensorReachAvoidMaskedRolloutBuffer)
from safety_sb3.testing.bicycle5d_tensor import BicycleGoalTensorVec  # noqa: E402

DEV = "cpu"
T, N = 8, 8                 # buffer_size x n_envs -> total = 64 samples
COMMON = dict(n_steps=T, batch_size=T * N, n_epochs=1, seed=0,
              gamma_anneal=False, adaptive_lr=False, normalize_obs=False,
              verbose=0, device=DEV)


def _env():
  return BicycleGoalTensorVec(N, spawn="wide", device=DEV, dtype=th.float32,
                              seed=0)


def _fill(buffer, data):
  """Write fixed synthetic rollout tensors into a (masked or plain) RA buffer."""
  buffer.reset()
  buffer.observations.copy_(data["obs"])
  buffer.actions.copy_(data["act"])
  buffer._values.copy_(data["val"])
  buffer.log_probs.copy_(data["logp"])
  buffer.advantages.copy_(data["adv"])
  buffer._returns.copy_(data["ret"])
  buffer.l_x.copy_(data["l_x"])
  buffer.pos = buffer.buffer_size
  buffer.full = True


def _make_data(obs_dim, act_dim):
  g = th.Generator().manual_seed(1234)
  r = lambda *s: th.randn(*s, generator=g)          # noqa: E731
  return dict(
    obs=r(T, N, obs_dim), act=th.clamp(r(T, N, act_dim), -1, 1),
    val=r(T, N), logp=r(T, N), adv=r(T, N), ret=r(T, N), l_x=r(T, N))


def test_masked_train_matches_stock_with_ones_mask():
  """(a) all-ones mask => identical parameter update to stock ReachAvoidPPO1P."""
  env = _env()
  obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
  data = _make_data(obs_dim, act_dim)

  m_mask = ReachAvoidMaskedPPO1P("MlpPolicy", env, **COMMON)
  m_plain = ReachAvoidPPO1P("MlpPolicy", env, **COMMON)
  m_mask.set_logger(configure_logger())
  m_plain.set_logger(configure_logger())
  # the masked learner must have selected the masked buffer automatically
  assert isinstance(m_mask.rollout_buffer, TensorReachAvoidMaskedRolloutBuffer)
  # identical initial weights
  m_plain.policy.load_state_dict(m_mask.policy.state_dict())

  _fill(m_mask.rollout_buffer, data)
  _fill(m_plain.rollout_buffer, data)
  # all-ones mask (this is also the reset() default; set it explicitly)
  m_mask.rollout_buffer.policy_mask.fill_(1.0)

  th.manual_seed(0); m_mask.train()
  th.manual_seed(0); m_plain.train()

  diffs = [(k, float((a - b).abs().max()))
           for (k, a), (_, b) in zip(m_mask.policy.state_dict().items(),
                                     m_plain.policy.state_dict().items())]
  worst = max(d for _, d in diffs)
  print(f"[identity] max |param_masked - param_stock| = {worst:.3e}")
  assert worst < 1e-6, f"masked(all-ones) != stock; worst {max(diffs, key=lambda x: x[1])}"
  # sanity: the update actually moved the weights (not a trivial no-op match)
  moved = max(float((a - b).abs().max())
              for (_, a), (_, b) in zip(
                m_mask.policy.state_dict().items(),
                ReachAvoidPPO1P("MlpPolicy", _env(), **COMMON).policy
                .state_dict().items()))
  assert moved > 1e-6, "weights did not change — the test is vacuous"


def test_masked_steps_give_zero_policy_grad_nonzero_value_grad():
  """(b) all-zeros mask => zero grad on the policy head, nonzero on the value."""
  env = _env()
  obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
  data = _make_data(obs_dim, act_dim)

  m = ReachAvoidMaskedPPO1P("MlpPolicy", env, **COMMON)
  m.set_logger(configure_logger())
  _fill(m.rollout_buffer, data)
  m.rollout_buffer.policy_mask.zero_()          # every step is a hand-off step

  th.manual_seed(0)
  m.train()   # grads persist after the final optimizer.step (no zero_grad after)

  # policy pathway: mean head + log_std must have EXACTLY zero gradient
  pol = m.policy
  pg_head = pol.action_net.weight.grad
  pg_std = pol.log_std.grad
  assert pg_head is not None and float(pg_head.abs().max()) == 0.0, \
    f"policy head got nonzero grad from masked steps: {float(pg_head.abs().max())}"
  assert pg_std is None or float(pg_std.abs().max()) == 0.0, \
    "log_std got nonzero grad from masked steps"
  # value pathway: the value head must have a NONZERO gradient (masked steps
  # still train the value)
  vg_head = pol.value_net.weight.grad
  assert vg_head is not None and float(vg_head.abs().max()) > 0.0, \
    "value head got zero grad — masked steps must still train the value"
  print(f"[mask] policy-head grad max={float(pg_head.abs().max()):.3e} "
        f"value-head grad max={float(vg_head.abs().max()):.3e}")


if __name__ == "__main__":
  test_masked_train_matches_stock_with_ones_mask()
  test_masked_steps_give_zero_policy_grad_nonzero_value_grad()
  print("ALL MASKED PPO TESTS PASSED")
