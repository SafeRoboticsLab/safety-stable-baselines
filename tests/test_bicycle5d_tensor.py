"""`BicycleGoalTensorVec` vs `BicycleGoalVec` — parity and contract.

The tensor env is a port, so the only interesting question is whether it is the
SAME env. The parity tests drive both from an IDENTICAL state with an IDENTICAL
action sequence and compare the whole 5-tuple step by step; the RNGs differ, so
on every terminated env the numpy state is copied back into the tensor env and
free-running resumes (hundreds of steps of divergence accumulate between those
re-syncs, which is the point).

Two precisions are checked separately, because they answer different questions:

* ``float64`` — "is the MATH the same?" Any divergence here is a port bug.
* ``float32`` — "how much does the training dtype cost?" Divergence here is
  round-off, and the test pins how big it is allowed to get.

Run standalone for the numbers: ``python tests/test_bicycle5d_tensor.py``.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch as th

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safety_sb3.testing.bicycle5d import DSTB_LIM  # noqa: E402
from safety_sb3.testing.bicycle5d_tensor import BicycleGoalTensorVec  # noqa: E402
from safety_sb3.testing.bicycle5d_vec import BicycleGoalVec  # noqa: E402

DEV = "cpu"          # parity is a math check; it must run without a GPU


def _sync(tenv, nenv, mask=None):
  """Copy numpy env state (all envs, or the masked rows) into the tensor env."""
  sel = slice(None) if mask is None else np.nonzero(mask)[0]
  idx = (slice(None) if mask is None
         else th.as_tensor(sel, dtype=th.long, device=tenv.device))
  tk = dict(dtype=tenv.dtype, device=tenv.device)
  tenv.s[idx] = th.as_tensor(nenv.s[sel], **tk)
  tenv.obst[idx] = th.as_tensor(nenv.obst[sel], **tk)
  tenv.goal[idx] = th.as_tensor(nenv.goal[sel], **tk)
  tenv.t[idx] = th.as_tensor(nenv.t[sel], dtype=th.int64, device=tenv.device)


def parity(dtype=th.float64, n=64, steps=400, adversary=False, spawn="wide",
           seed=0):
  """Step both envs on the same actions from the same state; return max diffs."""
  nenv = BicycleGoalVec(n, adversary=adversary, spawn=spawn, seed=seed)
  tenv = BicycleGoalTensorVec(n, adversary=adversary, spawn=spawn, seed=seed,
                              device=DEV, dtype=dtype)
  nobs = nenv.reset()
  _sync(tenv, nenv)
  worst = dict(obs=0.0, g=0.0, l=0.0, state=0.0, dones=0, timeouts=0,
               resyncs=0, steps=steps)
  # NB `BicycleGoalVec._obs()` / `step_wait()` emit float32 (the SB3 VecEnv
  # contract), so obs and g are compared in that float32 image — otherwise the
  # numpy side's own downcast, not the port, dominates the number.
  f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
  worst["obs"] = float(np.abs(nobs - f32(tenv._obs().cpu().numpy())).max())

  rng = np.random.default_rng(1234)
  adim = nenv.action_space.shape[0]
  for _ in range(steps):
    # Deliberately outside the action bounds so the env's own clipping (and,
    # with adversary=True, the ctrl/dstb split) is exercised on both sides.
    a = rng.uniform(-3.0, 3.0, (n, adim))
    nenv.step_async(a)
    n_obs_, n_g, n_dones, n_infos = nenv.step_wait()
    n_l = np.array([i["l_x"] for i in n_infos])
    n_to = np.array([i.get("TimeLimit.truncated", False) for i in n_infos])

    t_obs_, t_g, t_dones, t_to, t_l = tenv.step_tensor(
      th.as_tensor(a, dtype=dtype, device=DEV))
    t_obs_ = f32(t_obs_.cpu().numpy())

    worst["g"] = max(worst["g"], float(np.abs(n_g - f32(t_g.cpu().numpy())).max()))
    worst["l"] = max(worst["l"], float(np.abs(n_l - t_l.cpu().numpy()).max()))
    worst["dones"] += int((n_dones != t_dones.cpu().numpy()).sum())
    worst["timeouts"] += int((n_to != t_to.cpu().numpy()).sum())

    live = ~n_dones                       # obs of a done env is its RESET obs
    if live.any():
      worst["obs"] = max(worst["obs"],
                         float(np.abs(n_obs_[live] - t_obs_[live]).max()))
      worst["state"] = max(
        worst["state"],
        float(np.abs(nenv.s[live] - tenv.s.cpu().numpy()[live]).max()))
    if n_dones.any():                     # RNGs differ -> re-seed from numpy
      worst["resyncs"] += int(n_dones.sum())
      _sync(tenv, nenv, n_dones)
  return worst


# --------------------------------------------------------------------------
# parity
# --------------------------------------------------------------------------
def test_parity_float64_is_exact():
  """float64: the port must reproduce the numpy env to round-off (~1e-12)."""
  for adversary in (False, True):
    w = parity(th.float64, adversary=adversary)
    print(f"[parity f64 adversary={adversary}] max|dobs|={w['obs']:.3e} "
          f"max|dg|={w['g']:.3e} max|dl|={w['l']:.3e} "
          f"max|dstate|={w['state']:.3e} done/timeout mismatches="
          f"{w['dones']}/{w['timeouts']} over {w['steps']} steps "
          f"({w['resyncs']} resyncs)")
    # State and l are compared in full float64; obs and g in the float32 the
    # numpy VecEnv emits. All four must be exact -- anything else is a bug.
    assert w["state"] < 1e-12, f"state divergence {w['state']} is a port bug"
    assert w["l"] < 1e-12, f"l divergence {w['l']} is a port bug"
    assert w["obs"] == 0.0 and w["g"] == 0.0, "float32 images must be identical"
    assert w["dones"] == 0 and w["timeouts"] == 0


def test_parity_float32_is_roundoff_only():
  """float32 (the training dtype): divergence is round-off, not behavior."""
  w = parity(th.float32)
  print(f"[parity f32] max|dobs|={w['obs']:.3e} max|dg|={w['g']:.3e} "
        f"max|dl|={w['l']:.3e} max|dstate|={w['state']:.3e} "
        f"done/timeout mismatches={w['dones']}/{w['timeouts']}")
  # 400 steps of RK4 in float32 vs float64 from an identical state.
  assert w["state"] < 1e-3, f"float32 drift {w['state']} is too large"
  assert w["g"] < 1e-3 and w["l"] < 1e-3
  # A done flag can flip only when a margin sits within round-off of 0; over
  # 400 x 64 = 25,600 transitions that must be a handful at most.
  assert w["dones"] <= 5, f"{w['dones']} done mismatches is behavioral"


def test_spawn_modes_match_distribution():
  """The spawn RNG differs, so match the SUPPORT, not the sample."""
  for spawn, xlo, xhi in (("edge", -0.15, 0.15), ("wide", -0.2, 1.3),
                          ("map", -0.2, 2.6)):
    t = BicycleGoalTensorVec(2048, spawn=spawn, device=DEV, seed=3)
    t.reset()
    s = t.s.cpu().numpy()
    assert xlo - 1e-5 <= s[:, 0].min() and s[:, 0].max() <= xhi + 1e-5
    assert 0.0 <= s[:, 2].min() and s[:, 2].max() <= 0.4
    assert np.abs(s[:, 4]).max() <= 0.10 + 1e-5
    if spawn != "edge":                   # rejection sampling must have run
      d = (np.hypot(t.obst.cpu().numpy()[:, :, 0] - s[:, None, 0],
                    t.obst.cpu().numpy()[:, :, 1] - s[:, None, 1])
           - t.obst.cpu().numpy()[:, :, 2]).min(axis=1)
      assert (d >= 0.0).mean() > 0.99, "spawned inside an obstacle"
    print(f"[ok] spawn={spawn}: x in [{s[:, 0].min():.2f}, {s[:, 0].max():.2f}]")


# --------------------------------------------------------------------------
# contract: l_x, adversary split, done/timeout
# --------------------------------------------------------------------------
def test_l_x_semantics():
  """``l_x >= 0`` iff the car centre is inside the goal circle, and the
  piecewise scale gives +GOAL_VALUE at the centre."""
  from safety_sb3.testing.bicycle5d import CLAMP_L, GOAL_VALUE
  env = BicycleGoalTensorVec(256, spawn="wide", device=DEV, seed=5,
                             dtype=th.float64)
  env.reset()
  _, _, _, _, l = env.step_tensor(th.zeros(256, 2, dtype=th.float64))
  d = th.hypot(env.s[:, 0] - env.goal[:, 0], env.s[:, 1] - env.goal[:, 1])
  inside = d <= env.goal[:, 2]
  assert bool(((l >= 0) == inside).all()), "l >= 0 must mean 'in the goal'"
  assert float(l.max()) <= CLAMP_L and float(l.min()) >= -CLAMP_L
  # at the exact centre l == GOAL_VALUE
  env.s[:, 0], env.s[:, 1] = env.goal[:, 0], env.goal[:, 1]
  assert abs(float(env._l()[0]) - GOAL_VALUE) < 1e-12
  print(f"[ok] l_x: sign matches goal membership, l(centre)={GOAL_VALUE}")


def test_adversary_action_split():
  """``[0:2]`` drives, ``[2:7]`` disturbs, and only when ``adversary=True``."""
  def run(adversary, dstb, ctrl=(1.0, 0.0)):
    e = BicycleGoalTensorVec(8, adversary=adversary, spawn="edge", seed=7,
                             device=DEV, dtype=th.float64)
    e.reset()
    e.s.zero_()                            # identical, deterministic state
    a = th.tensor([list(ctrl) + list(dstb)], dtype=th.float64).repeat(8, 1)
    e.step_tensor(a[:, :7] if adversary else a[:, :2])
    return e.s.clone()

  zero_d = run(True, [0.0] * 5)
  push_d = run(True, [1.0, 0.0, 0.0, 0.0, 0.0])       # x-rate disturbance
  assert float((push_d[:, 0] - zero_d[:, 0]).abs().min()) > 1e-6, \
    "dstb dims must move the state when adversary=True"
  # ...and the disturbance is CLIPPED to DSTB_LIM: 1.0 and 10.0 land alike
  huge_d = run(True, [10.0, 0.0, 0.0, 0.0, 0.0])
  assert float((push_d - huge_d).abs().max()) < 1e-12, "dstb not clipped"
  # dx = DSTB_LIM[0] * dt exactly (v = 0, so the only x-rate is the dstb)
  dx = float((push_d[0, 0] - zero_d[0, 0]))
  assert abs(dx - DSTB_LIM[0] * 0.05) < 1e-9, f"dx={dx}"
  # single-player action space carries no dstb dims at all
  e1 = BicycleGoalTensorVec(4, adversary=False, device=DEV)
  assert e1.action_space.shape == (2,) and e1.ctrl_action_dim == 2
  e2 = BicycleGoalTensorVec(4, adversary=True, device=DEV)
  assert e2.action_space.shape == (7,) and e2.ctrl_action_dim == 2
  print(f"[ok] adversary split: dstb clipped to +-{DSTB_LIM[0]}, dx={dx:.5f}")


def test_done_and_timeout_semantics():
  """``timeouts = truncated & ~terminated``; the episode is exactly ``timeout``
  steps when nothing terminates; ``dones`` auto-resets the clock."""
  env = BicycleGoalTensorVec(16, spawn="edge", timeout=10,
                             terminate_on_goal=True, device=DEV,
                             obstacles=(), goal=(50.0, 0.0, 0.4))
  env.reset()
  zero = th.zeros(16, 2)
  for i in range(9):                       # no obstacles, goal unreachable
    _, g, dones, to, _ = env.step_tensor(zero)
    assert not bool(dones.any()), f"early done at step {i}"
    assert float(g.min()) == 3.0, "no obstacles -> g == CLAMP_G"
  _, _, dones, to, _ = env.step_tensor(zero)
  assert bool(dones.all()) and bool(to.all()), "step 10 must truncate"
  assert int(env.t.max()) == 0, "auto-reset must zero the clock"

  # a terminated (collided) env is done but NOT a timeout
  env2 = BicycleGoalTensorVec(4, spawn="edge", timeout=300, randomize=False,
                              device=DEV, dtype=th.float64)
  env2.reset()
  env2.s.zero_()
  env2.obst[:, 0, :] = th.tensor([0.0, 0.0, 0.5], dtype=th.float64)  # on top
  _, g, dones, to, _ = env2.step_tensor(th.zeros(4, 2, dtype=th.float64))
  assert bool((g < 0).all()) and bool(dones.all()) and not bool(to.any())
  print("[ok] done/timeout: truncation at `timeout`, termination is not a timeout")


if __name__ == "__main__":
  test_parity_float64_is_exact()
  test_parity_float32_is_roundoff_only()
  test_spawn_modes_match_distribution()
  test_l_x_semantics()
  test_adversary_action_split()
  test_done_and_timeout_semantics()
  print("ALL BICYCLE5D TENSOR TESTS PASSED")
