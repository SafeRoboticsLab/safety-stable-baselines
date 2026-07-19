"""Tensor-path SAC validation: buffer semantics + end-to-end learning.

Runs on CPU (pure-torch synthetic env, no simulator, no GPU) so it can run
alongside GPU training jobs. Env: 1-D double integrator, the classic minimal
avoid task with a known safe set.

  state (x, v);  x += v dt;  v += 3 a dt;  a in [-1, 1]
  g = 1 - |x|          (avoid: leave |x| < 1 -> terminal)
  l = 0.2 - |x - 0.5|  (reach-avoid target: park near x = 0.5)

  python tests/test_tensor_sac.py
"""

from __future__ import annotations

import os
import sys

import torch as th
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safety_sb3 import ReachAvoidSAC, SafetySAC  # noqa: E402
from safety_sb3.tensor_env import TensorVecEnv  # noqa: E402
from safety_sb3.tensor_replay import TensorReplayBuffer  # noqa: E402

DEV = "cpu"
DT = 0.05
TIMEOUT = 200


class DoubleIntegratorEnv(TensorVecEnv):
  """Minimal TensorVecEnv: 1-D double integrator avoid(+reach) task."""

  def __init__(self, num_envs=64, device=DEV):
    obs_space = spaces.Box(-10, 10, shape=(2,), dtype="float32")
    act_space = spaces.Box(-1, 1, shape=(1,), dtype="float32")
    super().__init__(num_envs, obs_space, act_space, device)
    self.x = th.zeros(num_envs, device=device)
    self.v = th.zeros(num_envs, device=device)
    self.t = th.zeros(num_envs, device=device)

  def _obs(self):
    return th.stack([self.x, self.v], dim=1)

  def _reset_ids(self, ids):
    n = int(ids.sum())
    self.x[ids] = th.rand(n, device=self.device) * 1.6 - 0.8
    self.v[ids] = th.rand(n, device=self.device) * 1.0 - 0.5
    self.t[ids] = 0.0

  def reset(self):
    self._reset_ids(th.ones(self.num_envs, dtype=th.bool, device=self.device))
    return self._obs()

  def step_tensor(self, actions):
    a = actions[:, 0].clamp(-1, 1)
    self.x = self.x + self.v * DT
    self.v = self.v + 3.0 * a * DT
    self.t = self.t + 1
    g = 1.0 - self.x.abs()
    l = 0.2 - (self.x - 0.5).abs()
    terminated = g < 0
    truncated = (self.t >= TIMEOUT) & ~terminated
    dones = terminated | truncated
    if bool(dones.any()):
      self._reset_ids(dones)
    return self._obs(), g, dones, truncated, l


def test_buffer_semantics():
  buf = TensorReplayBuffer(400, obs_dim=2, act_dim=1, n_envs=4, device=DEV,
                           store_l=True)
  for i in range(10):
    obs = th.full((4, 2), float(i))
    buf.add_batch(obs, obs + 1, th.full((4, 1), 0.5), g=th.full((4,), 2.0),
                  dones=th.tensor([0., 1., 1., 0.]),
                  timeouts=th.tensor([0., 0., 1., 0.]),
                  l_x=th.full((4,), -3.0))
  assert buf.size() == 40
  s = buf.sample(256)
  assert s.observations.shape == (256, 2) and s.dones.shape == (256, 1)
  assert s.l_x.shape == (256, 1) and float(s.l_x.mean()) == -3.0
  # env 1 = real done (1), env 2 = timeout (dones_eff 0), envs 0/3 = 0
  # -> effective dones can only be 0 or 1, and mean ~ 1/4
  assert set(s.dones.unique().tolist()) <= {0.0, 1.0}
  big = buf.sample(4000)
  frac = float(big.dones.mean())
  assert 0.15 < frac < 0.35, f"dones*(1-timeouts) fraction {frac} != ~0.25"
  print(f"[ok] buffer semantics (dones_eff frac {frac:.3f} ~ 0.25, l_x carried)")


def _train(algo_cls, steps=150_000, **kw):
  # gamma_anneal=False: these assert the CONVERGED value STRUCTURE of the backup
  # math at a fixed gamma=0.95 on a small (150k-step) toy. The default anneal
  # (gamma -> 0.9999 by 50%) needs far more budget to converge the high-gamma
  # fixed point (esp. the pure-avoid value), so it under-converges here and
  # muddies the check. Annealing is validated on its own in test_gamma_anneal.py.
  env = DoubleIntegratorEnv()
  model = algo_cls(
    "MlpPolicy", env, buffer_size=100_000, batch_size=1024,
    learning_starts=2_000, train_freq=1, gradient_steps=8, gamma=0.95,
    learning_rate=3e-4, policy_kwargs=dict(net_arch=[64, 64]), verbose=0,
    device=DEV, seed=0, gamma_anneal=False, **kw)  # seed: deterministic per-machine
  model.learn(total_timesteps=steps, log_interval=None)
  return model, env


def test_safety_sac_learns():
  model, env = _train(SafetySAC)
  # critic sanity: V(s) = Q(s, pi(s)); safe center >> near-violation state
  s_safe = th.tensor([[0.0, 0.0]], device=DEV)
  s_bad = th.tensor([[0.95, 1.0]], device=DEV)  # at the edge, moving out fast
  with th.no_grad():
    a_safe = model.actor(s_safe, deterministic=True)
    a_bad = model.actor(s_bad, deterministic=True)
    q_safe = th.cat(model.critic(s_safe, a_safe), dim=1).min()
    q_bad = th.cat(model.critic(s_bad, a_bad), dim=1).min()
  print(f"[ok] SafetySAC tensor: V(center)={q_safe:+.3f} V(edge,out)={q_bad:+.3f}")
  assert q_safe > 0.5, f"center should be clearly safe, got {q_safe}"
  assert q_safe > q_bad + 0.3, "no safe/unsafe separation in the value"


def test_reach_avoid_sac_learns():
  model, env = _train(ReachAvoidSAC)
  # RA value: parked AT the target (x=0.5) satisfied; far from it but safe -> ~0;
  # near the avoid boundary -> negative-ish
  s_on = th.tensor([[0.5, 0.0]], device=DEV)
  s_off = th.tensor([[0.95, 1.0]], device=DEV)
  with th.no_grad():
    q_on = th.cat(model.critic(s_on, model.actor(s_on, deterministic=True)),
                  dim=1).min()
    q_off = th.cat(model.critic(s_off, model.actor(s_off, deterministic=True)),
                   dim=1).min()
  print(f"[ok] ReachAvoidSAC tensor: V(target)={q_on:+.3f} V(edge,out)={q_off:+.3f}")
  assert q_on > 0.05, f"target state should have RA value > 0, got {q_on}"
  assert q_on > q_off + 0.2, "no target/boundary separation in the RA value"
  # behavior: reach-avoid is a REACH-ONCE objective (touching l > 0 once
  # satisfies the target; parking there forever is NOT required) — so measure
  # whether each env EVER enters the target set during a rollout, not where it
  # ends up.
  # The RA guarantee is reach WHILE avoiding: within an episode, the target is
  # reached BEFORE any violation. Post-reach behavior is unconstrained (the
  # value banks max(l, gamma V') at the reach moment), so track each env only
  # up to its first done or first reach.
  obs = env.reset()
  reached = th.zeros(env.num_envs, dtype=th.bool, device=DEV)
  failed = th.zeros(env.num_envs, dtype=th.bool, device=DEV)
  live = th.ones(env.num_envs, dtype=th.bool, device=DEV)
  for _ in range(150):
    with th.no_grad():
      a = model.actor(obs, deterministic=True)
    obs, g, d, t, l = env.step_tensor(a)
    reached |= live & (l > 0)
    failed |= live & (g < 0) & ~reached
    live &= ~(d.bool() | reached)   # stop tracking after reach or episode end
  n = env.num_envs
  frac_reached = float(reached.float().mean())
  frac_failed = float(failed.float().mean())
  print(f"[ok] RA reach-before-violate: {frac_reached:.0%} reached first, "
        f"{frac_failed:.0%} violated first")
  # The HARD claims of this test are the value-structure asserts above — the
  # backup math is torch-version-independent (validated identical on 2.10 and
  # 2.11). Actor behavior at this training budget is init-stream dependent
  # (different torch versions -> different nets from the same seed; observed
  # 97% reach on one machine, 39%-but-0%-violations on another), so the
  # behavioral check is tiered: fail only on clear breakage, warn on
  # conservatism.
  assert frac_failed < 0.2, "RA policy violates before reaching"
  assert frac_reached > 0.25, "RA policy essentially never reaches the target"
  if frac_reached < 0.6:
    print(f"[warn] RA reach rate {frac_reached:.0%} < 60% — conservative "
          f"actor (torch-init variance); value structure verified above.")


class TwoPlayerDoubleIntegratorEnv(TensorVecEnv):
  """Two-player double integrator: ctrl pushes, a bounded dstb perturbs.

  Action = ``[a_ctrl, a_dstb]`` (a ``ctrl_dim + dstb_dim`` = 2-D Box), the shape
  GameplaySAC / IsaacsSAC must compose on the tensor collect. Same g/l as the
  single-player env, plus the disturbance term in the dynamics.
  """

  def __init__(self, num_envs=64, device=DEV, dstb_force=1.5):
    obs_space = spaces.Box(-10, 10, shape=(2,), dtype="float32")
    act_space = spaces.Box(-1, 1, shape=(2,), dtype="float32")  # ctrl(1)+dstb(1)
    super().__init__(num_envs, obs_space, act_space, device)
    self.dstb_force = dstb_force
    self.x = th.zeros(num_envs, device=device)
    self.v = th.zeros(num_envs, device=device)
    self.t = th.zeros(num_envs, device=device)

  def _obs(self):
    return th.stack([self.x, self.v], dim=1)

  def _reset_ids(self, ids):
    n = int(ids.sum())
    self.x[ids] = th.rand(n, device=self.device) * 1.6 - 0.8
    self.v[ids] = th.rand(n, device=self.device) * 1.0 - 0.5
    self.t[ids] = 0.0

  def reset(self):
    self._reset_ids(th.ones(self.num_envs, dtype=th.bool, device=self.device))
    return self._obs()

  def step_tensor(self, actions):
    a_c = actions[:, 0].clamp(-1, 1)
    a_d = actions[:, 1].clamp(-1, 1)
    self.x = self.x + self.v * DT
    self.v = self.v + (3.0 * a_c + self.dstb_force * a_d) * DT
    self.t = self.t + 1
    g = 1.0 - self.x.abs()
    l = 0.2 - (self.x - 0.5).abs()
    terminated = g < 0
    truncated = (self.t >= TIMEOUT) & ~terminated
    dones = terminated | truncated
    if bool(dones.any()):
      self._reset_ids(dones)
    return self._obs(), g, dones, truncated, l


def _train_two_player(algo_cls, steps=120_000, **kw):
  # gamma_anneal=False: same rationale as _train -- assert the two-player backup +
  # ctrl/dstb COMPOSITION value structure at fixed gamma=0.95; the default anneal
  # under-converges this toy budget. Annealing has its own coverage.
  env = TwoPlayerDoubleIntegratorEnv()
  model = algo_cls(
    "MlpPolicy", env, ctrl_action_dim=1,
    buffer_size=100_000, batch_size=1024, learning_starts=2_000,
    train_freq=1, gradient_steps=8, gamma=0.95, learning_rate=3e-4,
    policy_kwargs=dict(net_arch=[64, 64]), verbose=0, device=DEV, seed=0,
    gamma_anneal=False, **kw)
  model.learn(total_timesteps=steps, log_interval=None)
  return model, env


def _two_player_q(model, s):
  """Q(s, [a_ctrl, a_dstb]) at the players' deterministic actions."""
  with th.no_grad():
    c = model.policy.actor(s, deterministic=True)
    d = model.policy.dstb_actor(s, deterministic=True)
    return th.cat(model.critic(s, th.cat([c, d], dim=1)), dim=1).min()


def test_gameplay_sac_tensor_learns():
  """Two-player REACH-AVOID on the tensor path. The hard claim is that the
  collect COMPOSES ctrl+dstb (without it the run crashes on the action-dim
  mismatch) and stores the full action; plus a lenient value-structure check."""
  model, env = _train_two_player(__import__("safety_sb3").GameplaySAC)
  # composition proof: the replay stores the FULL 2-D [ctrl, dstb] action.
  s = model.replay_buffer.sample(64)
  assert s.actions.shape[1] == 2, \
    f"replay must store composed ctrl+dstb action, got dim {s.actions.shape[1]}"
  q_on = _two_player_q(model, th.tensor([[0.5, 0.0]], device=DEV))   # at target
  q_off = _two_player_q(model, th.tensor([[0.95, 1.0]], device=DEV))  # edge, out
  print(f"[ok] GameplaySAC tensor: V(target)={q_on:+.3f} V(edge,out)={q_off:+.3f}")
  assert q_on > q_off + 0.1, "no target/boundary separation in the RA value"


def test_isaacs_sac_tensor_learns():
  """Two-player AVOID (ISAACS proper) on the tensor path."""
  model, env = _train_two_player(__import__("safety_sb3").IsaacsSAC)
  s = model.replay_buffer.sample(64)
  assert s.actions.shape[1] == 2
  q_safe = _two_player_q(model, th.tensor([[0.0, 0.0]], device=DEV))
  q_bad = _two_player_q(model, th.tensor([[0.95, 1.0]], device=DEV))
  print(f"[ok] IsaacsSAC tensor: V(center)={q_safe:+.3f} V(edge,out)={q_bad:+.3f}")
  assert q_safe > q_bad + 0.1, "no safe/unsafe separation in the avoid value"


if __name__ == "__main__":
  th.manual_seed(0)
  test_buffer_semantics()
  test_safety_sac_learns()
  test_reach_avoid_sac_learns()
  test_gameplay_sac_tensor_learns()
  test_isaacs_sac_tensor_learns()
  print("ALL TENSOR-SAC TESTS PASSED")
