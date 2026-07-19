"""Smoke tests for the on-policy family on the numpy/gym path.

Same 1-D double integrator as tests/test_tensor_sac.py, but exposed as a
plain gymnasium.Env so SafetyPPO / ReachAvoidPPO exercise the standard SB3
VecEnv path (g on the reward channel, l via info["l_x"]).
"""
import gymnasium as gym
import numpy as np
import torch as th

from safety_sb3 import ReachAvoidPPO, SafetyPPO, StdCapCallback

DT = 0.05
TIMEOUT = 120


class DoubleIntegratorGym(gym.Env):
  observation_space = gym.spaces.Box(-10, 10, (2,), dtype=np.float32)
  action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

  def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    self.x = self.np_random.uniform(-0.8, 0.8)
    self.v = self.np_random.uniform(-0.5, 0.5)
    self.t = 0
    return np.array([self.x, self.v], np.float32), {}

  def step(self, action):
    a = float(np.clip(action[0], -1, 1))
    self.x += self.v * DT
    self.v += 3.0 * a * DT
    self.t += 1
    g = 1.0 - abs(self.x)                 # avoid: leave |x| < 1
    l = 0.2 - abs(self.x - 0.5)           # reach: park near x = 0.5
    obs = np.array([self.x, self.v], np.float32)
    return obs, g, g < 0, self.t >= TIMEOUT, {"l_x": l}


def _values(model, states):
  obs = th.as_tensor(np.array(states, np.float32), device=model.device)
  with th.no_grad():
    return model.policy.predict_values(obs).cpu().numpy().ravel()


def test_safety_ppo_smoke_and_std_cap():
  # gamma_anneal=False: this is a minimal (8k-step, single-env, numpy-path) smoke
  # of the std-cap mechanism + value finiteness/ordering. The default gamma anneal
  # (0.99 -> 0.9999) deliberately drives gamma high within this tiny budget, which
  # muddies the near-boundary extrapolation probe [0.98, 0.8] (a marginal check
  # even at constant gamma). Annealing is validated on its own in
  # test_gamma_anneal.py and via the tensor-path value-structure tests; pin it off
  # here to isolate the std cap.
  model = SafetyPPO("MlpPolicy", DoubleIntegratorGym(), n_steps=256,
                    batch_size=256, seed=0, verbose=0, gamma_anneal=False)
  model.learn(total_timesteps=8_192, callback=StdCapCallback(max_std=0.3))
  assert float(model.policy.log_std.exp().max()) <= 0.3 + 1e-4
  vals = _values(model, [[0.0, 0.0], [0.98, 0.8]])
  assert np.all(np.isfinite(vals))
  # center of the safe set must be valued above the exiting boundary state
  assert vals[0] > vals[1], vals


def test_reach_avoid_ppo_smoke():
  """The critic must show a REACH signal, and converge to min(l, g) on target.

  Both probe states are safe with the SAME g = 0.5, so an avoid-only critic
  values them identically; only the reach term can separate them. (An earlier
  version of this test compared the target against a doomed boundary state,
  which the avoid signal alone already orders — and which sits off the
  converged policy's visitation, so the critic there is extrapolating.)
  """
  # gamma_anneal=False on purpose: this probe isolates the REACH signal with two
  # SAME-g states (on-target x=0.5 vs reachable-off-target x=-0.5). Their
  # separation is a DISCOUNTED-value property -- at the annealed limit gamma->1
  # both states reach the target, so both correctly converge to the SAME RA value
  # min(l, g)=0.2 and the reach probe no longer separates them. (The avoid-driven
  # separation against a DOOMED state does survive gamma->1 -- see the tensor RA
  # test.) Annealing itself is covered in test_gamma_anneal.py.
  model = ReachAvoidPPO("MlpPolicy", DoubleIntegratorGym(), n_steps=256,
                        batch_size=256, seed=0, verbose=0, gamma_anneal=False)
  model.learn(total_timesteps=40_960)
  # l_x must actually have been threaded into the buffer (not the 0.0 default)
  assert float(np.abs(model.rollout_buffer.l_x).sum()) > 0
  # x = 0.5 -> l = +0.2 (on target); x = -0.5 -> l = -0.8 (off target)
  vals = _values(model, [[0.5, 0.0], [-0.5, 0.0]])
  assert np.all(np.isfinite(vals))
  assert vals[0] > vals[1] + 0.05, vals
  # parking on target is a fixed point at V = min(l, g) = min(0.2, 0.5) = 0.2
  assert abs(vals[0] - 0.2) < 0.05, vals
