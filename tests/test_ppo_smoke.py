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
  model = SafetyPPO("MlpPolicy", DoubleIntegratorGym(), n_steps=256,
                    batch_size=256, seed=0, verbose=0)
  model.learn(total_timesteps=8_192, callback=StdCapCallback(max_std=0.3))
  assert float(model.policy.log_std.exp().max()) <= 0.3 + 1e-4
  vals = _values(model, [[0.0, 0.0], [0.98, 0.8]])
  assert np.all(np.isfinite(vals))
  # center of the safe set must be valued above the exiting boundary state
  assert vals[0] > vals[1], vals


def test_reach_avoid_ppo_smoke():
  model = ReachAvoidPPO("MlpPolicy", DoubleIntegratorGym(), n_steps=256,
                        batch_size=256, seed=0, verbose=0)
  model.learn(total_timesteps=8_192)
  # l_x must actually have been threaded into the buffer (not the 0.0 default)
  assert float(np.abs(model.rollout_buffer.l_x).sum()) > 0
  vals = _values(model, [[0.5, 0.0], [0.98, 0.8]])
  assert np.all(np.isfinite(vals))
  assert vals[0] > vals[1], vals
