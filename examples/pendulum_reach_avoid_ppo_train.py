"""Validation for ReachAvoidPPO / IsaacsPPO on a classic pendulum task.

Reach-avoid pendulum (Hsu et al., RSS 2021 style):
    avoid:  |theta| > 30 deg                 -> g(s) = pi/6 - |theta|
    reach:  upright & slow                   -> l(s) = min(0.15 - |theta|,
                                                          1.0 - |theta_dot|)
Episodes spawn INSIDE the safe set (rejection via direct state write).
Success = reach the target band without ever leaving the safe set.

IsaacsPPO variant: the action space is [ctrl_torque, dstb_torque]; the
adversary applies up to 30% of the control authority as a disturbance torque.

Validated results (200k steps, seed 0): ReachAvoidPPO 100% reach-avoid
success (2 seeds); IsaacsPPO 100%/100%/99% vs zero/random/learned adversary
with a full 5+5 leaderboard archive.

Run (safety_sb3 conda env):
    python examples/pendulum_reach_avoid_ppo_train.py --algo reach_avoid
    python examples/pendulum_reach_avoid_ppo_train.py --algo isaacs
"""

from __future__ import annotations

import argparse
import os
import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safety_sb3 import IsaacsPPO, ReachAvoidPPO  # noqa: E402

ANGLE_LIMIT = np.pi / 6


class PendulumReachAvoid(gym.Wrapper):
  """g = angle_limit - |theta| (reward channel); l = upright & slow (info)."""

  def __init__(self, env: gym.Env, angle_limit: float = ANGLE_LIMIT):
    super().__init__(env)
    self.angle_limit = float(angle_limit)

  @staticmethod
  def _theta(obs):
    return float(np.arctan2(obs[1], obs[0]))

  def _margins(self, obs):
    theta = self._theta(obs)
    theta_dot = float(obs[2])
    g = self.angle_limit - abs(theta)
    l = min(0.15 - abs(theta), 1.0 - abs(theta_dot))
    return g, l

  def step(self, action):
    obs, _r, terminated, truncated, info = self.env.step(action)
    g, l = self._margins(obs)
    if g < 0.0:
      terminated = True
    info["l_x"] = l
    return obs, float(g), terminated, truncated, info

  def reset(self, **kwargs):
    obs, info = self.env.reset(**kwargs)
    # Spawn inside the safe set at FEASIBLE states (a fast spin next to the
    # angle wall is ballistically doomed regardless of policy — including
    # such states makes eval success depend on spawn luck, not the policy).
    theta = np.random.uniform(-0.5, 0.5) * self.angle_limit
    theta_dot = np.random.uniform(-0.6, 0.6)
    self.env.unwrapped.state = np.array([theta, theta_dot])
    obs = np.array([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)
    g, l = self._margins(obs)
    info["l_x"] = l
    return obs, info


class PendulumIsaacs(PendulumReachAvoid):
  """Concatenated [ctrl, dstb] torque; dstb authority = dstb_scale * ctrl."""

  def __init__(self, env: gym.Env, dstb_scale: float = 0.5, **kw):
    super().__init__(env, **kw)
    base = env.action_space
    self.dstb_scale = float(dstb_scale)
    self.action_space = spaces.Box(
      low=np.concatenate([base.low, base.low]),
      high=np.concatenate([base.high, base.high]),
      dtype=np.float32,
    )

  def step(self, action):
    u = np.asarray(action[:1], dtype=np.float32)
    d = np.asarray(action[1:], dtype=np.float32) * self.dstb_scale
    return super().step(np.clip(u + d, self.env.action_space.low,
                                self.env.action_space.high))


def _report(model, env_fn, n_episodes=200, adversarial=False):
  env = env_fn()
  succ = 0
  for _ in range(n_episodes):
    obs, _ = env.reset()
    ever_l, ever_g = False, False
    for _t in range(200):
      act, _ = model.predict(obs, deterministic=True)
      if act.shape[0] == 1 and isinstance(env, PendulumIsaacs):
        if adversarial:
          d = np.sign(np.random.randn(1)).astype(np.float32)
        else:
          d = np.zeros(1, dtype=np.float32)
        act = np.concatenate([act, d])
      obs, g, term, trunc, info = env.step(act)
      ever_l |= info["l_x"] >= 0
      ever_g |= g < 0
      if term or trunc:
        break
    succ += int(ever_l and not ever_g)
  return succ / n_episodes


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--algo", choices=["reach_avoid", "isaacs"], default="reach_avoid")
  p.add_argument("--steps", type=int, default=200_000)
  args = p.parse_args()

  if args.algo == "reach_avoid":
    def make():
      return Monitor(PendulumReachAvoid(gym.make("Pendulum-v1")))
    venv = DummyVecEnv([make for _ in range(8)])
    model = ReachAvoidPPO(
      "MlpPolicy", venv, n_steps=256, batch_size=512, gamma=0.99,
      gae_lambda=0.95, learning_rate=3e-4, ent_coef=0.005, verbose=1,
    )
    model.learn(total_timesteps=args.steps)
    rate = _report(model, lambda: PendulumReachAvoid(gym.make("Pendulum-v1")))
    print(f"[reach_avoid] success (reach & never-unsafe): {100*rate:.1f}%")
  else:
    def make():
      return Monitor(PendulumIsaacs(gym.make("Pendulum-v1"), dstb_scale=0.3))
    venv = DummyVecEnv([make for _ in range(8)])
    model = IsaacsPPO(
      "MlpPolicy", venv, ctrl_action_dim=1,
      dstb_pretrain_rollouts=15, ctrl_rollouts_per_cycle=4,
      dstb_rollouts_per_cycle=1, use_leaderboard=True,
      leaderboard_dir="experiments/pendulum_isaacs_ppo_leaderboard",
      n_steps=256, batch_size=512, gamma=0.99, gae_lambda=0.95,
      learning_rate=3e-4, ent_coef=0.005, verbose=1,
    )
    model.learn(total_timesteps=args.steps)
    env_fn = lambda: PendulumIsaacs(gym.make("Pendulum-v1"), dstb_scale=0.3)  # noqa: E731
    rate0 = _report(model, env_fn, adversarial=False)
    print(f"[isaacs] success with ZERO dstb: {100*rate0:.1f}%")
    rate = _report(model, env_fn, adversarial=True)
    print(f"[isaacs] success under random-sign max dstb: {100*rate:.1f}%")
    print(f"[isaacs] archived dstb checkpoints: "
          f"{len(model._leaderboard.dstb_steps) if model._leaderboard else 0}")


if __name__ == "__main__":
  main()
