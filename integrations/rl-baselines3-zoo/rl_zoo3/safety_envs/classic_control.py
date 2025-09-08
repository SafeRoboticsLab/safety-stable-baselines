# safety_sb3/integrations/rl_zoo/envs/safety_pendulum.py
import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register


class PendulumSafety(gym.Wrapper):
    """
    Safety reward wrapper for Pendulum-v1.
    Keeps the pendulum within ±margin_deg around the upright (theta=0).

    obs = [cos(theta), sin(theta), theta_dot]
    safety margin (reward):
        g(s) = margin_rad - |theta|      (>=0 safe, <0 violation)
    Optionally terminate when g(s) < 0.
    """

    def __init__(self, env: gym.Env, margin_deg: float = 30.0):
        super().__init__(env)
        self.margin_rad = float(margin_deg) * np.pi / 180.0

    def step(self, action):
        obs, _task_r, terminated, truncated, info = self.env.step(action)
        # decode theta
        c, s = float(obs[0]), float(obs[1])
        theta = float(np.arctan2(s, c))  # [-pi, pi]
        g = self.margin_rad - abs(theta)  # radians; positive inside the cone

        # end episode on safety failure
        if g < 0.0:
            terminated = True

        # logging
        info = dict(info or {})
        info.update({
            "g_total": g,
            "theta_rad": theta,
            "margin_rad": self.margin_rad,
            "safe": float(g >= 0.0),
        })
        # NOTE: reward is the safety margin
        return obs, float(g), terminated, truncated, info


def make_pendulum_safety(margin_deg: float = 30.0, render_mode=None):
    env = gym.make("Pendulum-v1", render_mode=render_mode)
    env = PendulumSafety(env, margin_deg=margin_deg)
    return env


# Gym registration so you can call gym.make("SafetyPendulum-v1")
register(
    id="SafetyPendulum-v1",
    entry_point="safety_sb3.integrations.rl_zoo3.safety_envs.classic_control:make_pendulum_safety",
    kwargs={"margin_deg": 30.0},
)
