"""
Adversarial disturbance test on Pendulum-v1 (safety-only).

- Train SafetySAC where env reward = safety margin g(s).
- Start eval from upright (theta=0).
- Apply an adversarial torque that pushes the pendulum away from upright.
- Compare unshielded vs shielded rollouts and save GIFs.
"""

import os, sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType, Wrapper

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safety_sb3 import SafetySAC, SafetyDQN  # or: from safety_sac import...


class PendulumSafety(gym.Wrapper):
    """
    Reward == safety margin g(s) for avoid:
        g(s) = (pi/6) - |theta|
    where theta is the pole angle (0 = upright).
    Safe iff g(s) >= 0, i.e., |theta| <= 30 degrees.
    """

    def __init__(self, env: gym.Env, angle_limit=np.pi / 6):
        super().__init__(env)
        self.angle_limit = float(angle_limit)

    @staticmethod
    def _theta_from_obs(obs: np.ndarray) -> float:
        # Pendulum obs = [cos(theta), sin(theta), theta_dot]
        c, s = obs[0], obs[1]
        return float(np.arctan2(s, c))  # [-pi, pi]

    def step(self, action):
        obs, _base_reward, terminated, truncated, info = self.env.step(action)

        theta = self._theta_from_obs(obs)
        g = self.angle_limit - abs(theta)  # g(s): positive inside ±30°, negative outside

        # overwrite terminated when g(s) < 0
        if g < 0.0:
            terminated = True  # end episode on safety breach

        return obs, float(g), terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class DiscretizeActionWrapper(Wrapper):
    """Map Discrete(N) to original Box action via N bins."""

    def __init__(self, env, n_bins=11):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.Box) and env.action_space.shape == (1,)
        self.n_bins = n_bins
        self.action_space = spaces.Discrete(n_bins)
        low, high = float(env.action_space.low[0]), float(env.action_space.high[0])
        self.bins = np.linspace(low, high, n_bins)

    def step(self, action: ActType):
        cont = np.array([self.bins[int(action)]], dtype=np.float32)
        obs, rew, term, trunc, info = self.env.step(cont)
        return obs, rew, term, trunc, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def train_SAC():
    base_env = gym.make("Pendulum-v1")
    env = PendulumSafety(base_env)

    model = SafetySAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.01,
        gamma=0.995,  # safety discount
        train_freq=(1, "step"),
        gradient_steps=1,
        ent_coef="auto",
        seed=0,
        device="auto",
        verbose=1,
    )

    model.learn(100_000)

    # ---- SAVE ----
    save_dir = "./examples/models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "pendulum")
    model.save(save_path)
    print(f"Training complete! Saved trained SafetySAC model to {save_path}.zip")


def train_DQN():
    base_env = gym.make("Pendulum-v1")
    env = PendulumSafety(base_env)
    env = DiscretizeActionWrapper(env, n_bins=21)

    model = SafetyDQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.01,
        gamma=0.995,  # safety discount
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10_000,  # update target network every X steps
        exploration_fraction=0.1,  # fraction of total timesteps for ε-greedy decay
        exploration_final_eps=0.05,  # final ε value
        seed=0,
        device="auto",
        verbose=1,
    )

    model.learn(100_000)

    # ---- SAVE ----
    save_dir = "./examples/models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "pendulum_disc")
    model.save(save_path)
    print(f"Training complete! Saved trained SafetyDQN model to {save_path}.zip")


if __name__ == "__main__":
    train_SAC()
