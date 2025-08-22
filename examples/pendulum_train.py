"""
Adversarial disturbance test on Pendulum-v1 (safety-only).

- Train SafetySAC where env reward = safety margin g(s).
- Start eval from upright (theta=0).
- Apply an adversarial torque that pushes the pendulum away from upright.
- Compare unshielded vs shielded rollouts and save GIFs.
"""

import os, sys
import gymnasium as gym
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safety_sb3 import SafetySAC  # or: from safety_sac import SafetySAC


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
        return obs, float(g), terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def train():
    # --- Train SafetySAC with reward == g(s) via wrapper ---
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


if __name__ == "__main__":
    train()
