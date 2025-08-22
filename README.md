# safety-stable-baselines
Lightweight add-on for [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) that implements the Safety RL (Fisac et al., ICRA’19, Wang and Hu et al., WAFR'24).

## TODO List

- [ ] Check if it is correct to add ent reg to next_q_values
- [ ] Decide if we should max or min over all critics targets to get the target Q (for the avoid-only case)

This repo provides:
- `SafetySAC(SAC)`: a drop-in subclass that learns a **safety critic** using the safety RL (Fisac et al., ICRA'19)
- `SafetyPPO(PPO)`: [WIP] PPO version of safety RL
- `MagicsSAC(SafetySAC)`: [WIP] SAC version of MAGICS (Wang and Hu et al., WAFR'24)
- `MagicsPPO(SafetyPPO)`: [WIP] PPO version of MAGICS (Wang and Hu et al., WAFR'24)

**Design principle: keep upstream SB3 untouched. This lives in a separate repo and still feels native to SB3 users.**

## Installation
```bash
pip install -U torch stable-baselines3 gymnasium
pip install -e .
```

## Quick start
```python
import os, sys
import gymnasium as gym
import numpy as np
from safety_sac import SafetySAC


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
```


## References
- Fisac et al., “[Bridging Hamilton-Jacobi Safety Analysis and Reinforcement Learning](https://ieeexplore.ieee.org/document/8794107),” ICRA 2019.
- Wang and Hu et al., "[MAGICS: Adversarial RL with Minimax Actors Guided by Implicit Critic Stackelberg for Convergent Neural Synthesis of Robot Safety](https://www.algorithmic-robotics.org/papers/45_MAGICS_Adversarial_RL_with_.pdf)," WAFR 2024.
