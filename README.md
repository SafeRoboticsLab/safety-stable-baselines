# safety-stable-baselines
Lightweight add-on for [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) that implements the Safety RL (Fisac et al., ICRA’19).

## TODO List

- [ ] Check if for SAC it is correct to add ent reg to next_q_values

This repo provides:
- `SafetySAC(SAC)`: SAC version of safety RL - off policy
- `SafetyDQN(DQN)`: DQN version of safety RL - off policy
- `SafetyPPO(PPO)`: PPO version of safety RL - on policy
- `SafetyA2C(A2C)`: A2C version of safety RL - on policy

**Design principle: keep upstream SB3 untouched. This lives in a separate repo and still feels native to SB3 users.**

## Installation
We need to enforce python 3.10 so that `safety-gymnasium` works. This repository includes [`safety-gymnasium` from Safe Robotics Lab (forked from original)](https://github.com/SafeRoboticsLab/safety-gymnasium) as a git submodule, configured to track the `safety_sb3` branch.

```bash
# Create conda env
conda create --name safety_sb3 python=3.10
conda activate safety_sb3

# Clone this repo with submodules
git clone --recurse-submodules git@github.com:SafeRoboticsLab/safety-stable-baselines.git
cd safety-stable-baselines

# Install safety-sb3
pip install -U torch stable-baselines3 gymnasium
pip install -e .

# Install rl_baselines3_zoo deps
cd integrations/rl_baselines3_zoo
pip install -e .

# Install safety-gymnasium
cd integrations/safety-gymnasium
pip install -e .
```

### To update safety-gymnasium later:
```bash
# Update submodule to latest commit on safety_sb3 branch
git submodule update --remote integrations/safety-gymnasium
```

## Quick start and example
```python
import os, sys
import gymnasium as gym
import numpy as np
from safety_sb3 import SafetySAC


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
