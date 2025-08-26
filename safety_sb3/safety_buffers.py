import torch as th
import numpy as np
from stable_baselines3.common.buffers import RolloutBuffer


class SafetyRolloutBuffer(RolloutBuffer):

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Largely follows the original RolloutBuffer's compute_returns_and_advantage_safety method
        from stable_baselines3.
        We replace the usual GAE with the safety Bellman backup.
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()  # type: ignore[assignment]

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                v_next = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                v_next = self.values[step + 1]

            # Safety Bellman backup
            g_t = self.rewards[step]
            v_to_go = np.minimum(g_t, v_next)
            delta = (
                1.0 - self.gamma * next_non_terminal
            ) * g_t + self.gamma * next_non_terminal * v_to_go  # ensures that the full gs is returned at terminal states
            delta -= self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        self.returns = self.advantages + self.values
