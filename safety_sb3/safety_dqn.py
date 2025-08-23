import numpy as np
import torch as th
import torch.nn.functional as F

from stable_baselines3.dqn.dqn import DQN


class SafetyDQN(DQN):
    """Safety DQN.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int) -> None:
        """Largely follows the original DQN train method from stable_baselines3.
        We use the safety Bellman backup.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )  # type: ignore[union-attr]

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)

                # Safety Bellman: 1-step TD target
                gs = replay_data.rewards  # immediate safety margin values g(s) from env rewards
                not_done = 1.0 - replay_data.dones
                v_to_go = th.minimum(gs, next_q_values)
                target_q_values = (1.0 - self.gamma) * gs + self.gamma * not_done * v_to_go

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
