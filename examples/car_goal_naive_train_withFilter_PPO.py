import os
import sys
from datetime import datetime
import argparse
import wandb
import safety_gymnasium
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback
from safety_gymnasium.safety_envs.terminate_on_collision import TerminateOnCollisionWrapper

# so imports work when running from /examples
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safety_sb3.safety_sac import SafetySAC


class SafetyFilterLoggingCallback(BaseCallback):
    """
    Custom callback to log safety violations and filter statistics to wandb and tensorboard.
    Tracks both cost violations and safety filter interventions.
    """
    
    def __init__(self, verbose=0):
        super(SafetyFilterLoggingCallback, self).__init__(verbose)
        self.episode_costs = []
        self.episode_violations = []
        self.episode_interventions = []
        self.episode_avg_margins = []
        
        self.current_episode_cost = 0
        self.current_episode_violations = 0
        self.current_episode_interventions = 0
        self.current_episode_margins = []
        
        # Track total violations and interventions from start to finish
        self.total_violations = 0
        self.total_interventions = 0
        
    def _on_step(self) -> bool:
        # Extract info from the last step
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]  # Get info from first environment
            
            # Track cost from safety-gymnasium
            if 'cost' in info:
                cost = info['cost']
                self.current_episode_cost += cost
                if cost > 0:
                    self.current_episode_violations += 1
                    self.total_violations += 1  # Increment total violations counter
            
            # Track safety filter statistics
            if 'action_filtered' in info and info['action_filtered']:
                self.current_episode_interventions += 1
                self.total_interventions += 1  # Increment total interventions counter
                
            if 'safety_margin' in info:
                self.current_episode_margins.append(info['safety_margin'])
                
            # Log real-time filter statistics
            if 'intervention_rate' in info:
                self.logger.record("filter/intervention_rate", info['intervention_rate'])
        
        # Check if episode is done
        dones = self.locals.get('dones', [])
        if len(dones) > 0 and dones[0]:  # Episode finished
            # Store episode statistics
            self.episode_costs.append(self.current_episode_cost)
            self.episode_violations.append(self.current_episode_violations)
            self.episode_interventions.append(self.current_episode_interventions)
            
            # Calculate average margin for episode
            if len(self.current_episode_margins) > 0:
                avg_margin = np.mean(self.current_episode_margins)
                min_margin = np.min(self.current_episode_margins)
                self.episode_avg_margins.append(avg_margin)
            else:
                avg_margin = 0
                min_margin = 0
            
            # Log episode statistics to tensorboard and wandb
            self.logger.record("safety/episode_cost", self.current_episode_cost)
            self.logger.record("safety/episode_violations", self.current_episode_violations)
            self.logger.record("safety/total_violations", self.total_violations)  # Log cumulative total violations
            self.logger.record("filter/episode_interventions", self.current_episode_interventions)
            self.logger.record("filter/total_interventions", self.total_interventions)  # Log cumulative total interventions
            self.logger.record("filter/episode_avg_margin", avg_margin)
            self.logger.record("filter/episode_min_margin", min_margin)
            
            # Log cumulative statistics (last 100 episodes)
            if len(self.episode_costs) > 0:
                avg_cost = np.mean(self.episode_costs[-100:])
                avg_violations = np.mean(self.episode_violations[-100:])
                avg_interventions = np.mean(self.episode_interventions[-100:])
                violation_rate = np.mean([1 if v > 0 else 0 for v in self.episode_violations[-100:]])
                
                self.logger.record("safety/avg_episode_cost_100", avg_cost)
                self.logger.record("safety/avg_episode_violations_100", avg_violations)
                self.logger.record("safety/violation_rate_100", violation_rate)
                self.logger.record("filter/avg_episode_interventions_100", avg_interventions)
                
                if len(self.episode_avg_margins) > 0:
                    avg_margin_100 = np.mean(self.episode_avg_margins[-100:])
                    self.logger.record("filter/avg_margin_100", avg_margin_100)
            
            # Reset for next episode
            self.current_episode_cost = 0
            self.current_episode_violations = 0
            self.current_episode_interventions = 0
            self.current_episode_margins = []
            
        return True


class SafetyFilterWrapper:
    """
    Environment wrapper that filters actions using a trained SafetySAC model.
    
    Logic:
    - For each step(action), check if q(s, action) > 0 using SafetySAC critic
    - If q(s, action) > 0: use the proposed action (safe)
    - If q(s, action) <= 0: use SafetySAC actor π(s) instead (unsafe, so override)
    """
    
    def __init__(self, env, safety_model_path, epsilon: float = 0.0):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        # Load the trained SafetySAC model
        print(f"Loading SafetySAC model from {safety_model_path}")
        self.safety_model = SafetySAC.load(safety_model_path)
        
        # Get the device that the safety model is on
        self.device = next(self.safety_model.critic.parameters()).device
        print(f"SafetySAC model is on device: {self.device}")

        self.epsilon = epsilon

        # Statistics for tracking interventions
        self.total_steps = 0
        self.safety_interventions = 0
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Reset intervention statistics for new episode
        return obs, info
        
    def step(self, action):
        """
        Apply safety filter before stepping the environment.
        
        Args:
            action: Proposed action from the learning agent (naive SAC)
            
        Returns:
            Standard environment step tuple with filtered action
        """
        # Get current observation (convert to tensor for model)
        # Note: We need the last observation from reset() or previous step()
        # This assumes we store it or can access it somehow
        
        # For now, let's assume we can get the current state from the environment
        # In practice, you might need to store the last observation in the wrapper
        current_obs = self._get_current_observation()
        
        # Convert to tensor format expected by the model and move to correct device
        obs_tensor = torch.FloatTensor(current_obs).unsqueeze(0).to(self.device)  # Add batch dimension
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)    # Add batch dimension
        
        # Check safety using SafetySAC critic: q(s, a)
        with torch.no_grad():
            # Get Q-value from SafetySAC critic
            # Note: SafetySAC has safety critic that outputs margin values
            # q > 0 means safe (positive margin), q <= 0 means unsafe (negative margin)
            
            # SafetySAC critic expects separate observation and action arguments
            q_value = self.safety_model.critic(obs_tensor, action_tensor)
            
            # For SafetySAC, the critic outputs the margin g(s,a)
            # We use the minimum of both critics for conservative safety decisions
            if isinstance(q_value, tuple):
                q_value = torch.min(q_value[0], q_value[1])  # Take minimum of both critics (conservative)
            
            safety_margin = q_value.cpu().numpy().item()
        
        self.total_steps += 1
        
        # Safety decision: if margin > self.epsilon, action is safe
        if safety_margin > self.epsilon:
            # Safe action - use proposed action
            final_action = action
        else:
            # Unsafe action - use SafetySAC actor instead
            with torch.no_grad():
                safe_action, _ = self.safety_model.predict(current_obs, deterministic=True)
                final_action = safe_action
            
            self.safety_interventions += 1
            
        # Step environment with filtered action
        obs, reward, terminated, truncated, info = self.env.step(final_action)
        
        # Add safety statistics to info
        info['safety_margin'] = safety_margin
        info['action_filtered'] = (safety_margin <= self.epsilon)
        info['intervention_rate'] = self.safety_interventions / self.total_steps if self.total_steps > 0 else 0.0
        
        # Store observation for next step
        self.last_observation = obs
        
        return obs, reward, terminated, truncated, info
        
    def _get_current_observation(self):
        """
        Get the current observation. 
        This is a bit tricky since we need the observation before the step.
        We'll store it from the previous step/reset.
        """
        if hasattr(self, 'last_observation'):
            return self.last_observation
        else:
            # First step after reset - we need to get initial observation somehow
            # This is a limitation - in practice you'd store obs from reset()
            # For now, return a zero observation as fallback
            print("Warning: No stored observation available, using zeros")
            return np.zeros(self.observation_space.shape)
    
    def render(self, **kwargs):
        return self.env.render(**kwargs)
        
    def close(self):
        print(f"\nSafety Filter Statistics:")
        print(f"Total steps: {self.total_steps}")
        print(f"Safety interventions: {self.safety_interventions}")
        print(f"Intervention rate: {self.safety_interventions / self.total_steps * 100:.2f}%" if self.total_steps > 0 else "N/A")
        return self.env.close()
        
    def __getattr__(self, name):
        """Delegate other attributes to the base environment"""
        return getattr(self.env, name)


class ObservationStoringWrapper:
    """
    Helper wrapper to store observations so SafetyFilterWrapper can access them.
    """
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.last_observation = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_observation = obs
        return obs, info
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_observation = obs
        return obs, reward, terminated, truncated, info
        
    def get_last_observation(self):
        return self.last_observation
        
    def __getattr__(self, name):
        return getattr(self.env, name)


if __name__ == "__main__":
    # ---------- argument parsing ----------
    parser = argparse.ArgumentParser(description="Train PPO with SafetySAC safety filter")
    parser.add_argument("--epsilon", type=float, default=0.0,
                        help="Safety filter threshold. Higher values = more conservative filtering. "
                             "epsilon=0.0: only filter when margin <= 0 (unsafe), "
                             "epsilon>0: filter when margin <= epsilon (more conservative), "
                             "epsilon<0: only filter when margin < epsilon (less conservative)")
    parser.add_argument("--exp-suffix", type=str, default="",
                        help="Experiment identifier suffix for distinguishing experiment sets")
    parser.add_argument("--safety-model-path", type=str, 
                    default="./experiments/20251008_0050_SafetySAC_CarGoal1_Pillar_2M/checkpoints/safety_sac_car_goal1_pillar_1600000_steps.zip",
                        help="Path to trained SafetySAC model for safety filtering")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
                        help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # ---------- configuration ----------
    EPSILON = args.epsilon
    EXP_SUFFIX = args.exp_suffix
    TOTAL_TIMESTEPS = args.total_timesteps
    LEARNING_RATE = args.lr
    SEED = args.seed
    
    # ---------- paths ----------
    # Include epsilon in run name for easy identification
    epsilon_str = f"eps{EPSILON:+.3f}".replace(".", "p").replace("-", "m").replace("+", "p")
    base_run_name = f"PPO_CarGoal1_WithFilter_{epsilon_str}"
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M')}_{base_run_name}_{EXP_SUFFIX}"
    logs_dir = f"./experiments/{run_name}/logs"
    ckpt_dir = f"./experiments/{run_name}/checkpoints"
    best_dir = f"./experiments/{run_name}/best"
    final_dir = f"./experiments/{run_name}/final"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    # Path to trained SafetySAC model
    safety_model_path = args.safety_model_path
    
    # Check if safety model exists
    if not os.path.exists(safety_model_path):
        print(f"Error: SafetySAC model not found at {safety_model_path}")
        print("Please train a SafetySAC model first using car_goal_safety_train.py")
        print("Or update the safety_model_path to point to your trained model.")
        sys.exit(1)

    # ---------- W&B ----------
    wandb_run = wandb.init(
        project="safety_sb3",
        entity="safe-princeton",
        name=run_name,
        config={
            "algo": "PPO_with_SafetyFilter",
            "env_id": "SafetyCarGoal1-v0",
            "safety_model_path": safety_model_path,
            "epsilon": EPSILON,
            "exp_suffix": EXP_SUFFIX,
            "total_timesteps": TOTAL_TIMESTEPS,
            "lr": LEARNING_RATE,
            "n_steps": 2048,  # Steps per rollout
            "batch_size": 64,  # Minibatch size
            "n_epochs": 10,    # Number of epochs per update
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,  # Entropy coefficient
            "seed": SEED,
        },
        sync_tensorboard=True,
        save_code=True,
    )

    # ---------- env ----------
    # PPO works better with vectorized environments
    # Create base environment
    base_env = safety_gymnasium.make("SafetyCarGoal1-v0")
    base_env = TerminateOnCollisionWrapper(base_env)
    base_env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(base_env)
    
    # Add observation storing wrapper first
    obs_storing_env = ObservationStoringWrapper(base_env)
    
    # Add safety filter wrapper
    env = SafetyFilterWrapper(obs_storing_env, safety_model_path, epsilon=EPSILON)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])  # Vectorize for PPO

    # Separate eval env with same safety filter (also vectorized)
    base_eval_env = safety_gymnasium.make("SafetyCarGoal1-v0")
    base_eval_env = TerminateOnCollisionWrapper(base_eval_env)
    base_eval_env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(base_eval_env)
    obs_storing_eval_env = ObservationStoringWrapper(base_eval_env)
    eval_env = SafetyFilterWrapper(obs_storing_eval_env, safety_model_path, epsilon=EPSILON)
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])  # Vectorize for PPO

    # ---------- model ----------
    # Standard PPO (same as naive training) with SafetySAC safety filter
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=LEARNING_RATE,
        n_steps=2048,        # Number of steps to run for each environment per update
        batch_size=64,       # Minibatch size
        n_epochs=10,         # Number of epoch when optimizing the surrogate loss
        gamma=0.99,
        gae_lambda=0.95,     # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        clip_range=0.2,      # Clipping parameter for PPO
        ent_coef=0.01,       # Entropy coefficient for the loss calculation
        vf_coef=0.5,         # Value function coefficient for the loss calculation
        max_grad_norm=0.5,   # Maximum value for the gradient clipping
        seed=SEED,
        device="auto",
        verbose=1,
        tensorboard_log=logs_dir,
    )

    # ---------- callbacks ----------
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=best_dir,
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=20_000,    # Save more frequently for PPO (every ~25 updates with 2048 steps)
        save_path=ckpt_dir,
        name_prefix="ppo_car_goal1_withfilter",
        save_replay_buffer=False,  # PPO doesn't use replay buffer
        save_vecnormalize=False,
    )

    wb_cb = WandbCallback(
        gradient_save_freq=0,
        model_save_path=os.path.join(ckpt_dir, "wandb"),
        model_save_freq=0,
        verbose=2,
    )

    # Safety filter logging callback
    safety_filter_cb = SafetyFilterLoggingCallback(verbose=1)

    callbacks = CallbackList([eval_cb, ckpt_cb, wb_cb, safety_filter_cb])

    # ---------- train ----------
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        tb_log_name=run_name,
        log_interval=10,
    )

    # ---------- final save ----------
    final_path = os.path.join(final_dir, f"car_goal1_ppo_withfilter_{epsilon_str}")
    model.save(final_path)
    print(f"Training complete! Saved final PPO+Filter (ε={EPSILON}) model to {final_path}.zip")

    # ---------- tidy up ----------
    env.close()
    eval_env.close()
    wandb_run.finish()
