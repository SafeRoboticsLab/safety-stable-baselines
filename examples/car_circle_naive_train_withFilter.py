# train_sac_circle_with_safety_filter.py - SAC training with SafetySAC safety filter
import os
import sys
import time
import wandb
import safety_gymnasium
import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from wandb.integration.sb3 import WandbCallback

# so imports work when running from /examples
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safety_sb3.safety_sac import SafetySAC


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
            q_value = self.safety_model.critic(obs_tensor, action_tensor)
            
            # For SafetySAC, the critic outputs the margin g(s,a)
            # We use the first critic (if there are multiple) and take mean if needed
            if isinstance(q_value, tuple):
                q_value = q_value[0]  # Take first critic
            
            safety_margin = q_value.cpu().numpy().item()
        
        self.total_steps += 1
        
        # Safety decision: if margin > self.epsilon, action is safe
        if safety_margin > self.epsilon:
            # Safe action - use proposed action
            final_action = action
        else:
            # Unsafe action - use SafetySAC actor instead
            with torch.no_grad():
                safe_action, _ = self.safety_model.predict(current_obs, deterministic=False)
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
    # ---------- configuration ----------
    # Safety filter threshold - higher values = more conservative filtering
    # epsilon = 0.0: only filter when margin <= 0 (unsafe)
    # epsilon > 0: filter when margin <= epsilon (more conservative)
    # epsilon < 0: only filter when margin < epsilon (less conservative)
    
    # test different values: 0.15, 0.1, 0.05, 0.0, -0.05, -0.1, -0.15
    EPSILON = -0.15
    
    # ---------- paths ----------
    # Include epsilon in run name for easy identification
    epsilon_str = f"eps{EPSILON:+.3f}".replace(".", "p").replace("-", "m").replace("+", "p")
    run_name = f"SAC_CarCircle2_WithFilter_{epsilon_str}_{int(time.time())}"
    logs_dir = f"./experiments/{run_name}/logs"
    ckpt_dir = f"./experiments/{run_name}/checkpoints"
    best_dir = f"./experiments/{run_name}/best"
    final_dir = f"./experiments/{run_name}/final"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    # Path to trained SafetySAC model
    # Update this path to point to your trained SafetySAC model
    safety_model_path = "./experiments/SafetySAC_CarCircle2_1758776721/final/car_circle2.zip"
    
    # Check if safety model exists
    if not os.path.exists(safety_model_path):
        print(f"Error: SafetySAC model not found at {safety_model_path}")
        print("Please train a SafetySAC model first using car_circle_safety_train.py")
        print("Or update the safety_model_path to point to your trained model.")
        sys.exit(1)

    # ---------- W&B ----------
    wandb_run = wandb.init(
        project="safety_sb3",
        entity="safe-princeton",
        name=run_name,
        config={
            "algo": "SAC_with_SafetyFilter",
            "env_id": "SafetyCarCircle2-v0",
            "safety_model_path": safety_model_path,
            "epsilon": EPSILON,
            "total_timesteps": 100_000,
            "lr": 3e-4,
            "buffer_size": 100_000,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.01,
        },
        sync_tensorboard=True,
        save_code=True,
    )

    # ---------- env ----------
    # Create base environment
    base_env = safety_gymnasium.make("SafetyCarCircle2-v0")
    base_env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(base_env)
    
    # Add observation storing wrapper first
    obs_storing_env = ObservationStoringWrapper(base_env)
    
    # Add safety filter wrapper
    env = SafetyFilterWrapper(obs_storing_env, safety_model_path, epsilon=EPSILON)
    env = Monitor(env)

    # Separate eval env with same safety filter
    base_eval_env = safety_gymnasium.make("SafetyCarCircle2-v0")
    base_eval_env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(base_eval_env)
    obs_storing_eval_env = ObservationStoringWrapper(base_eval_env)
    eval_env = SafetyFilterWrapper(obs_storing_eval_env, safety_model_path, epsilon=EPSILON)
    eval_env = Monitor(eval_env)

    # ---------- model ----------
    # Standard SAC (same as naive training)
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.01,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        ent_coef="auto",
        seed=0,
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
        save_freq=10_000,
        save_path=ckpt_dir,
        name_prefix="sac_car_circle2_withfilter",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )

    wb_cb = WandbCallback(
        gradient_save_freq=0,
        model_save_path=os.path.join(ckpt_dir, "wandb"),
        model_save_freq=0,
        verbose=2,
    )

    callbacks = CallbackList([eval_cb, ckpt_cb, wb_cb])

    # ---------- train ----------
    model.learn(
        total_timesteps=100_000,
        callback=callbacks,
        tb_log_name=run_name,
        log_interval=10,
    )

    # ---------- final save ----------
    final_path = os.path.join(final_dir, f"car_circle2_sac_withfilter_{epsilon_str}")
    model.save(final_path)
    print(f"Training complete! Saved final SAC+Filter (ε={EPSILON}) model to {final_path}.zip")

    # ---------- tidy up ----------
    env.close()
    eval_env.close()
    wandb_run.finish()
