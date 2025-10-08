import os
import sys
from datetime import datetime
import argparse
import wandb
import safety_gymnasium
import numpy as np
import torch
import copy
import mujoco

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback
from safety_gymnasium.safety_envs.terminate_on_collision import TerminateOnCollisionWrapper

# so imports work when running from /examples
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safety_sb3.safety_sac import SafetySAC


class SafetyRolloutLoggingCallback(BaseCallback):
    """
    Custom callback to log safety violations and rollout filter statistics to wandb and tensorboard.
    Tracks both cost violations and safety filter interventions.
    """
    
    def __init__(self, verbose=0):
        super(SafetyRolloutLoggingCallback, self).__init__(verbose)
        self.episode_costs = []
        self.episode_violations = []
        self.episode_interventions = []
        self.episode_rollout_lengths = []
        
        self.current_episode_cost = 0
        self.current_episode_violations = 0
        self.current_episode_interventions = 0
        self.current_episode_rollout_lengths = []
        
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
                
            if 'rollout_length' in info:
                self.current_episode_rollout_lengths.append(info['rollout_length'])
                
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
            
            # Calculate average rollout length for episode
            if len(self.current_episode_rollout_lengths) > 0:
                avg_rollout_length = np.mean(self.current_episode_rollout_lengths)
                max_rollout_length = np.max(self.current_episode_rollout_lengths)
                self.episode_rollout_lengths.append(avg_rollout_length)
            else:
                avg_rollout_length = 0
                max_rollout_length = 0
            
            # Log episode statistics to tensorboard and wandb
            self.logger.record("safety/episode_cost", self.current_episode_cost)
            self.logger.record("safety/episode_violations", self.current_episode_violations)
            self.logger.record("safety/total_violations", self.total_violations)  # Log cumulative total violations
            self.logger.record("filter/episode_interventions", self.current_episode_interventions)
            self.logger.record("filter/total_interventions", self.total_interventions)  # Log cumulative total interventions
            self.logger.record("filter/episode_avg_rollout_length", avg_rollout_length)
            self.logger.record("filter/episode_max_rollout_length", max_rollout_length)
            
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
                
                if len(self.episode_rollout_lengths) > 0:
                    avg_rollout_length_100 = np.mean(self.episode_rollout_lengths[-100:])
                    self.logger.record("filter/avg_rollout_length_100", avg_rollout_length_100)
            
            # Reset for next episode
            self.current_episode_cost = 0
            self.current_episode_violations = 0
            self.current_episode_interventions = 0
            self.current_episode_rollout_lengths = []
            
        return True


class SafetyRolloutFilter:
    """
    Environment wrapper that filters actions using rollout-based safety checking.
    
    Logic:
    1. Create a dedicated rollout environment (reused for efficiency)
    2. Copy exact MuJoCo state (qpos, qvel, etc.) from main env to rollout env
    3. Take 1 step forward using the proposed action
    4. For the next H-1 steps, apply trained safety policy continuously
    5. If any violation occurs during rollout OR velocity becomes ~0, evaluate safety
    6. Use proposed action if safe, otherwise use safety policy action
    """
    
    def __init__(self, env, safety_model_path, horizon: int = 10, velocity_threshold: float = 0.1):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        # Load the trained SafetySAC model for safety policy
        print(f"Loading SafetySAC model from {safety_model_path}")
        self.safety_model = SafetySAC.load(safety_model_path)
        
        # Get the device that the safety model is on
        self.device = next(self.safety_model.critic.parameters()).device
        print(f"SafetySAC model is on device: {self.device}")

        self.horizon = horizon
        self.velocity_threshold = velocity_threshold

        # Statistics for tracking interventions
        self.total_steps = 0
        self.safety_interventions = 0
        
        # Create a dedicated rollout environment (reused for efficiency)
        self.rollout_env = self._create_rollout_env()
        
        print(f"RolloutFilter configured: horizon={horizon}, velocity_threshold={velocity_threshold}")
        print("Dedicated rollout environment created for efficient state copying")
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Store initial observation immediately for rollout filter
        self.last_observation = obs
        # Reset intervention statistics for new episode
        return obs, info
        
    def step(self, action):
        """
        Apply rollout-based safety filter before stepping the environment.
        
        Args:
            action: Proposed action from the learning agent
            
        Returns:
            Standard environment step tuple with filtered action
        """
        # Get current observation
        current_obs = self._get_current_observation()
        
        # Perform rollout safety check
        is_safe, rollout_length = self._rollout_safety_check(current_obs, action)
        
        self.total_steps += 1
        
        # Safety decision based on rollout
        if is_safe:
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
        info['action_filtered'] = not is_safe
        info['rollout_length'] = rollout_length
        info['intervention_rate'] = self.safety_interventions / self.total_steps if self.total_steps > 0 else 0.0
        
        # Store NEW observation for next step
        self.last_observation = obs
        
        return obs, reward, terminated, truncated, info

    def _init_rollout_from_main(self):
        # time_limit wrapper reset)
        self.rollout_env.soft_reset()

        # ---- 2) copy physics state through World API
        main_world = self.env.unwrapped.task.world
        rollout_world = self.rollout_env.unwrapped.task.world

        state = main_world.get_state()      # {'time','qpos','qvel','act' or None}
        rollout_world.set_state(state)
        mujoco.mj_forward(rollout_world.model, rollout_world.data)
    
    def _rollout_safety_check(self, current_obs, proposed_action, verbose=False):
        """
        Perform rollout-based safety checking with proper MuJoCo state copying.
        
        Args:
            current_obs: Current observation state
            proposed_action: Action to test for safety
            
        Returns:
            tuple: (is_safe: bool, rollout_length: int)
        """
        try:
            # Copy exact MuJoCo state from main environment to rollout environment
            self._init_rollout_from_main()
            
            # Step 1: Take one step with the proposed action
            rollout_obs, rollout_reward, rollout_terminated, rollout_truncated, rollout_info = self.rollout_env.step(proposed_action)
            
            # Check if first step already causes violation
            if self._has_violation(rollout_info):
                if verbose: print("1, violation")
                return False, 1
            
            # Check if we've reached a safe state (low velocity)
            if self._is_velocity_safe(rollout_obs):
                if verbose: print("1, safe")
                return True, 1
            
            # If episode ended, consider it safe if no violation occurred
            if rollout_terminated or rollout_truncated:
                if verbose: print("1, timeout")
                return False, 1
            
            # Step 2: Continue rollout for H-1 steps using safety policy
            for rollout_step in range(2, self.horizon + 1):
                # Get safe action from safety policy
                with torch.no_grad():
                    safety_action, _ = self.safety_model.predict(rollout_obs, deterministic=True)
                
                # Take step with safety policy
                rollout_obs, rollout_reward, rollout_terminated, rollout_truncated, rollout_info = self.rollout_env.step(safety_action)
                
                # Check for violation during rollout
                if self._has_violation(rollout_info):
                    if verbose: print(f"{rollout_step}, violation")
                    return False, rollout_step
                
                # Check if we've reached a safe state (low velocity)
                if self._is_velocity_safe(rollout_obs):
                    if verbose: print(f"{rollout_step}, safe")
                    return True, rollout_step
                
                # If episode ended naturally, consider it safe
                if rollout_terminated or rollout_truncated:
                    if verbose: print(f"{rollout_step}, timeout")
                    return False, rollout_step
            
            # If we completed the full horizon without violations, it's safe
            if verbose: print(f"{self.horizon}, safe")
            return True, self.horizon
            
        except Exception as e:
            print(f"Error during rollout safety check: {e}")
            # In case of error, be conservative and reject the action
            return False, 1
    
    def _create_rollout_env(self):
        """
        Create a dedicated rollout environment that will be reused for efficiency.
        This is called once during initialization.
        """
        try:
            # Create a fresh environment instance
            rollout_env = safety_gymnasium.make("SafetyCarCircle2-v0", render_mode=None)
            rollout_env = TerminateOnCollisionWrapper(rollout_env)
            rollout_env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(rollout_env)
            
            # Reset the environment to initialize it
            rollout_env.reset()
            
            print("Rollout environment created successfully")
            return rollout_env
            
        except Exception as e:
            print(f"Error creating rollout environment: {e}")
            raise
    
    def _has_violation(self, info):
        """
        Check if there's a safety violation in the step info.
        
        Args:
            info: Environment info dictionary
            
        Returns:
            bool: True if violation detected
        """
        # Check for cost violations
        if 'cost' in info and info['cost'] > 0:
            return True
        
        # Check for specific violation types
        violation_keys = ['cost_out_of_boundary', 'cost_pillars', 'cost_sigwalls', 'cost_hazards', 'cost_sum']
        for key in violation_keys:
            if key in info and info[key] > 0:
                return True
        
        return False
    
    def _is_velocity_safe(self, obs):
        """
        Check if the agent velocity is low enough to be considered safe.
        
        Args:
            obs: Current observation
            
        Returns:
            bool: True if velocity is below threshold
        """
        try:
            # For car environments, velocity is typically in the observation
            # This is environment-specific and might need adjustment
            
            # Assuming velocity components are in the observation
            # You might need to adjust indices based on your observation space
            if len(obs) >= 6:  # Typical car obs: [x, y, theta, vx, vy, vtheta, ...]
                vx = obs[3] if len(obs) > 3 else 0
                vy = obs[4] if len(obs) > 4 else 0
                velocity_magnitude = np.sqrt(vx**2 + vy**2)
                return velocity_magnitude < self.velocity_threshold
            
            return False
            
        except Exception as e:
            print(f"Warning: Could not extract velocity from observation: {e}")
            return False
        
    def _get_current_observation(self):
        """
        Get the current observation.
        """
        if hasattr(self, 'last_observation') and self.last_observation is not None:
            return self.last_observation
        else:
            # This should only happen on the very first step after environment creation
            if hasattr(self.env, 'get_last_observation'):
                obs = self.env.get_last_observation()
                if obs is not None:
                    return obs
            
            # Last resort - this indicates a serious problem
            print("ERROR: No stored observation available! This will cause poor safety decisions.")
            print("Consider calling env.reset() before using the rollout filter.")
            return np.zeros(self.observation_space.shape)
    
    def render(self, **kwargs):
        return self.env.render(**kwargs)
        
    def close(self):
        print(f"\nSafety Rollout Filter Statistics:")
        print(f"Total steps: {self.total_steps}")
        print(f"Safety interventions: {self.safety_interventions}")
        print(f"Intervention rate: {self.safety_interventions / self.total_steps * 100:.2f}%" if self.total_steps > 0 else "N/A")
        
        # Clean up rollout environment
        if hasattr(self, 'rollout_env') and self.rollout_env is not None:
            try:
                self.rollout_env.close()
                print("Rollout environment closed successfully")
            except Exception as e:
                print(f"Warning: Error closing rollout environment: {e}")
        
        return self.env.close()
        
    def __getattr__(self, name):
        """Delegate other attributes to the base environment"""
        return getattr(self.env, name)


class ObservationStoringWrapper:
    """
    Helper wrapper to store observations so SafetyRolloutFilter can access them.
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
    parser = argparse.ArgumentParser(description="Train PPO with SafetySAC rollout filter on CarCircle")
    parser.add_argument("--horizon", type=int, default=10,
                        help="Rollout horizon for safety checking (number of steps to simulate)")
    parser.add_argument("--velocity-threshold", type=float, default=0.1,
                        help="Velocity threshold below which the system is considered safe")
    parser.add_argument("--exp-suffix", type=str, default="",
                        help="Experiment identifier suffix for distinguishing experiment sets")
    parser.add_argument("--safety-model-path", type=str, 
                        default="./experiments/20250926_1953_SafetySAC_CarCircle2_lr1em5/final/car_circle2.zip",
                        help="Path to trained SafetySAC model for safety policy")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
                        help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # ---------- configuration ----------
    HORIZON = args.horizon
    VELOCITY_THRESHOLD = args.velocity_threshold
    EXP_SUFFIX = args.exp_suffix
    TOTAL_TIMESTEPS = args.total_timesteps
    LEARNING_RATE = args.lr
    SEED = args.seed
    
    # ---------- paths ----------
    base_run_name = f"PPO_CarCircle2_WithRolloutFilter_h{HORIZON}_vt{VELOCITY_THRESHOLD}"
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
        print("Please train a SafetySAC model first using car_circle_safety_train.py")
        print("Or update the safety_model_path to point to your trained model.")
        sys.exit(1)

    # ---------- W&B ----------
    wandb_run = wandb.init(
        project="safety_sb3",
        entity="safe-princeton",
        name=run_name,
        config={
            "algo": "PPO_with_RolloutFilter",
            "env_id": "SafetyCarCircle2-v0",
            "safety_model_path": safety_model_path,
            "horizon": HORIZON,
            "velocity_threshold": VELOCITY_THRESHOLD,
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
    base_env = safety_gymnasium.make("SafetyCarCircle2-v0", render_mode=None)
    base_env = TerminateOnCollisionWrapper(base_env)
    base_env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(base_env)
    
    # Add observation storing wrapper first
    obs_storing_env = ObservationStoringWrapper(base_env)
    
    # Add rollout safety filter wrapper
    env = SafetyRolloutFilter(obs_storing_env, safety_model_path, 
                              horizon=HORIZON, velocity_threshold=VELOCITY_THRESHOLD)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])  # Vectorize for PPO

    # Separate eval env with same rollout filter (also vectorized)
    base_eval_env = safety_gymnasium.make("SafetyCarCircle2-v0")
    base_eval_env = TerminateOnCollisionWrapper(base_eval_env)
    base_eval_env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(base_eval_env)
    obs_storing_eval_env = ObservationStoringWrapper(base_eval_env)
    eval_env = SafetyRolloutFilter(obs_storing_eval_env, safety_model_path,
                                   horizon=HORIZON, velocity_threshold=VELOCITY_THRESHOLD)
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])  # Vectorize for PPO

    # ---------- model ----------
    # Standard PPO (same as naive training) with SafetySAC rollout filter
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
        save_freq=50_000,    # Save more frequently for PPO (every ~25 updates with 2048 steps)
        save_path=ckpt_dir,
        name_prefix="ppo_car_circle2_withrolloutfilter",
        save_replay_buffer=False,  # PPO doesn't use replay buffer
        save_vecnormalize=False,
    )

    wb_cb = WandbCallback(
        gradient_save_freq=0,
        model_save_path=os.path.join(ckpt_dir, "wandb"),
        model_save_freq=0,
        verbose=2,
    )

    # Safety rollout filter logging callback
    safety_rollout_cb = SafetyRolloutLoggingCallback(verbose=1)

    callbacks = CallbackList([eval_cb, ckpt_cb, wb_cb, safety_rollout_cb])

    # ---------- train ----------
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        tb_log_name=run_name,
        log_interval=10,
    )

    # ---------- final save ----------
    final_path = os.path.join(final_dir, f"car_circle2_ppo_withrolloutfilter_h{HORIZON}_vt{VELOCITY_THRESHOLD}")
    model.save(final_path)
    print(f"Training complete! Saved final PPO+RolloutFilter (h={HORIZON}, vt={VELOCITY_THRESHOLD}) model to {final_path}.zip")

    # ---------- tidy up ----------
    env.close()
    eval_env.close()
    wandb_run.finish()