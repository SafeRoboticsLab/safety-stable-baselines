#!/usr/bin/env python3
"""
Script to load and replay a trained SAC model in the original safety-gymnasium Circle environment.
Supports testing with different safety filter configurations:
1. No filter (default)
2. Safety value filter (based on Q-value margin)
3. Safety rollout filter (based on rollout simulation)
"""

import os
import sys
import argparse
import time
import safety_gymnasium
import torch
import numpy as np
import mujoco
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from safety_gymnasium.safety_envs.terminate_on_collision import TerminateOnCollisionWrapper

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safety_sb3.safety_sac import SafetySAC

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
    Helper wrapper to store observations so SafetyFilter env wrapper can access them.
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


def print_model_architecture(model, env):
    """
    Print detailed architecture information of the SAC model.
    
    Args:
        model: Trained SAC model
        env: Environment (for getting observation/action space info)
    """
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE INFORMATION")
    print("="*80)
    
    # Input/Output shapes
    print("\nINPUT/OUTPUT SHAPES:")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Observation shape: {env.observation_space.shape}")
    print(f"   Observation dimension: {env.observation_space.shape[0]}")
    print(f"   Action space: {env.action_space}")
    print(f"   Action shape: {env.action_space.shape}")
    print(f"   Action dimension: {env.action_space.shape[0]}")
    
    # Actor network
    print("\nACTOR NETWORK (Policy):")
    print(f"   Type: {type(model.actor).__name__}")
    print(f"   Input dim: {env.observation_space.shape[0]}")
    print(f"   Output dim: {env.action_space.shape[0]} (mean and log_std)")
    
    if hasattr(model.actor, 'latent_pi'):
        print(f"\n   Latent Policy Network:")
        for i, layer in enumerate(model.actor.latent_pi):
            if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
                print(f"      Layer {i}: Linear({layer.in_features} -> {layer.out_features})")
            else:
                print(f"      Layer {i}: {layer}")
    
    if hasattr(model.actor, 'mu'):
        print(f"   Mean output layer: {model.actor.mu}")
    if hasattr(model.actor, 'log_std'):
        print(f"   Log std output layer: {model.actor.log_std}")
    
    # Critic network
    print("\nCRITIC NETWORK (Q-function):")
    print(f"   Type: {type(model.critic).__name__}")
    print(f"   Number of critics: {model.critic.n_critics if hasattr(model.critic, 'n_critics') else 'N/A'}")
    print(f"   Input dim: {env.observation_space.shape[0]} (obs) + {env.action_space.shape[0]} (action)")
    print(f"   Output dim: 1 (Q-value)")
    
    if hasattr(model.critic, 'q_networks') and len(model.critic.q_networks) > 0:
        print(f"\n   Q-Network 1:")
        for i, layer in enumerate(model.critic.q_networks[0]):
            if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
                print(f"      Layer {i}: Linear({layer.in_features} -> {layer.out_features})")
            else:
                print(f"      Layer {i}: {layer}")
        
        if len(model.critic.q_networks) > 1:
            print(f"\n   Q-Network 2 (same architecture)")
    
    # Policy kwargs
    print("\nPOLICY CONFIGURATION:")
    if hasattr(model, 'policy_kwargs') and model.policy_kwargs:
        for key, value in model.policy_kwargs.items():
            print(f"   {key}: {value}")
    else:
        print("   Using default configuration")
    
    # Additional model parameters
    print("\nTRAINING PARAMETERS:")
    print(f"   Learning rate: {model.learning_rate}")
    print(f"   Buffer size: {model.buffer_size}")
    print(f"   Batch size: {model.batch_size}")
    print(f"   Gamma (discount): {model.gamma}")
    print(f"   Tau (soft update): {model.tau}")
    if hasattr(model, 'ent_coef'):
        ent_coef = model.ent_coef
        if isinstance(ent_coef, torch.Tensor):
            ent_coef = ent_coef.item()
        print(f"   Entropy coefficient: {ent_coef}")
    
    # Total parameters
    total_params = sum(p.numel() for p in model.policy.parameters())
    trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"\nTOTAL PARAMETERS:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    
    print("="*80 + "\n")


def replay_sac_model(model_path: str, env_id: str = "SafetyCarCircle2-v0", 
                     num_episodes: int = 5, deterministic: bool = True, 
                     step_delay: float = 0.02, filter_mode: str = "none",
                     safety_model_path: str = None, filter_epsilon: float = 0.0,
                     rollout_horizon: int = 10, rollout_velocity_threshold: float = 0.1):
    """
    Load and replay a trained SAC model with optional safety filters.
    
    Args:
        model_path: Path to the saved model (.zip file)
        env_id: Safety-gymnasium environment ID
        num_episodes: Number of episodes to run
        deterministic: Use deterministic actions (True) or sample from policy (False)
        step_delay: Delay between steps for better visualization (seconds)
        filter_mode: Safety filter mode - "none", "value", or "rollout"
        safety_model_path: Path to trained SafetySAC model (required for filters)
        filter_epsilon: Safety margin threshold for value filter
        rollout_horizon: Rollout horizon for rollout filter
        rollout_velocity_threshold: Velocity threshold for rollout filter
    """
    print(f"Loading SAC model from: {model_path}")
    
    # Create original safety-gymnasium environment with rendering
    env = safety_gymnasium.make(env_id, render_mode="human")
    env = TerminateOnCollisionWrapper(env)
    env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)
    
    # Apply safety filter if requested
    if filter_mode == "value":
        if safety_model_path is None:
            print("Error: --safety-model is required for value filter mode")
            return
        if not os.path.exists(safety_model_path):
            print(f"Error: Safety model not found at {safety_model_path}")
            return
        print(f"\nApplying Safety Value Filter (epsilon={filter_epsilon})")
        env = ObservationStoringWrapper(env)
        env = SafetyFilterWrapper(env, safety_model_path, epsilon=filter_epsilon)
        env = Monitor(env)
        
    elif filter_mode == "rollout":
        if safety_model_path is None:
            print("Error: --safety-model is required for rollout filter mode")
            return
        if not os.path.exists(safety_model_path):
            print(f"Error: Safety model not found at {safety_model_path}")
            return
        print(f"\nApplying Safety Rollout Filter (horizon={rollout_horizon}, velocity_threshold={rollout_velocity_threshold})")
        env = ObservationStoringWrapper(env)
        env = SafetyRolloutFilter(env, safety_model_path, 
                                 horizon=rollout_horizon, 
                                 velocity_threshold=rollout_velocity_threshold)
        env = Monitor(env)
    
    # Load the trained SAC model
    try:
        model = SAC.load(model_path, env=env)
        print(f"Successfully loaded SAC model")
        
        # Print architecture information
        print_model_architecture(model, env)
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Make sure the model path is correct and the model was saved properly.")
        return
    
    print(f"\nStarting SAC replay with:")
    print(f"   Environment: {env_id}")
    print(f"   Filter mode: {filter_mode}")
    if filter_mode == "value":
        print(f"   Filter epsilon: {filter_epsilon}")
    elif filter_mode == "rollout":
        print(f"   Rollout horizon: {rollout_horizon}")
        print(f"   Velocity threshold: {rollout_velocity_threshold}")
    print(f"   Episodes: {num_episodes}")
    print(f"   Deterministic: {deterministic}")
    print(f"   Step delay: {step_delay}s")
    print(f"\nGoal: Navigate using trained policy {'with safety filter' if filter_mode != 'none' else 'without filter'}!")
    print(f"   Safety violations will be tracked via environment's built-in cost signal\n")
    
    episode_stats = []
    
    try:
        for episode in range(num_episodes):
            print(f"Episode {episode + 1}/{num_episodes}")
            
            obs, info = env.reset()
            episode_reward = 0.0
            episode_cost = 0.0  # Track safety violations via cost
            episode_steps = 0
            cost_violations = 0
            filter_interventions = 0
            
            while True:
                # Get action from trained SAC model
                action, _states = model.predict(obs, deterministic=deterministic)
                
                # Take step in environment
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Track statistics
                episode_reward += reward
                episode_steps += 1
                
                # Track cost/safety violations if available
                if 'cost' in info:
                    cost = info['cost']
                    episode_cost += cost
                    if cost > 0:
                        cost_violations += 1
                
                # Track filter interventions
                if 'action_filtered' in info and info['action_filtered']:
                    filter_interventions += 1
                
                # Render the environment
                env.render()
                
                # Add delay for better visualization
                if step_delay > 0:
                    time.sleep(step_delay)
                
                # Check if episode is done
                if terminated or truncated:
                    break
            
            # Episode summary
            final_info = {
                'episode': episode + 1,
                'reward': episode_reward,
                'cost': episode_cost,
                'steps': episode_steps,
                'cost_violations': cost_violations,
                'filter_interventions': filter_interventions,
                'terminated': terminated,
                'truncated': truncated
            }
            episode_stats.append(final_info)
            
            print(f"      Episode {episode + 1} complete!")
            print(f"      Reward: {episode_reward:.3f}")
            print(f"      Cost: {episode_cost:.3f}")
            print(f"      Steps: {episode_steps}")
            print(f"      Cost violations: {cost_violations}")
            if filter_mode != "none":
                print(f"      Filter interventions: {filter_interventions}")
                intervention_rate = (filter_interventions / episode_steps * 100) if episode_steps > 0 else 0
                print(f"      Intervention rate: {intervention_rate:.1f}%")
            print(f"      Reason: {'Task complete/Safety violation' if terminated else 'Timeout' if truncated else 'Complete'}")
            print()
            
            # Brief pause between episodes
            time.sleep(1.0)
    
    except KeyboardInterrupt:
        print(f"\nReplay interrupted by user")
    
    finally:
        env.close()
        
        # Print overall statistics
        if episode_stats:
            print(f"\nOverall Statistics ({len(episode_stats)} episodes):")
            avg_reward = sum(ep['reward'] for ep in episode_stats) / len(episode_stats)
            avg_cost = sum(ep['cost'] for ep in episode_stats) / len(episode_stats)
            avg_steps = sum(ep['steps'] for ep in episode_stats) / len(episode_stats)
            total_violations = sum(ep['cost_violations'] for ep in episode_stats)
            
            print(f"   Average reward: {avg_reward:.3f}")
            print(f"   Average cost: {avg_cost:.3f}")
            print(f"   Average steps: {avg_steps:.1f}")
            print(f"   Total cost violations: {total_violations}")
            
            if filter_mode != "none":
                total_interventions = sum(ep['filter_interventions'] for ep in episode_stats)
                total_steps = sum(ep['steps'] for ep in episode_stats)
                avg_intervention_rate = (total_interventions / total_steps * 100) if total_steps > 0 else 0
                print(f"   Total filter interventions: {total_interventions}")
                print(f"   Overall intervention rate: {avg_intervention_rate:.1f}%")
            
            # Safety rate (episodes with zero cost)
            safe_episodes = sum(1 for ep in episode_stats if ep['cost'] == 0)
            safety_rate = safe_episodes / len(episode_stats) * 100
            print(f"   Safety rate: {safety_rate:.1f}% ({safe_episodes}/{len(episode_stats)} episodes with zero cost)")



def main():
    parser = argparse.ArgumentParser(
        description="Replay a trained SAC model on safety-gymnasium with optional safety filters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Replay without filter (default)
  python replay_trained_sac_model_car_circle.py --model path/to/model.zip

  # Replay with safety value filter
  python replay_trained_sac_model_car_circle.py --model path/to/model.zip \\
      --filter value --safety-model path/to/safety_model.zip --epsilon 0.0

  # Replay with safety rollout filter
  python replay_trained_sac_model_car_circle.py --model path/to/model.zip \\
      --filter rollout --safety-model path/to/safety_model.zip \\
      --horizon 10 --velocity-threshold 0.1
        """
    )
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Path to the trained SAC model file (.zip)")
    parser.add_argument("--env", type=str, default="SafetyCarCircle2-v0",
                        choices=["SafetyCarCircle1-v0", "SafetyCarCircle2-v0", 
                                "SafetyPointCircle1-v0", "SafetyPointCircle2-v0"],
                        help="Safety-gymnasium environment ID")
    parser.add_argument("--episodes", "-e", type=int, default=5,
                        help="Number of episodes to run")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy (default: deterministic)")
    parser.add_argument("--delay", "-d", type=float, default=0.02,
                        help="Delay between steps for visualization (seconds)")
    
    # Filter options
    parser.add_argument("--filter", type=str, default="none",
                        choices=["none", "value", "rollout"],
                        help="Safety filter mode: 'none' (no filter), 'value' (Q-value based), 'rollout' (simulation based)")
    parser.add_argument("--safety-model", type=str, default=None,
                        help="Path to trained SafetySAC model (required for value/rollout filters)")
    
    # Value filter options
    parser.add_argument("--epsilon", type=float, default=0.0,
                        help="Safety margin threshold for value filter (default: 0.0)")
    
    # Rollout filter options
    parser.add_argument("--horizon", type=int, default=10,
                        help="Rollout horizon for rollout filter (default: 10)")
    parser.add_argument("--velocity-threshold", type=float, default=0.1,
                        help="Velocity threshold for rollout filter (default: 0.1)")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        print("Make sure you have trained a SAC model first")
        return
    
    # Validate filter configuration
    if args.filter in ["value", "rollout"] and args.safety_model is None:
        print(f"Error: --safety-model is required when using --filter {args.filter}")
        return
    
    replay_sac_model(
        model_path=args.model,
        env_id=args.env,
        num_episodes=args.episodes,
        deterministic=not args.stochastic,
        step_delay=args.delay,
        filter_mode=args.filter,
        safety_model_path=args.safety_model,
        filter_epsilon=args.epsilon,
        rollout_horizon=args.horizon,
        rollout_velocity_threshold=args.velocity_threshold
    )


if __name__ == "__main__":
    main()