import os
import sys
import safety_gymnasium
import torch
import numpy as np
import mujoco

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
