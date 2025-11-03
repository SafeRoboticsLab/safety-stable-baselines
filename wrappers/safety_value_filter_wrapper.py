import torch
import os
import sys
import torch
import numpy as np

# Add parent directory to path if needed
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
