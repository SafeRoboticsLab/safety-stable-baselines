from gymnasium import logger

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

    def soft_reset(self, **kwargs):
        """A soft reset that does not reset the environment state."""
        if hasattr(self.env, 'soft_reset'):
            self._elapsed_steps = 0
            return self.env.soft_reset(**kwargs)
        else:
            logger.warn(
                'The environment does not support soft reset. '
                'Falling back to hard reset.',
            )
            return self.reset(**kwargs)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_observation = obs
        return obs, reward, terminated, truncated, info
        
    def get_last_observation(self):
        return self.last_observation
    
    def render(self, **kwargs):
        return self.env.render(**kwargs)
        
    def __getattr__(self, name):
        return getattr(self.env, name)

