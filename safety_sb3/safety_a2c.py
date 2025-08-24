from stable_baselines3.a2c.a2c import A2C
from .safety_buffers import SafetyRolloutBuffer


class SafetyA2C(A2C):
    """Safety A2C.
    """

    def __init__(self, *args, rollout_buffer_class=None, rollout_buffer_kwargs=None, **kwargs):
        # Default to SafetyRolloutBuffer unless the caller overrides it
        if rollout_buffer_class is None:
            rollout_buffer_class = SafetyRolloutBuffer
        super().__init__(
            *args, rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs, **kwargs
        )

    def _setup_model(self) -> None:
        # Builds policy, optimizer, and rollout buffer
        super()._setup_model()
        # Now the buffer exists → safe to assert
        assert isinstance(self.rollout_buffer, SafetyRolloutBuffer), (
            "SafetyA2C requires SafetyRolloutBuffer. "
            "Pass `rollout_buffer_class=SafetyRolloutBuffer` (and rollout_buffer_kwargs if needed)."
        )
