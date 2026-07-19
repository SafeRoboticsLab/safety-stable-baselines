from stable_baselines3.a2c.a2c import A2C
from .gamma_anneal import GammaAnnealMixin
from .safety_buffers import SafetyRolloutBuffer


class SafetyA2C(GammaAnnealMixin, A2C):
    """Safety A2C.

    ``gamma_anneal`` (ON by default) anneals the discount 0.99 -> 0.9999 over the
    first 50% of training (read off ``rollout_buffer.gamma`` in GAE); applied via
    ``_update_current_progress_remaining`` each iteration. See ``gamma_anneal.py``.
    """

    def __init__(self, *args, rollout_buffer_class=None, rollout_buffer_kwargs=None,
                 gamma_anneal=True, **kwargs):
        # Default to SafetyRolloutBuffer unless the caller overrides it
        if rollout_buffer_class is None:
            rollout_buffer_class = SafetyRolloutBuffer
        super().__init__(
            *args, rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs, **kwargs
        )
        self._setup_gamma_anneal(gamma_anneal)

    def _setup_model(self) -> None:
        # Builds policy, optimizer, and rollout buffer
        super()._setup_model()

        # Now the buffer exists → safe to assert
        assert isinstance(self.rollout_buffer, SafetyRolloutBuffer), (
            "SafetyA2C requires SafetyRolloutBuffer. "
            "Pass `rollout_buffer_class=SafetyRolloutBuffer` (and rollout_buffer_kwargs if needed)."
        )
