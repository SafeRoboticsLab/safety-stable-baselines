"""Environment guards shared by every learner in the library.

Lives in its own module because both algorithm families need it and neither
should have to import the other to get it (before v0.4.0 the SAC base imported
this function out of the PPO module).
"""
from __future__ import annotations

from stable_baselines3.common.vec_env import VecNormalize

from .tensor_env import TensorVecNormalize


def guard_and_normalize_env(env, normalize_obs: bool):
    """Reject reward-normalizing wrappers and optionally add obs normalization.

    In the two safety modes the reward IS the physical safety margin ``g(s)``,
    an absolute quantity. Normalizing it (``VecNormalize``'s ``norm_reward=True``
    default) rescales ``g`` by a running std and destroys the safety Bellman
    backup — the same class of corruption as timeout value-bootstrapping. So we
    hard-error on it. Observation normalization, by contrast, is REQUIRED for
    hard robot tasks (rsl_rl normalizes obs on both actor and critic);
    ``normalize_obs=True`` wraps the env in ``VecNormalize(norm_obs=True,
    norm_reward=False)`` for you (mirroring rsl_rl's built-in running normalizer).
    """
    if isinstance(env, VecNormalize) and env.norm_reward:
        raise ValueError(
            "Safety*/ReachAvoid* algorithms require an UN-normalized reward: the "
            "reward is the physical safety margin g(s), and "
            "VecNormalize(norm_reward=True) rescales it, corrupting the safety "
            "Bellman backup. Re-wrap with VecNormalize(env, norm_obs=True, "
            "norm_reward=False), or pass normalize_obs=True to the algorithm and "
            "drop your VecNormalize."
        )
    if getattr(env, "is_tensor_env", False):
        # GPU-resident path: reward normalization does not exist by design;
        # obs normalization uses the on-device running normalizer.
        if normalize_obs and not isinstance(env, TensorVecNormalize):
            env = TensorVecNormalize(env)
        return env
    if normalize_obs and not isinstance(env, VecNormalize):
        env = VecNormalize(env, norm_obs=True, norm_reward=False)
    return env
