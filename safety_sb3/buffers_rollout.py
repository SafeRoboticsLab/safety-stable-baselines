"""On-policy rollout buffers — one per Mode, player-agnostic.

Rollout buffers carry the **M** of MAP and nothing else. A buffer does not know
or care whether one player or two produced the data in it, so there is no
``1P``/``2P`` in these names: :class:`~safety_sb3.ppo_2p.AbstractPPO2P` builds
*two* instances of the very same class, one per player.

Each buffer computes the backup for its ``_MODE`` through
:func:`safety_sb3.backups.target`, so the operator is a parameter rather than a
class body; ``mode=`` overrides the class default per instance.

Each buffer also **captures its own extras**. The safety margin ``g(s)`` always
rides on the reward channel, but the reach-avoid operator additionally needs the
target margin ``l(s)``, which SB3 delivers only in ``infos``. Rather than teach
every algorithm to thread ``l``, the collect loop hands each step's ``infos`` to
the buffer via :meth:`record_extras` and the buffer keeps whatever its own
operator needs — nothing, for the two modes that have no ``l``. That is why the
reach-avoid *learners* need almost no reach-avoid code.
"""
import numpy as np
import torch as th
from stable_baselines3.common.buffers import RolloutBuffer

from . import backups


class SafetyRolloutBuffer(RolloutBuffer):
    """Rollout buffer for the **avoid** backup (Fisac et al. 2019).

    ``g(s)`` rides on the reward channel; the operator is
    ``V(s) = (1-gamma)*g + gamma*min(g, V(s'))`` with terminal target ``g``.
    See :mod:`safety_sb3.backups`, which defines it.
    """

    _MODE = backups.AVOID

    def __init__(self, *args, mode: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._MODE = backups.check_mode(self._MODE if mode is None else mode)

    def record_extras(self, infos: list) -> None:
        """Capture per-step extras from ``infos`` into the slot ``add()`` will fill.

        Called by the on-policy collect loop immediately before ``add()``.
        No-op here: the avoid operator needs only ``g``, which arrives on the
        reward channel. :class:`ReachAvoidRolloutBuffer` overrides it to keep
        ``l(s)``.
        """

    def _target(self, step: int, v_next: np.ndarray,
                not_done: np.ndarray) -> np.ndarray:
        l_x = getattr(self, "l_x", None)
        return backups.target(
            self._MODE, self.rewards[step], v_next, not_done, self.gamma,
            l=None if l_x is None else l_x[step],
            terminal_type=getattr(self, "terminal_type", "all"))

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """SB3's GAE loop with this buffer's Bellman backup substituted for the
        one-step TD target inside ``delta`` (see :meth:`_target`)."""
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()  # type: ignore[assignment]

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                v_next = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                v_next = self.values[step + 1]

            target = self._target(step, v_next, next_non_terminal)
            delta = target - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        self.returns = self.advantages + self.values


class ReachAvoidRolloutBuffer(SafetyRolloutBuffer):
    """Rollout buffer for the **reach-avoid** backup (Hsu et al. RSS'21 eq. 15 /
    Gameplay Filters eq. 6a)::

        V(s) = (1-gamma)*min(l, g) + gamma*min(g, max(l, V(s')))

    Stores the per-step target margin ``l(s)`` beside the safety margin ``g(s)``
    (which rides on ``rewards``). ``l`` is read out of ``info["l_x"]`` by
    :meth:`record_extras` — the buffer captures it itself, which is what keeps
    the reach-avoid learners free of ``l``-plumbing.

    See :mod:`safety_sb3.backups` for why the anchor is ``min(l, g)`` and not
    ``g``, and why avoid is not expressible by degenerating ``l``.

    :param terminal_type: ``"all"`` (default) -> terminal target ``min(l, g)``;
        ``"g"`` -> ``g``. See :func:`safety_sb3.backups.reach_avoid_target`.
    """

    _MODE = backups.REACH_AVOID

    def __init__(self, *args, terminal_type: str = "all", **kwargs):
        super().__init__(*args, **kwargs)
        self.terminal_type = backups.check_terminal_type(terminal_type)

    def reset(self) -> None:
        super().reset()
        self.l_x = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def record_extras(self, infos: list) -> None:
        """Keep this step's target margin ``l(s)``, read from ``info["l_x"]``."""
        self.l_x[self.pos] = np.array(
            [float(info.get("l_x", 0.0)) for info in infos], dtype=np.float32)


class CumulativeRolloutBuffer(SafetyRolloutBuffer):
    """Rollout buffer for the **cumulative** (ordinary RL) backup
    ``V(s) = r + gamma*V(s')`` — which makes the loop above exactly stock SB3 GAE.

    Not a safety buffer: the reward channel carries a *reward*, not a margin, and
    the resulting value has no zero-level-set meaning. It exists so a
    reward-maximizing baseline runs through every line of the same code as the
    safety learners (see :mod:`safety_sb3.backups`).
    """

    _MODE = backups.CUMULATIVE


def rollout_buffer_classes(mode: str) -> tuple[type, type]:
    """``(numpy_cls, tensor_cls)`` implementing ``mode``'s backup.

    The on-policy learners look their buffers up here from their own ``_MODE``
    instead of naming them as class attributes, so a buffer cannot drift out of
    step with the mode its owner declares. (It used to: the two-player avoid
    learner hardcoded the reach-avoid buffer for its min player and so trained
    that player on the wrong game.) It also makes every concrete learner in the
    library a genuine one-liner — ``_MODE = ...`` and nothing else.
    """
    from .buffers_tensor import (TensorCumulativeRolloutBuffer,
                                 TensorReachAvoidRolloutBuffer,
                                 TensorSafetyRolloutBuffer)
    return {
        backups.AVOID: (SafetyRolloutBuffer, TensorSafetyRolloutBuffer),
        backups.REACH_AVOID: (ReachAvoidRolloutBuffer,
                              TensorReachAvoidRolloutBuffer),
        backups.CUMULATIVE: (CumulativeRolloutBuffer,
                             TensorCumulativeRolloutBuffer),
    }[backups.check_mode(mode)]


def check_rollout_buffer_mode(owner, buffer) -> None:
    """The buffer must compute the operator its owner declares.

    In the on-policy families the buffer *is* the backup, so a mismatch here is
    not a type error — it is silently optimizing the wrong fixed point, which is
    exactly the class of bug this library exists to make impossible.
    """
    mode = getattr(buffer, "_MODE", None)
    if mode != owner._MODE:
        raise TypeError(
            f"{type(owner).__name__} declares _MODE={owner._MODE!r} but its "
            f"rollout buffer {type(buffer).__name__} computes {mode!r}. In the "
            "on-policy families the buffer carries the backup, so these must "
            "agree. Drop rollout_buffer_class= to get the right one automatically."
        )
