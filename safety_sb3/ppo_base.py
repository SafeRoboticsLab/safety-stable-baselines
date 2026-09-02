"""PPO, written without a fixed Bellman backup — the shared half of the family.

:class:`AbstractPPO` is everything the PPO-family learners share regardless of
how many players are in the game: the env guards, the rsl_rl-parity training
recipe, the KL-adaptive learning rate, gamma annealing, and the choice of
rollout buffer from ``_MODE``. It deliberately owns **no rollout loop** — that
is the axis its subclasses split on:

    :class:`~safety_sb3.ppo_1p.AbstractPPO1P` — one actor, one buffer.
    :class:`~safety_sb3.ppo_2p.AbstractPPO2P` — two actors, two value nets,
        two buffers, and a phase machine.

In the PPO family the backup is carried by the **rollout buffer**, not by
``train()``: ``compute_returns_and_advantage`` substitutes the mode's operator
for the one-step TD target inside GAE. So a learner selects its problem simply
by declaring ``_MODE``, and
:func:`safety_sb3.buffers_rollout.rollout_buffer_classes` hands it the matching
buffers.

The rsl_rl-parity pieces (validated: they are what lets this learn hard robot
tasks where stock-SB3-PPO-on-``g`` stalls):

* **No timeout bootstrapping** (``bootstrap_on_timeout=False``, the default).
  Stock PPO adds ``gamma * V(terminal)`` to the reward at truncations
  (``on_policy_algorithm.collect_rollouts``). In the two safety modes the reward
  IS the physical safety margin ``g(s)`` — an absolute quantity — so adding a
  value estimate destroys its semantics and silently corrupts the backup the
  moment episodes start reaching the time limit. rsl_rl's SafetyPPO overrides
  ``process_env_step`` for exactly this reason; we gate it in the collect loops.

* **KL-adaptive learning rate** (``adaptive_lr=True`` with ``desired_kl``).
  rsl_rl raises the LR when the update KL sits well below target and lowers it
  when above. SB3's ``target_kl`` only *early-stops*; it never adjusts the LR, so
  in the low-KL / weak-gradient regime typical of the sparse safety signal the
  policy barely moves. Enabling this recovers rsl_rl's behavior.

Observation normalization is orthogonal and handled the SB3 way — wrap the env
in ``VecNormalize`` — which mirrors rsl_rl's built-in running obs normalizer.

All knobs are constructor arguments, so downstream projects configure the
algorithm without subclassing.
"""

from __future__ import annotations

from stable_baselines3.common.utils import update_learning_rate
from stable_baselines3.ppo.ppo import PPO

from . import backups
from .buffers_rollout import (check_rollout_buffer_mode,
                              rollout_buffer_classes)
from .env_checks import guard_and_normalize_env
from .gamma_anneal import GammaAnnealMixin


class AbstractPPO(GammaAnnealMixin, PPO):
    """PPO whose Bellman backup is chosen by ``_MODE`` (see the module docstring).

    :param bootstrap_on_timeout: if False (default) skip PPO's timeout value
        bootstrapping — correct whenever the reward is a margin ``g(s)``.
        Legal only in ``CUMULATIVE`` mode, where the reward is an ordinary dense
        return and an episode cut at the time limit has NOT ended: leaving the
        bootstrap off there teaches the policy that the horizon itself is a
        terminal event. Supported on both the numpy and the tensor path.
    :param normalize_obs: wrap the env in VecNormalize(norm_obs=True,
        norm_reward=False) — obs normalization is needed to match rsl_rl on hard
        robot tasks; reward normalization is refused (it corrupts ``g``).
    :param adaptive_lr: enable rsl_rl-style KL-adaptive learning rate.
    :param desired_kl: target KL for the adaptive LR controller.
    :param lr_bounds: (min, max) bounds for the adaptive LR.
    :param adaptive_lr_factor: multiplicative step for the adaptive LR.
    :param gamma_anneal: discount-factor annealing (ON by default). ``True``
        anneals gamma 0.99 -> 0.9999 over the first 50% of training then holds
        (the reach-avoid boundary only sharpens as gamma -> 1; see
        ``gamma_anneal.py``). ``False`` keeps gamma constant; a callable
        ``frac -> gamma`` supplies a custom schedule.

    GPU-resident path: pass a :class:`~safety_sb3.tensor_env.TensorVecEnv` and
    everything (rollout, buffer, backup, minibatching) stays on device — no numpy
    bounce. Detected automatically; env ``metrics()`` (curriculum levels etc.)
    are forwarded to the logger every rollout.
    """

    #: the Bellman operator this learner converges to; concretes set it
    _MODE = backups.AVOID

    def __init__(
        self,
        *args,
        learning_rate=3e-4,
        rollout_buffer_class=None,
        rollout_buffer_kwargs=None,
        bootstrap_on_timeout: bool = False,
        normalize_obs: bool = False,
        adaptive_lr: bool = False,
        desired_kl: float | None = 0.01,
        lr_bounds: tuple[float, float] = (1e-5, 1e-2),
        adaptive_lr_factor: float = 1.5,
        gamma_anneal=True,
        **kwargs,
    ):
        backups.check_mode(self._MODE)
        # Guard the reward-normalization footgun + optionally add obs norm.
        # env is the 2nd positional PPO arg (policy, env, ...) or a kwarg.
        args = list(args)
        if "env" in kwargs:
            kwargs["env"] = guard_and_normalize_env(kwargs["env"], normalize_obs)
        elif len(args) >= 2:
            args[1] = guard_and_normalize_env(args[1], normalize_obs)
        args = tuple(args)

        # GPU-resident path? (detected from the env; see tensor_env.py)
        _env = kwargs.get("env", args[1] if len(args) >= 2 else None)
        self._tensor_path = bool(getattr(_env, "is_tensor_env", False))
        if bootstrap_on_timeout and self._MODE != backups.CUMULATIVE:
            raise ValueError(
                f"bootstrap_on_timeout=True is invalid for mode {self._MODE!r}: the reward is the "
                "safety margin g(s), and bootstrapping it would add a value to a physical quantity."
            )

        # Default buffer: the pair implementing THIS learner's mode -- numpy or
        # torch depending on the path.
        if rollout_buffer_class is None:
            numpy_cls, tensor_cls = rollout_buffer_classes(self._MODE)
            rollout_buffer_class = tensor_cls if self._tensor_path else numpy_cls

        self.bootstrap_on_timeout = bool(bootstrap_on_timeout)
        self.adaptive_lr = bool(adaptive_lr)
        self.desired_kl = desired_kl
        self.lr_min, self.lr_max = float(lr_bounds[0]), float(lr_bounds[1])
        self.adaptive_lr_factor = float(adaptive_lr_factor)
        if self.adaptive_lr:
            if callable(learning_rate):
                raise ValueError(
                    "adaptive_lr=True requires a float initial learning_rate, "
                    f"got a schedule: {learning_rate!r}"
                )
            # Mutable current LR, adjusted each update from the measured KL.
            self._adaptive_lr = float(learning_rate)

        super().__init__(
            *args,
            learning_rate=learning_rate,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            **kwargs,
        )
        # Resolve the gamma-anneal schedule now that super() has set self.gamma.
        self._setup_gamma_anneal(gamma_anneal)

    def _setup_model(self) -> None:
        # Builds policy, optimizer, and rollout buffer.
        super()._setup_model()
        self._check_rollout_buffer(self.rollout_buffer)

    def _check_rollout_buffer(self, buffer) -> None:
        check_rollout_buffer_mode(self, buffer)

    def collect_rollouts(self, *args, **kwargs):
        """Refuse to fall through to SB3's ``OnPolicyAlgorithm.collect_rollouts``.

        This class has no rollout loop by design — that is the Players axis.
        Without this guard, instantiating it directly would inherit the stock
        loop, which bootstraps values on timeout (corrupting a margin reward) and
        never offers the buffer its per-step extras (so a reach-avoid buffer
        would silently back up ``l = 0`` everywhere).
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no rollout loop. Use a concrete learner "
            "(SafetyPPO1P, ReachAvoidPPO2P, ...), or AbstractPPO1P / "
            "AbstractPPO2P.")

    # ---------------------------------------------------------- adaptive LR
    def _update_learning_rate(self, optimizers) -> None:
        """Set the optimizer LR.  With ``adaptive_lr`` use the KL-controlled
        value instead of SB3's progress schedule (called at each ``train()``)."""
        if not self.adaptive_lr:
            super()._update_learning_rate(optimizers)
            return
        self.logger.record("train/learning_rate", self._adaptive_lr)
        update_learning_rate(optimizers, self._adaptive_lr)

    def train(self) -> None:
        super().train()
        # Adjust the LR for the NEXT update from the KL just measured, mirroring
        # rsl_rl: KL >> target -> shrink LR; KL << target -> grow LR.
        if self.adaptive_lr and self.desired_kl is not None:
            kl = self.logger.name_to_value.get("train/approx_kl")
            if kl is not None and kl > 0.0:
                if kl > self.desired_kl * 2.0:
                    self._adaptive_lr = max(
                        self.lr_min, self._adaptive_lr / self.adaptive_lr_factor
                    )
                elif kl < self.desired_kl / 2.0:
                    self._adaptive_lr = min(
                        self.lr_max, self._adaptive_lr * self.adaptive_lr_factor
                    )
            self.logger.record("train/adaptive_lr", self._adaptive_lr)
