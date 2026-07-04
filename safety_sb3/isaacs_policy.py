"""Custom SB3 policy for ISAACS (two-player reach-avoid).

`IsaacsPolicy` subclasses SB3's :class:`SACPolicy` so ISAACS stays plug-and-play
for existing SB3 code: ``model.predict`` / save / load behave normally and return
the **control** action.  Internally it holds two actors over disjoint sub-action
spaces and one twin critic over the FULL (concatenated) action:

* ``self.actor``      — control actor (max-player), over the first ``ctrl_action_dim``
  action dims.  This is the deployable safety controller; ``predict`` uses it.
* ``self.dstb_actor`` — disturbance actor (min-player), over the remaining dims.
* ``self.critic`` / ``self.critic_target`` — ``Q(s, [a_ctrl, a_dstb])`` over the
  full action space.

The env's ``action_space`` is the concatenation ``Box(ctrl_dim + dstb_dim)``; the
env splits it and applies ctrl as control + dstb as a bounded disturbance.
``ctrl_action_dim`` tells the policy where to split (pass via ``policy_kwargs``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.sac.policies import Actor, SACPolicy


class IsaacsPolicy(SACPolicy):
  def __init__(self, *args, ctrl_action_dim: int, **kwargs) -> None:
    # Must be set before super().__init__ → _build().
    self.ctrl_action_dim = int(ctrl_action_dim)
    super().__init__(*args, **kwargs)

  # --- helpers ---
  def _split_action_space(self) -> tuple[spaces.Box, spaces.Box]:
    full = self.action_space
    assert isinstance(full, spaces.Box) and full.shape is not None
    c = self.ctrl_action_dim
    assert 0 < c < full.shape[0], (
      f"ctrl_action_dim={c} must be in (0, {full.shape[0]}) for a "
      f"concatenated [ctrl, dstb] action space."
    )
    ctrl = spaces.Box(full.low[:c], full.high[:c], dtype=full.dtype)  # type: ignore[index]
    dstb = spaces.Box(full.low[c:], full.high[c:], dtype=full.dtype)  # type: ignore[index]
    return ctrl, dstb

  def _make_actor_for(self, action_space: spaces.Box) -> Actor:
    actor_kwargs = self._update_features_extractor(self.actor_kwargs, None)
    actor_kwargs["action_space"] = action_space
    return Actor(**actor_kwargs).to(self.device)

  # --- build two actors + full critic ---
  def _build(self, lr_schedule: Schedule) -> None:
    ctrl_space, dstb_space = self._split_action_space()
    self._ctrl_space, self._dstb_space = ctrl_space, dstb_space

    # Control actor == self.actor (used by predict / deployment).
    self.actor = self._make_actor_for(ctrl_space)
    self.actor.optimizer = self.optimizer_class(
      self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
    )
    # Disturbance actor (min-player).
    self.dstb_actor = self._make_actor_for(dstb_space)
    self.dstb_actor.optimizer = self.optimizer_class(
      self.dstb_actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
    )

    # Twin critic over the FULL concatenated action (own features extractor).
    self.critic = self.make_critic(features_extractor=None)
    self.critic.optimizer = self.optimizer_class(
      list(self.critic.parameters()), lr=lr_schedule(1), **self.optimizer_kwargs
    )
    self.critic_target = self.make_critic(features_extractor=None)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_target.set_training_mode(False)

  def _get_constructor_parameters(self) -> dict[str, Any]:
    data = super()._get_constructor_parameters()
    data.update(ctrl_action_dim=self.ctrl_action_dim)
    return data

  def set_training_mode(self, mode: bool) -> None:
    self.actor.set_training_mode(mode)
    self.dstb_actor.set_training_mode(mode)
    self.critic.set_training_mode(mode)
    self.training = mode

  @property
  def dstb_action_dim(self) -> int:
    assert self.action_space.shape is not None
    return int(self.action_space.shape[0]) - self.ctrl_action_dim

  def predict(self, observation, state=None, episode_start=None, deterministic=False):
    """Deployment: return the CONTROL action only (the safety controller, no
    adversary), in the control sub-space bounds.  Overrides SB3's predict, which
    would reshape the ctrl-dim output to the full concatenated action space."""
    self.set_training_mode(False)
    obs_tensor, vectorized = self.obs_to_tensor(observation)
    with __import__("torch").no_grad():
      actions = self.actor(obs_tensor, deterministic=deterministic)
    actions = actions.cpu().numpy().reshape((-1, self.ctrl_action_dim))
    low, high = self._ctrl_space.low, self._ctrl_space.high
    actions = low + 0.5 * (actions + 1.0) * (high - low)  # [-1,1] -> ctrl bounds
    actions = np.clip(actions, low, high)
    if not vectorized:
      actions = actions.squeeze(axis=0)
    return actions, state
