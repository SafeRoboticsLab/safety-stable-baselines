"""Asymmetric (privileged-critic) actor-critic policy for the PPO family.

Stock SB3 :class:`ActorCriticPolicy` assumes ONE observation feeds both the actor
and the value net. rsl_rl (and the unitree_rl_mjlab velocity task the factory
mirrors) instead trains an **asymmetric** critic: the actor reads the deployable
proprioceptive obs (blind), while the value net additionally sees privileged
state — base linear velocity above all, the exact quantity the tracking reward
scores. A critic that observes it directly gives low-variance advantages for
"faster => more return", the gradient a blind critic cannot form cleanly.

:class:`AsymmetricActorCriticPolicy` is that policy, kept minimal and surgical:

* the ACTOR half is byte-for-byte SB3's — ``pi_features_extractor`` +
  ``mlp_extractor.forward_actor`` + ``action_net`` + ``log_std`` — so a policy
  saved/loaded/deployed through it behaves exactly like an ``MlpPolicy`` on the
  actor obs (``predict`` / ``_predict`` / ``get_distribution`` are unchanged).
* the CRITIC half is a SEPARATE stack over ``critic_observation_space``: its own
  ``vf_features_extractor`` + ``critic_mlp_extractor`` + ``value_net``.

Three call surfaces are overridden to route the critic obs:
:meth:`forward` (rollout — takes an explicit ``critic_obs``),
:meth:`predict_values` (its argument IS the critic obs), and
:meth:`evaluate_actions` (the update — receives the critic obs packed onto the
actor obs by the tensor rollout buffer, and splits it back out).

The deployed filter never touches any of this: only the actor half runs on
hardware, and the critic net is discarded at eval. See the factory's A3 note.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Optional

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy, BaseModel
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule


class AsymmetricActorCriticPolicy(ActorCriticPolicy):
  """ActorCriticPolicy whose value net reads a DIFFERENT (privileged) obs.

  :param critic_observation_space: the observation space the value net consumes.
      Its dimension may differ from ``observation_space`` (the actor's). Required.
  """

  def __init__(self, observation_space: spaces.Space, action_space: spaces.Space,
               lr_schedule: Schedule, *args,
               critic_observation_space: Optional[spaces.Space] = None,
               **kwargs) -> None:
    if critic_observation_space is None:
      raise ValueError(
        "AsymmetricActorCriticPolicy requires critic_observation_space (the "
        "privileged obs the value net reads). For a symmetric critic use the "
        "stock MlpPolicy.")
    self.critic_observation_space = critic_observation_space
    self.actor_obs_dim = int(np.prod(observation_space.shape))
    self.critic_obs_dim = int(np.prod(critic_observation_space.shape))
    # The actor and critic never share a features extractor here — their input
    # dims differ. Force it off so SB3 builds two extractors (we repoint the vf
    # one at the critic space in _build).
    kwargs["share_features_extractor"] = False
    super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

  # --- build ----------------------------------------------------------------
  def _make_critic_features_extractor(self):
    return self.features_extractor_class(self.critic_observation_space,
                                         **self.features_extractor_kwargs)

  def _build_mlp_extractor(self) -> None:
    # Actor-side extractor on the actor feature dim (stock).
    self.mlp_extractor = MlpExtractor(
      self.features_dim, net_arch=self.net_arch,
      activation_fn=self.activation_fn, device=self.device)
    # Critic-side extractor on the (possibly larger) critic feature dim. Only its
    # value branch (forward_critic) is used; the policy branch is inert.
    self.critic_mlp_extractor = MlpExtractor(
      self.critic_features_dim, net_arch=self.net_arch,
      activation_fn=self.activation_fn, device=self.device)

  def _build(self, lr_schedule: Schedule) -> None:
    # super().__init__ built vf_features_extractor on the ACTOR space (it does not
    # know about the critic space); rebuild it on the critic space before the mlp
    # extractors are constructed, since _build_mlp_extractor reads its dim.
    self.vf_features_extractor = self._make_critic_features_extractor()
    self.critic_features_dim = self.vf_features_extractor.features_dim
    super()._build(lr_schedule)
    # super()._build wired value_net onto the actor mlp's vf latent; repoint it at
    # the critic mlp (same latent dim — same net_arch — so this is a fresh Linear,
    # not a reshape) and give the critic stack the same init the actor stack got.
    self.value_net = nn.Linear(self.critic_mlp_extractor.latent_dim_vf, 1).to(self.device)
    if self.ortho_init:
      self.critic_mlp_extractor.apply(partial(self.init_weights, gain=np.sqrt(2)))
      self.value_net.apply(partial(self.init_weights, gain=1))
    # Rebuild the optimizer so it owns the rebuilt vf extractor, the critic mlp and
    # the new value_net (all created/replaced after super()._build's optimizer).
    self.optimizer = self.optimizer_class(
      self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]

  # --- feature helpers (BaseModel.extract_features, NOT ActorCriticPolicy's,
  #     whose non-shared override returns a (pi, vf) tuple) ------------------
  def _actor_latent(self, obs: PyTorchObs) -> th.Tensor:
    features = BaseModel.extract_features(self, obs, self.pi_features_extractor)
    return self.mlp_extractor.forward_actor(features)

  def predict_values(self, obs: PyTorchObs) -> th.Tensor:
    """Value of the CRITIC observation ``obs`` (privileged; different dim)."""
    features = BaseModel.extract_features(self, obs, self.vf_features_extractor)
    latent_vf = self.critic_mlp_extractor.forward_critic(features)
    return self.value_net(latent_vf)

  # --- the three routed call surfaces ---------------------------------------
  def forward(self, obs: th.Tensor, deterministic: bool = False,
              critic_obs: Optional[th.Tensor] = None
              ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """(actions, values, log_prob). ``obs`` is the actor obs; ``critic_obs`` the
    privileged critic obs (falls back to ``obs`` if not supplied)."""
    latent_pi = self._actor_latent(obs)
    values = self.predict_values(obs if critic_obs is None else critic_obs)
    distribution = self._get_action_dist_from_latent(latent_pi)
    actions = distribution.get_actions(deterministic=deterministic)
    log_prob = distribution.log_prob(actions)
    actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
    return actions, values, log_prob

  def evaluate_actions(self, obs: PyTorchObs, actions: th.Tensor,
                       critic_obs: Optional[th.Tensor] = None
                       ) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
    """Update-time evaluation. The tensor rollout buffer packs the critic obs onto
    the actor obs (``cat([actor, critic], dim=-1)``) so stock ``PPO.train`` can
    hand it through unchanged; split it back out here. An explicit ``critic_obs=``
    is honoured too (then ``obs`` is the actor obs alone)."""
    if critic_obs is None:
      total = self.actor_obs_dim + self.critic_obs_dim
      if obs.shape[-1] == total:
        obs, critic_obs = obs[..., :self.actor_obs_dim], obs[..., self.actor_obs_dim:]
      else:
        raise ValueError(
          f"asymmetric evaluate_actions expected packed obs of dim {total} "
          f"(actor {self.actor_obs_dim} + critic {self.critic_obs_dim}) or an "
          f"explicit critic_obs=; got last dim {obs.shape[-1]}.")
    latent_pi = self._actor_latent(obs)
    distribution = self._get_action_dist_from_latent(latent_pi)
    log_prob = distribution.log_prob(actions)
    entropy = distribution.entropy()
    values = self.predict_values(critic_obs)
    return values, log_prob, entropy

  def _get_constructor_parameters(self) -> dict[str, Any]:
    data = super()._get_constructor_parameters()
    data.update(critic_observation_space=self.critic_observation_space)
    return data
