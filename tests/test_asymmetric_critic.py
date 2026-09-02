"""Asymmetric (privileged) critic — the value net reads a DIFFERENT obs than the actor.

Two things are pinned here:

1. The policy is wired correctly: the ACTOR net's input width is the actor obs dim, the CRITIC
   (value) net's input width is the (different) critic obs dim, and each half only responds to its
   own obs (change the critic obs => value moves, actions don't; change the actor obs => actions
   move, value doesn't).
2. The whole learner path runs on a tiny tensor env that exposes a critic group: CumulativePPO1P
   auto-detects the group, builds the asymmetric policy + a buffer that carries the critic obs, and
   trains without error — and a symmetric env (no critic group) stays on the stock MlpPolicy.
"""
import gymnasium as gym
import numpy as np
import torch as th

from safety_sb3 import AsymmetricActorCriticPolicy, CumulativePPO1P
from safety_sb3.tensor_env import TensorVecEnv

ACT_DIM, CRIT_DIM, N_ACT = 8, 12, 3


def _first_linear_in_features(seq) -> int:
  for m in seq:
    if isinstance(m, th.nn.Linear):
      return m.in_features
  raise AssertionError("no Linear in the net")


def _make_policy():
  obs_space = gym.spaces.Box(-np.inf, np.inf, (ACT_DIM,), np.float32)
  crit_space = gym.spaces.Box(-np.inf, np.inf, (CRIT_DIM,), np.float32)
  act_space = gym.spaces.Box(-1.0, 1.0, (N_ACT,), np.float32)
  return AsymmetricActorCriticPolicy(
    obs_space, act_space, lr_schedule=lambda _: 3e-4,
    critic_observation_space=crit_space,
    net_arch=dict(pi=[32, 32], vf=[16, 16]))


def test_actor_and_critic_input_widths():
  """The actor net takes actor-dim input; the value (critic) net takes critic-dim input."""
  pol = _make_policy()
  assert _first_linear_in_features(pol.mlp_extractor.policy_net) == ACT_DIM
  assert _first_linear_in_features(pol.critic_mlp_extractor.value_net) == CRIT_DIM
  # The stock actor value branch and the critic feature extractor are on the right widths too.
  assert pol.actor_obs_dim == ACT_DIM and pol.critic_obs_dim == CRIT_DIM
  assert pol.vf_features_extractor.features_dim == CRIT_DIM
  assert pol.pi_features_extractor.features_dim == ACT_DIM


def test_forward_and_predict_values_route_the_right_obs():
  pol = _make_policy()
  pol.set_training_mode(False)
  actor_obs = th.randn(5, ACT_DIM)
  critic_obs = th.randn(5, CRIT_DIM)

  # predict_values takes the CRITIC obs (its width), not the actor's.
  v = pol.predict_values(critic_obs)
  assert v.shape == (5, 1)
  with_wrong_width_raised = False
  try:
    pol.predict_values(actor_obs)  # 8 != 12 -> the critic Linear must reject it
  except (RuntimeError, ValueError):
    with_wrong_width_raised = True
  assert with_wrong_width_raised, "value net accepted actor-width obs; it is not on the critic obs"

  # forward: value depends on the critic obs only; the sampled action on the actor obs only.
  th.manual_seed(0)
  a1, val1, _ = pol(actor_obs, critic_obs=critic_obs)
  a2, val2, _ = pol(actor_obs, critic_obs=critic_obs + 3.0)   # change critic obs only
  assert not th.allclose(val1, val2), "value ignored the critic obs"
  # deterministic actions must be identical when only the critic obs changed
  da1 = pol._predict(actor_obs, deterministic=True)
  da2 = pol._predict(actor_obs, deterministic=True)
  assert th.allclose(da1, da2)


def test_evaluate_actions_splits_packed_obs():
  """The tensor buffer packs [actor|critic]; evaluate_actions must split at actor_obs_dim."""
  pol = _make_policy()
  actor_obs = th.randn(4, ACT_DIM)
  critic_obs = th.randn(4, CRIT_DIM)
  actions = th.randn(4, N_ACT)
  packed = th.cat([actor_obs, critic_obs], dim=-1)
  v_packed, lp_packed, ent_packed = pol.evaluate_actions(packed, actions)
  v_split, lp_split, _ = pol.evaluate_actions(actor_obs, actions, critic_obs=critic_obs)
  assert v_packed.shape == (4, 1)
  assert th.allclose(v_packed, v_split, atol=1e-6)
  assert th.allclose(lp_packed, lp_split, atol=1e-6)


# --- end-to-end on a tiny tensor env --------------------------------------------------------------

class _TinyAsymEnv(TensorVecEnv):
  """Minimal tensor env with a privileged critic group (critic obs = actor obs + 4 extra dims)."""

  def __init__(self, n=16, device="cpu", asymmetric=True):
    obs_space = gym.spaces.Box(-np.inf, np.inf, (ACT_DIM,), np.float32)
    act_space = gym.spaces.Box(-1.0, 1.0, (N_ACT,), np.float32)
    super().__init__(n, obs_space, act_space, device)
    self.n = n
    self._asym = asymmetric
    if asymmetric:
      self.critic_observation_space = gym.spaces.Box(-np.inf, np.inf, (CRIT_DIM,), np.float32)
    self._t = th.zeros(n, device=device)
    self._s = th.zeros(n, ACT_DIM, device=device)
    self._last_c = None

  def _obs(self):
    return self._s.clone()

  def _critic(self):
    # actor obs + 4 privileged extras (a stand-in for base_lin_vel etc.)
    extra = self._s[:, :4] * 2.0
    return th.cat([self._s, extra], dim=-1)

  def reset(self):
    self._s = th.randn(self.n, ACT_DIM, device=self.device)
    self._t.zero_()
    if self._asym:
      self._last_c = self._critic()
    return self._obs()

  def critic_obs(self):
    return self._last_c if self._asym else None

  def step_tensor(self, actions):
    self._s = 0.98 * self._s + 0.02 * th.randn(self.n, ACT_DIM, device=self.device)
    self._s[:, :N_ACT] += 0.01 * actions
    self._t += 1
    rew = -(self._s ** 2).mean(dim=1)
    timeouts = self._t >= 20
    dones = timeouts.clone()
    if bool(dones.any()):
      m = dones.bool()
      self._s[m] = th.randn(int(m.sum()), ACT_DIM, device=self.device)
      self._t[m] = 0
    if self._asym:
      self._last_c = self._critic()
    l_x = th.zeros(self.n, device=self.device)
    return self._obs(), rew, dones, timeouts, l_x


def test_learner_autodetects_and_trains_asymmetric():
  env = _TinyAsymEnv(n=16, device="cpu", asymmetric=True)
  model = CumulativePPO1P(
    "MlpPolicy", env, n_steps=20, batch_size=80, n_epochs=2, seed=0, verbose=0,
    device="cpu",  # match the CPU env (default "auto" would put the policy on cuda)
    gamma_anneal=False, bootstrap_on_timeout=True, normalize_obs=True,
    policy_kwargs=dict(net_arch=dict(pi=[32, 32], vf=[16, 16])))
  assert model._asymmetric, "learner did not auto-detect the critic group"
  assert isinstance(model.policy, AsymmetricActorCriticPolicy)
  assert model.rollout_buffer.critic_obs_dim == CRIT_DIM
  # value net on critic width, actor net on actor width
  assert _first_linear_in_features(model.policy.critic_mlp_extractor.value_net) == CRIT_DIM
  assert _first_linear_in_features(model.policy.mlp_extractor.policy_net) == ACT_DIM
  model.learn(total_timesteps=640)  # a few rollouts + updates
  # the buffer actually carried a critic obs (not the zero default)
  assert float(model.rollout_buffer.critic_observations.abs().sum()) > 0


def test_symmetric_env_stays_on_mlp_policy():
  env = _TinyAsymEnv(n=16, device="cpu", asymmetric=False)
  model = CumulativePPO1P(
    "MlpPolicy", env, n_steps=20, batch_size=80, n_epochs=2, seed=0, verbose=0,
    device="cpu",  # match the CPU env
    gamma_anneal=False, bootstrap_on_timeout=True, normalize_obs=True)
  assert not model._asymmetric
  assert not isinstance(model.policy, AsymmetricActorCriticPolicy)
  assert model.rollout_buffer.critic_obs_dim == 0
  model.learn(total_timesteps=640)
