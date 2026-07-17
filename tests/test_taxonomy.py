"""The 2x2: {avoid, reach-avoid} x {single-player, two-player}.

Pins which problem each exported class actually solves. v0.1.0 shipped PPO and
SAC computing *different fixed points under the same name*, and named the
two-player reach-avoid game after a paper (ISAACS) that is an avoid game — both
because nothing asserted the taxonomy. This file asserts it.

    Isaacs*   = ISAACS (Hsu et al. 2022, eq. 7)   -> two-player AVOID
    Gameplay* = Gameplay Filters (eq. 6a)         -> two-player REACH-AVOID
"""
import gymnasium as gym
import numpy as np

from safety_sb3 import (GameplayPPO, GameplaySAC, IsaacsPPO, IsaacsSAC,
                        ReachAvoidPPO, ReachAvoidSAC, SafetyPPO, SafetySAC,
                        backups)
from safety_sb3.safety_buffers import (ReachAvoidRolloutBuffer,
                                       SafetyRolloutBuffer)
from safety_sb3.tensor_buffers import (TensorReachAvoidRolloutBuffer,
                                       TensorSafetyRolloutBuffer)

AVOID, RA = backups.AVOID, backups.REACH_AVOID


def test_sac_family_modes():
  """Each SAC class declares the problem it actually backs up."""
  assert ReachAvoidSAC._MODE == RA
  assert GameplaySAC._MODE == RA      # two-player reach-avoid (Gameplay Filters)
  assert IsaacsSAC._MODE == AVOID     # two-player avoid (ISAACS proper)
  # the two-player classes share the adversary machinery
  assert issubclass(IsaacsSAC, GameplaySAC)


def test_ppo_family_modes_and_buffers():
  """PPO picks its backup by buffer class; the buffer must match the mode."""
  assert ReachAvoidPPO._MODE == RA
  assert GameplayPPO._MODE == RA
  assert IsaacsPPO._MODE == AVOID

  # reach-avoid classes carry l; the avoid class must NOT
  for cls in (ReachAvoidPPO, GameplayPPO):
    assert cls.numpy_rollout_buffer_class is ReachAvoidRolloutBuffer, cls
    assert cls.tensor_rollout_buffer_class is TensorReachAvoidRolloutBuffer, cls
  assert IsaacsPPO.numpy_rollout_buffer_class is SafetyRolloutBuffer
  assert IsaacsPPO.tensor_rollout_buffer_class is TensorSafetyRolloutBuffer


def test_avoid_classes_ignore_l_entirely():
  """The avoid column has no target set — nothing should thread l.

  This is the guard against the l_neg pattern coming back: an avoid task must
  not be able to smuggle a target margin into an avoid learner.
  """
  assert not IsaacsPPO(  # constructed below via _is_reach_avoid on the class
    "MlpPolicy", _TwoPlayerEnv(), ctrl_action_dim=1, n_steps=8, batch_size=8,
    device="cpu")._is_reach_avoid
  assert ReachAvoidPPO("MlpPolicy", _SingleEnv(), n_steps=8, batch_size=8,
                       device="cpu")._is_reach_avoid


class _SingleEnv(gym.Env):
  observation_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
  action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

  def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    return np.zeros(2, np.float32), {}

  def step(self, action):
    return np.zeros(2, np.float32), 1.0, False, False, {"l_x": -1.0}


class _TwoPlayerEnv(_SingleEnv):
  # one concatenated Box(ctrl + dstb), split by ctrl_action_dim
  action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)


def test_terminal_type_is_first_class_on_ppo():
  """terminal_type reaches the buffer as a constructor kwarg (not just via
  rollout_buffer_kwargs), mirroring ReachAvoidSAC."""
  for tt in ("all", "g"):
    m = ReachAvoidPPO("MlpPolicy", _SingleEnv(), terminal_type=tt,
                      n_steps=32, batch_size=32, device="cpu")
    assert m.rollout_buffer.terminal_type == tt, (tt, m.rollout_buffer.terminal_type)
  # invalid value rejected at construction
  try:
    ReachAvoidPPO("MlpPolicy", _SingleEnv(), terminal_type="bogus",
                  n_steps=32, batch_size=32, device="cpu")
    raise AssertionError("expected ValueError")
  except ValueError:
    pass
  # avoid learner does not carry it into its Safety buffer
  m = IsaacsPPO("MlpPolicy", _TwoPlayerEnv(), ctrl_action_dim=1,
                n_steps=32, batch_size=32, device="cpu")
  assert not hasattr(m.rollout_buffer, "terminal_type")
