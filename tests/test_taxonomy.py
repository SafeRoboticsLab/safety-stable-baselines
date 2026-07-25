"""MAP: Mode x Algorithm x Players. This file asserts the taxonomy holds.

v0.1.0 shipped PPO and SAC computing *different fixed points under the same
name*, and named the two-player reach-avoid game after a paper (ISAACS) that is
an avoid game — both because nothing asserted the taxonomy. So it is asserted
here, now against the v0.4.0 names, which encode the answer:

    <Mode><Algorithm><Players>   e.g. ReachAvoidSAC2P

The interesting properties are structural, not numeric:

  * every concrete learner's ``_MODE`` matches the Mode in its own name;
  * the buffers are player-agnostic (Mode only — no 1P/2P), because a buffer
    cannot tell how many players filled it;
  * reach-avoid is a MIXIN composed onto both player counts, and the avoid
    learners never receive it — the ``_is_reach_avoid`` predicate that used to
    switch it back off is gone and cannot come back;
  * DQN has no reach-avoid variant, loudly.
"""
import gymnasium as gym
import numpy as np
import pytest

import safety_sb3
from safety_sb3 import (CumulativeA2C1P, CumulativeDQN1P, CumulativePPO1P,
                        CumulativeSAC1P, ReachAvoidA2C1P, ReachAvoidPPO1P,
                        ReachAvoidPPO2P, ReachAvoidSAC1P, ReachAvoidSAC2P,
                        SafetyA2C1P, SafetyDQN1P, SafetyPPO1P, SafetyPPO2P,
                        SafetySAC1P, SafetySAC2P, backups)
from safety_sb3.buffers_rollout import (CumulativeRolloutBuffer,
                                        ReachAvoidRolloutBuffer,
                                        SafetyRolloutBuffer,
                                        rollout_buffer_classes)
from safety_sb3.buffers_tensor import (TensorCumulativeRolloutBuffer,
                                       TensorReachAvoidRolloutBuffer,
                                       TensorSafetyRolloutBuffer)
from safety_sb3.reach_avoid_mixin import _ReachAvoidPlumbing

AVOID, RA, CUM = backups.AVOID, backups.REACH_AVOID, backups.CUMULATIVE

# The full roster: class -> (Mode, Players). This IS the MAP table.
ROSTER = {
  SafetyPPO1P: (AVOID, 1), ReachAvoidPPO1P: (RA, 1), CumulativePPO1P: (CUM, 1),
  SafetyPPO2P: (AVOID, 2), ReachAvoidPPO2P: (RA, 2),
  SafetySAC1P: (AVOID, 1), ReachAvoidSAC1P: (RA, 1), CumulativeSAC1P: (CUM, 1),
  SafetySAC2P: (AVOID, 2), ReachAvoidSAC2P: (RA, 2),
  SafetyA2C1P: (AVOID, 1), ReachAvoidA2C1P: (RA, 1), CumulativeA2C1P: (CUM, 1),
  SafetyDQN1P: (AVOID, 1), CumulativeDQN1P: (CUM, 1),
}


def test_every_class_name_states_its_mode_and_players():
  """The name is the spec: parse it back and check it against the class."""
  prefix = {AVOID: "Safety", RA: "ReachAvoid", CUM: "Cumulative"}
  for cls, (mode, players) in ROSTER.items():
    assert cls._MODE == mode, f"{cls.__name__} declares _MODE={cls._MODE}"
    assert cls.__name__.startswith(prefix[mode]), cls.__name__
    assert cls.__name__.endswith(f"{players}P"), cls.__name__


def test_roster_is_exactly_what_the_package_exports():
  """No concrete learner escapes the table above (or the MAP naming law)."""
  exported = {
    getattr(safety_sb3, n) for n in safety_sb3.__all__
    if isinstance(getattr(safety_sb3, n), type)
    and n.endswith(("1P", "2P")) and not n.startswith("Abstract")
  }
  assert exported == set(ROSTER), exported ^ set(ROSTER)


def test_no_isaacs_or_gameplay_names_survive():
  """v0.4.0 retires the paper-named classes completely — no shims."""
  for gone in ("IsaacsPPO", "IsaacsSAC", "GameplayPPO", "GameplaySAC",
               "IsaacsPolicy", "SafetyPPO", "SafetySAC", "ReachAvoidPPO",
               "ReachAvoidSAC", "SafetyA2C", "SafetyDQN"):
    assert not hasattr(safety_sb3, gone), f"{gone} still exported"


# --- buffers are player-agnostic -------------------------------------------

def test_buffers_are_mode_only_and_match_their_mode():
  assert rollout_buffer_classes(AVOID) == (SafetyRolloutBuffer,
                                           TensorSafetyRolloutBuffer)
  assert rollout_buffer_classes(RA) == (ReachAvoidRolloutBuffer,
                                        TensorReachAvoidRolloutBuffer)
  assert rollout_buffer_classes(CUM) == (CumulativeRolloutBuffer,
                                         TensorCumulativeRolloutBuffer)
  for mode in backups.MODES:
    for buf in rollout_buffer_classes(mode):
      assert buf._MODE == mode
      # player-agnostic: nothing in a buffer name mentions a player count
      assert "1P" not in buf.__name__ and "2P" not in buf.__name__


def test_only_the_reach_avoid_buffer_carries_l():
  """The guard against the l_neg pattern: an avoid learner must not be able to
  smuggle a target margin into an avoid backup."""
  kw = dict(observation_space=_SingleEnv.observation_space,
            action_space=_SingleEnv.action_space, device="cpu", n_envs=1)
  ra = ReachAvoidRolloutBuffer(4, **kw)
  ra.reset()
  assert hasattr(ra, "l_x") and hasattr(ra, "terminal_type")
  for cls in (SafetyRolloutBuffer, CumulativeRolloutBuffer):
    b = cls(4, **kw)
    b.reset()
    assert not hasattr(b, "l_x"), cls
    assert not hasattr(b, "terminal_type"), cls
    # ...and its record_extras is a no-op even when handed a real l
    b.record_extras([{"l_x": 0.7}])
    assert not hasattr(b, "l_x"), cls


def test_reach_avoid_buffer_captures_l_itself():
  """The v0.4.0 move: the BUFFER captures l, not the algorithm."""
  buf = ReachAvoidRolloutBuffer(
    4, observation_space=_SingleEnv.observation_space,
    action_space=_SingleEnv.action_space, device="cpu", n_envs=2)
  buf.reset()
  buf.record_extras([{"l_x": 0.25}, {"l_x": -0.5}])
  assert np.allclose(buf.l_x[0], [0.25, -0.5])
  # missing l_x defaults to 0.0 rather than exploding
  buf.pos = 1
  buf.record_extras([{}, {}])
  assert np.allclose(buf.l_x[1], [0.0, 0.0])


# --- reach-avoid is a mixin, and avoid never gets it ------------------------

def test_reach_avoid_is_a_mixin_on_both_player_counts():
  """The 2x2 cannot be linearised, so the mode axis is composition."""
  for cls in (ReachAvoidPPO1P, ReachAvoidPPO2P, ReachAvoidA2C1P):
    assert issubclass(cls, _ReachAvoidPlumbing), cls
  # the mixin is NOT a base of the update loops -- it composes with both
  assert not issubclass(safety_sb3.AbstractPPO1P, _ReachAvoidPlumbing)
  assert not issubclass(safety_sb3.AbstractPPO2P, _ReachAvoidPlumbing)


def test_avoid_learners_never_receive_reach_avoid_machinery():
  """REGRESSION: this replaces the ``_is_reach_avoid`` flag.

  Before v0.4.0 the avoid classes INHERITED the reach-avoid class and switched
  its machinery off through a predicate, so every l-touching line needed a
  guard. Now they simply never receive it, and the predicate is unreachable --
  so it is gone.
  """
  for cls in (SafetyPPO1P, SafetyPPO2P, SafetySAC1P, SafetySAC2P,
              CumulativePPO1P, SafetyA2C1P, SafetyDQN1P):
    assert not issubclass(cls, _ReachAvoidPlumbing), cls
  # and nowhere in the roster at all
  for cls in ROSTER:
    assert not hasattr(cls, "_is_reach_avoid"), cls


def test_two_player_avoid_gives_both_players_the_avoid_game():
  """The bug this structure prevents: the min player used to get the wrong
  backup because its buffer was hardcoded rather than looked up from _MODE."""
  m = SafetyPPO2P("MlpPolicy", _TwoPlayerEnv(), ctrl_action_dim=1, n_steps=8,
                  batch_size=8, device="cpu")
  assert m.rollout_buffer._MODE == AVOID
  assert m.dstb_rollout_buffer._MODE == AVOID
  assert not hasattr(m.dstb_rollout_buffer, "l_x")


def test_both_two_player_buffers_get_the_same_kwargs():
  """v0.4.0 fix: terminal_type reached the ctrl buffer but NOT the dstb buffer,
  so the two players could score terminal states differently."""
  m = ReachAvoidPPO2P("MlpPolicy", _TwoPlayerEnv(), ctrl_action_dim=1,
                      terminal_type="g", n_steps=8, batch_size=8, device="cpu")
  assert m.rollout_buffer.terminal_type == "g"
  assert m.dstb_rollout_buffer.terminal_type == "g"


def test_buffer_mode_mismatch_is_refused():
  """In the on-policy families the buffer IS the backup, so a mismatch is not a
  type error -- it is silently solving the wrong problem."""
  with pytest.raises(TypeError, match="carries the backup"):
    SafetyPPO1P("MlpPolicy", _SingleEnv(), n_steps=8, batch_size=8,
                device="cpu", rollout_buffer_class=ReachAvoidRolloutBuffer)


# --- the one hole in the product -------------------------------------------

def test_dqn_has_no_reach_avoid_and_says_so():
  assert not hasattr(safety_sb3, "ReachAvoidDQN1P")
  with pytest.raises(ValueError, match="l\\(s\\)"):
    SafetyDQN1P("MlpPolicy", _DiscreteEnv(), mode=backups.REACH_AVOID,
                device="cpu")


def test_terminal_type_is_first_class_on_the_on_policy_family():
  for cls in (ReachAvoidPPO1P, ReachAvoidA2C1P):
    for tt in ("all", "g"):
      m = cls("MlpPolicy", _SingleEnv(), terminal_type=tt, n_steps=32,
              device="cpu")
      assert m.rollout_buffer.terminal_type == tt, (cls, tt)
    with pytest.raises(ValueError):
      cls("MlpPolicy", _SingleEnv(), terminal_type="bogus", n_steps=32,
          device="cpu")


# --- envs -------------------------------------------------------------------

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


class _DiscreteEnv(_SingleEnv):
  action_space = gym.spaces.Discrete(3)
