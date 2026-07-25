"""Per-network / per-actor learning rates + StepLR decay for the SAC family.

Audit issues 1 & 2 vs safe_adaptation_dev: each network (ctrl actor, dstb actor,
critic) and each actor's entropy (alpha) optimizer must carry its OWN lr, and the
blanket SB3 ``_update_learning_rate`` must NOT collapse them onto one value.

CPU-only, tiny budget -- runs alongside GPU jobs.

  python -m pytest tests/test_isaacs_lr.py -q
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safety_sb3 import GameplaySAC, IsaacsSAC  # noqa: E402
from tests.test_tensor_sac import TwoPlayerDoubleIntegratorEnv  # noqa: E402

DEV = "cpu"

# Distinct lrs so any collapse/misassignment is unambiguous.
CTRL_LR = 3e-4
CRITIC_LR = 7e-4
DSTB_LR = 5e-4
ENT_LR = 1e-3
DSTB_ENT_LR = 2e-3


def _lr(opt) -> float:
  return float(opt.param_groups[0]["lr"])


def _make(algo_cls=GameplaySAC, gamma_anneal=False, **kw):
  env = TwoPlayerDoubleIntegratorEnv(num_envs=32, device=DEV)
  model = algo_cls(
    "MlpPolicy", env, ctrl_action_dim=1,
    buffer_size=20_000, batch_size=256, learning_starts=64,
    train_freq=1, gradient_steps=1, gamma=0.95,
    learning_rate=CTRL_LR, critic_learning_rate=CRITIC_LR,
    dstb_learning_rate=DSTB_LR, ent_coef_lr=ENT_LR,
    dstb_ent_coef_lr=DSTB_ENT_LR,
    policy_kwargs=dict(net_arch=[32, 32]), verbose=0, device=DEV, seed=0,
    gamma_anneal=gamma_anneal, **kw)
  return model, env


def _assert_distinct(model):
  assert _lr(model.actor.optimizer) == CTRL_LR
  assert _lr(model.critic.optimizer) == CRITIC_LR
  assert _lr(model.dstb_actor.optimizer) == DSTB_LR
  assert _lr(model.ent_coef_optimizer) == ENT_LR
  assert _lr(model.dstb_ent_coef_optimizer) == DSTB_ENT_LR


def test_distinct_lrs_at_construction():
  """Each optimizer carries its own lr right after construction (before any
  train step), not the shared learning_rate."""
  model, _ = _make()
  _assert_distinct(model)
  # The five lrs really are distinct (would trivially pass if all shared).
  lrs = {_lr(model.actor.optimizer), _lr(model.critic.optimizer),
         _lr(model.dstb_actor.optimizer), _lr(model.ent_coef_optimizer),
         _lr(model.dstb_ent_coef_optimizer)}
  assert len(lrs) == 5, f"expected 5 distinct lrs, got {sorted(lrs)}"
  print("[ok] distinct per-network lrs at construction")


def test_lrs_survive_train():
  """A short learn() calls train() -> _update_learning_rate; the dedicated lrs
  must NOT collapse onto the single shared learning_rate."""
  model, _ = _make()
  model.learn(total_timesteps=64 * 6, log_interval=None)
  _assert_distinct(model)  # unchanged: constant dedicated lrs survive train()
  print("[ok] dedicated lrs not collapsed after train()")


def test_isaacs_and_gamma_anneal_alpha_reset():
  """IsaacsSAC (avoid) construction shares the machinery; the gamma-anneal alpha
  reset must fire without error AND rebuild the entropy optimizers at their
  dedicated lrs (issue 3), not the shared one."""
  model, _ = _make(algo_cls=IsaacsSAC, gamma_anneal=True)
  _assert_distinct(model)
  # Directly trigger the reset the gamma jump performs.
  model._reset_entropy_temp()
  assert _lr(model.ent_coef_optimizer) == ENT_LR, "ctrl ent lr lost on reset"
  assert _lr(model.dstb_ent_coef_optimizer) == DSTB_ENT_LR, "dstb ent lr lost on reset"
  # And a short run with the default discrete-jump schedule must not error
  # (a jump at 20% fires _on_gamma_jump -> _reset_entropy_temp on the tensor path).
  model.learn(total_timesteps=64 * 8, log_interval=None)
  print("[ok] IsaacsSAC alpha reset preserves dedicated entropy lrs; jump ran")


def test_steplr_decays_dedicated_lrs():
  """With lr_schedule on, the ctrl/dstb/critic lrs StepLR-decay by lr_decay every
  lr_period env-steps; the entropy lrs stay constant (alpha schedule = follow-up)."""
  model, _ = _make(lr_schedule=True, lr_period=64, lr_decay=0.5, lr_end=1e-6)
  # After learning_starts (64) + a few train steps, num_timesteps crosses one or
  # more 64-step periods, so ctrl/dstb/critic are decayed below their base.
  model.learn(total_timesteps=64 * 6, log_interval=None)
  n = int(model.num_timesteps)
  num_decay = n // 64
  assert num_decay >= 1, f"test needs >=1 decay period, got {n} steps"
  exp_critic = max(CRITIC_LR * 0.5 ** num_decay, 1e-6)
  assert abs(_lr(model.critic.optimizer) - exp_critic) < 1e-9, (
    f"critic lr {_lr(model.critic.optimizer)} != StepLR {exp_critic}")
  # ctrl and dstb decayed from their own bases by the same factor.
  assert abs(_lr(model.actor.optimizer) - max(CTRL_LR * 0.5 ** num_decay, 1e-6)) < 1e-9
  assert abs(_lr(model.dstb_actor.optimizer) - max(DSTB_LR * 0.5 ** num_decay, 1e-6)) < 1e-9
  # entropy lrs unaffected by the network StepLR.
  assert _lr(model.ent_coef_optimizer) == ENT_LR
  assert _lr(model.dstb_ent_coef_optimizer) == DSTB_ENT_LR
  print(f"[ok] StepLR: {num_decay} decays -> critic lr {_lr(model.critic.optimizer):.2e}")


if __name__ == "__main__":
  test_distinct_lrs_at_construction()
  test_lrs_survive_train()
  test_isaacs_and_gamma_anneal_alpha_reset()
  test_steplr_decays_dedicated_lrs()
  print("ALL ISAACS-LR TESTS PASSED")
