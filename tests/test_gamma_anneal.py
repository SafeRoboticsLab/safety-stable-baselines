"""Gamma annealing: schedule math + on-by-default wiring across Safety* algos.

The reach-avoid boundary only sharpens as gamma -> 1, so every Safety* algorithm
anneals gamma 0.99 -> 0.9999 over the first 50% of training by default. These
tests pin (1) the schedule landmarks, (2) that the schedule is attached by
default and disabled by ``gamma_anneal=False``, and (3) that gamma actually
moves from init toward end during a real (tiny) training run on BOTH the
on-policy (PPO, buffer.gamma) and off-policy (SAC, self.gamma) tensor paths.
"""

import gymnasium.spaces as spaces
import torch as th
from stable_baselines3.common.callbacks import BaseCallback

from safety_sb3 import (GeometricGammaAnneal, ReachAvoidPPO, ReachAvoidSAC,
                        SafetyPPO, SafetySAC, make_default_gamma_schedule)
from safety_sb3.tensor_env import TensorVecEnv

DEV = "cuda" if th.cuda.is_available() else "cpu"
DT, TIMEOUT = 0.05, 60


# --------------------------------------------------------------- schedule math
def test_schedule_landmarks():
  s = make_default_gamma_schedule()
  assert abs(s(0.0) - 0.99) < 1e-9
  assert abs(s(0.25) - 0.999) < 1e-9      # safe_adaptation_dev landmark
  assert abs(s(0.5) - 0.9999) < 1e-9
  assert abs(s(0.75) - 0.9999) < 1e-9     # holds after anneal_frac
  assert abs(s(1.0) - 0.9999) < 1e-9
  xs = [s(i / 200) for i in range(201)]
  assert all(b >= a - 1e-12 for a, b in zip(xs, xs[1:])), "not monotone"
  print("[ok] landmarks 0.99 -> 0.999(@25%) -> 0.9999(@50%), monotone, held")


def test_schedule_seeded_from_gamma_and_degenerate():
  s = GeometricGammaAnneal(init=0.95, end=0.9999)
  assert abs(s(0.0) - 0.95) < 1e-9 and abs(s(0.5) - 0.9999) < 1e-9
  d = GeometricGammaAnneal(init=0.9999, end=0.9999)  # nothing to anneal
  assert all(abs(d(f) - 0.9999) < 1e-9 for f in (0.0, 0.3, 1.0))
  print("[ok] custom init seeds; init>=end holds constant")


# ------------------------------------------------------------------ tiny env
class DoubleIntegratorEnv(TensorVecEnv):
  """1-D double integrator avoid+reach TensorVecEnv (5-tuple step_tensor)."""

  def __init__(self, num_envs=64, device=DEV):
    super().__init__(num_envs, spaces.Box(-10, 10, (2,), "float32"),
                     spaces.Box(-1, 1, (1,), "float32"), device)
    self.x = th.zeros(num_envs, device=device)
    self.v = th.zeros(num_envs, device=device)
    self.t = th.zeros(num_envs, device=device)

  def _obs(self):
    return th.stack([self.x, self.v], dim=1)

  def _reset_ids(self, ids):
    n = int(ids.sum())
    self.x[ids] = th.rand(n, device=self.device) * 1.6 - 0.8
    self.v[ids] = th.rand(n, device=self.device) * 1.0 - 0.5
    self.t[ids] = 0.0

  def reset(self):
    self._reset_ids(th.ones(self.num_envs, dtype=th.bool, device=self.device))
    return self._obs()

  def step_tensor(self, actions):
    a = actions[:, 0].clamp(-1, 1)
    self.x = self.x + self.v * DT
    self.v = self.v + 3.0 * a * DT
    self.t = self.t + 1
    g = 1.0 - self.x.abs()
    l = 0.2 - (self.x - 0.5).abs()
    terminated = g < 0
    truncated = (self.t >= TIMEOUT) & ~terminated
    dones = terminated | truncated
    if bool(dones.any()):
      self._reset_ids(dones)
    return self._obs(), g, dones, truncated, l


class _GammaTrace(BaseCallback):
  """Snapshot (progress_fraction, model.gamma) each step."""

  def __init__(self):
    super().__init__()
    self.trace = []

  def _on_step(self) -> bool:
    total = self.model._total_timesteps or 1
    self.trace.append((self.model.num_timesteps / total, float(self.model.gamma)))
    return True


# ---------------------------------------------------------------- wiring: on by default
def test_on_by_default_and_toggle():
  env = DoubleIntegratorEnv(num_envs=8)
  on = SafetySAC("MlpPolicy", env, gamma=0.99, device=DEV, seed=0,
                 buffer_size=2000, learning_starts=0)
  assert on._gamma_schedule is not None, "gamma anneal must be ON by default"
  off = SafetySAC("MlpPolicy", env, gamma=0.99, device=DEV, seed=0,
                  buffer_size=2000, learning_starts=0, gamma_anneal=False)
  assert off._gamma_schedule is None, "gamma_anneal=False must disable it"
  # custom schedule respected verbatim
  sched = GeometricGammaAnneal(init=0.9, end=0.999)
  cust = SafetyPPO("MlpPolicy", env, gamma=0.9, n_steps=16, device=DEV,
                   gamma_anneal=sched)
  assert cust._gamma_schedule is sched
  print("[ok] default ON; gamma_anneal=False OFF; callable passthrough")


# ---------------------------------------------------------- functional: it moves
def _final_and_trace(algo_cls, off_policy):
  env = DoubleIntegratorEnv()
  kw = dict(gamma=0.99, device=DEV, seed=0,
            policy_kwargs=dict(net_arch=[32, 32]), verbose=0)
  if off_policy:
    kw.update(buffer_size=20_000, batch_size=256, learning_starts=500,
              train_freq=1, gradient_steps=4)
  else:
    kw.update(n_steps=64, batch_size=256, n_epochs=2)
  model = algo_cls("MlpPolicy", env, **kw)
  cb = _GammaTrace()
  model.learn(total_timesteps=12_000, callback=cb, log_interval=None)
  return model, cb.trace


def test_gamma_anneals_during_training_sac():
  model, trace = _final_and_trace(ReachAvoidSAC, off_policy=True)
  gammas = [g for _, g in trace]
  assert abs(gammas[0] - 0.99) < 5e-3, f"should start ~0.99, got {gammas[0]}"
  assert max(gammas) > 0.999, f"gamma should climb past 0.999, got max {max(gammas)}"
  assert abs(model.gamma - 0.9999) < 1e-6, f"end ~0.9999, got {model.gamma}"
  # monotone non-decreasing along the run
  assert all(b >= a - 1e-9 for a, b in zip(gammas, gammas[1:]))
  print(f"[ok] SAC gamma: {gammas[0]:.4f} -> {model.gamma:.4f} (max {max(gammas):.4f})")


def test_gamma_anneals_during_training_ppo():
  model, trace = _final_and_trace(ReachAvoidPPO, off_policy=False)
  gammas = [g for _, g in trace]
  # PPO consumes gamma off the rollout buffer -- verify BOTH tracked and buffer.
  assert abs(gammas[0] - 0.99) < 5e-3, f"should start ~0.99, got {gammas[0]}"
  assert max(gammas) > 0.999, f"gamma should climb past 0.999, got max {max(gammas)}"
  assert abs(model.gamma - 0.9999) < 1e-6, f"end ~0.9999, got {model.gamma}"
  assert abs(model.rollout_buffer.gamma - model.gamma) < 1e-9, \
      "rollout buffer gamma must track the annealed self.gamma"
  print(f"[ok] PPO gamma: {gammas[0]:.4f} -> {model.gamma:.4f}; "
        f"buffer.gamma={model.rollout_buffer.gamma:.4f}")


if __name__ == "__main__":
  test_schedule_landmarks()
  test_schedule_seeded_from_gamma_and_degenerate()
  test_on_by_default_and_toggle()
  test_gamma_anneals_during_training_sac()
  test_gamma_anneals_during_training_ppo()
  print("\nALL GAMMA-ANNEAL TESTS PASSED")
