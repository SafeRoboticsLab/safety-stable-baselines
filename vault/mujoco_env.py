"""ContactSafetyEnv -- MuJoCo-contact reach-avoid RL environment for the 4-state balance robot.

Same shape as env.py's BalanceSafetyEnv (obs/action/SafetySAC contract), but steps the real
MuJoCo plant (mujoco_plant.MujocoPlant, contact_geometry=True) instead of the fast analytic
f_cert model, and folds in the new contact/slam margins (contact_margin.py) alongside the
existing FULL-ODD margin (f_cert.odd_margin) -- so pitch-topple, centrifugal-liftoff, AND leaving
the certified v/psi_dot envelope all terminate episodes, in addition to the new non-wheel-contact
and slam failure modes.

Uses odd_margin, not the plain margin(): margin() only binds pitch/roll and does NOT bound forward
speed or yaw rate, which let early rollouts drift to v several multiples of V_ODD's upper bound --
far outside anything the critic was ever trained on -- before eventually failing unpredictably
once the extrapolation broke down. odd_margin adds the missing v-in-ODD and psi-in-ODD terms
("leaving the ODD is failure by construction", per f_cert.py's own docstring), which keeps
training inside the domain the critic actually has support over.

Far slower per step than env.py (real MuJoCo substeps vs. one closed-form call) -- expect
orders of magnitude fewer env-steps/sec. Start training with far fewer --steps than the
analytic pipeline's 300000 default (see train_contact.py).

  observation = [v, theta, theta_dot, psi_dot, mu]   (mu-aware value, matches env.py)
  action      = normalized [tau_L, tau_R] in [-1, 1]^2
  reward      = min(f_cert.odd_margin(x), contact_margin.margin(plant))
  terminated  = reward < 0  (left the safe set, any failure mode)
                OR, if reach_avoid=True, target_margin.target_margin(x) >= 0 (reached the target
                set -- see target_margin.py). The env's reward stays the avoid margin g(x) either
                way; the reach-avoid Bellman extension (reach_avoid_sac.py) recomputes the target
                margin l(x) on the fly from stored next_observations, so no buffer schema change
                is needed here -- reach_avoid only changes WHEN an episode ends, not what reward
                is returned.

No obstacles / flat terrain / no [x,y,psi] in the state -- this is the "warmup" track. Unlike
env.py, there's no hand-crafted bounded-disturbance injection here; a learned adversary would
instead drive plant.apply_disturbance (the ISAACS hook already in mujoco_plant.py).

reset() samples across C.DOMAIN_* (matching grid.AXES_FULL's certified domain), not just a narrow
near-upright band -- this deliberately includes challenging and already-failed ("no-win") states
(e.g. theta beyond THETA_MAX), so the critic and pi_safe learn the actual limits of the robot's
capabilities instead of only ever seeing typical operating conditions.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import config as C
from . import contact_margin as CM
from . import f_cert as F
from . import target_margin as T
from .mujoco_plant import MujocoPlant


class ContactSafetyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, mu_range=C.MU_RANGE, tau_max=C.TAU_MAX, max_steps=200,
                 wheel="cylinder", seed=None, reach_avoid=False):
        super().__init__()
        self.mu_range = mu_range
        self.tau_max = tau_max
        self.max_steps = max_steps
        self.wheel = wheel
        self.reach_avoid = reach_avoid
        hi = np.array([2.0, 1.5, 8.0, 3.0, 1.0], np.float32)
        self.observation_space = spaces.Box(-hi, hi, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self.plant = None

    def _obs(self):
        return np.array([*self.x, self.mu], np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.mu = float(self._rng.uniform(*self.mu_range))
        # Rebuild the plant at the sampled mu (~1ms, negligible next to a 200-step rollout) --
        # mu is a compiled-in MuJoCo contact-friction coefficient, not a runtime arg like in
        # f_cert_step, so it can't be changed on an existing plant instance.
        self.plant = MujocoPlant(wheel=self.wheel, mu=self.mu, contact_geometry=True)
        x0 = np.array([self._rng.uniform(*C.DOMAIN_V), self._rng.uniform(*C.DOMAIN_THETA),
                       self._rng.uniform(*C.DOMAIN_THETA_DOT), self._rng.uniform(*C.DOMAIN_PSI_DOT)])
        self.plant.reset(x0)
        self.x = self.plant.get_state()
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        u = np.clip(np.asarray(action, float), -1.0, 1.0) * self.tau_max
        self.x = self.plant.step(u)
        self.t += 1
        g = min(float(F.odd_margin(self.x)), CM.margin(self.plant))
        l = float(T.target_margin(self.x))
        reached_target = self.reach_avoid and g >= 0.0 and l >= 0.0
        terminated = bool(g < 0.0 or reached_target)
        truncated = bool(self.t >= self.max_steps)
        return self._obs(), g, terminated, truncated, {"mu": self.mu, "g": g, "l": l}
