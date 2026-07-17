"""5-D bicycle in a plane of circular obstacles, with a circular goal.

The library's reference validation env. Small, dependency-light (numpy only),
trains to convergence in minutes on CPU, and -- the point -- it makes the
difference between the two problems VISIBLE:

* **avoid** (`SafetyPPO` / `IsaacsPPO`): nothing rewards motion. `g > 0` is
  already satisfied where the car starts, so the optimal avoid policy is to
  **sit still** (and swerve only if something approaches). That is correct, and
  it is the negative control.
* **reach-avoid** (`ReachAvoidPPO` / `GameplayPPO`): the car must drive to the
  goal without hitting anything.

So `reach_rate(reach-avoid) >> reach_rate(avoid)` is the discriminating
assertion, and it is exactly the assertion a wrong reach-avoid anchor fails: a
`g`-anchored backup makes sitting still worth `V = g > 0`, which beats driving,
so the "reach-avoid" car sits still too and becomes indistinguishable from the
avoid car. That regression is invisible to a value unit test and obvious here
-- see RELEASE_NOTES v0.2.0 and :mod:`safety_sb3.backups`.

**`v_min = 0` is load-bearing**: the car must be *able* to stop, or loitering is
not available and the contrast collapses to "drives in circles" vs "drives to
the goal".

Dynamics (Princeton race car; matches `safe_adaptation_dev`'s
`simulators/dynamics/bicycle5D_dstb.py`, reimplemented in numpy -- the reference
is JAX and carries a spline race track we do not need)::

    state   = [x, y, v, psi, delta]          position, speed, heading, steering
    control = [accel, omega]                 omega = steering RATE
    dstb    = [d0..d4]                       additive on every derivative

    xdot     = v cos(psi)         + d0
    ydot     = v sin(psi)         + d1
    vdot     = accel              + d2
    psidot   = v tan(delta) / L   + d3
    deltadot = omega              + d4

integrated with RK4, then `v` and `delta` clipped to their bounds.

Margins (the env contract -- see the README):

* ``g`` = signed distance from the car's rectangular footprint to the nearest
  obstacle, normalized. ``g >= 0`` iff not in collision. Rides on `reward`.
* ``l`` = goal radius minus distance from the car's center to the goal center,
  normalized. ``l >= 0`` iff the car is in the goal. Rides on `info["l_x"]`.

Both are clipped to +/-3 and +/-1 respectively (`CLAMP_G`, `CLAMP_L`).
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

# --- world defaults ---------------------------------------------------------
DT = 0.05
WHEELBASE = 0.257          # Princeton race car
V_MIN, V_MAX = 0.0, 2.0    # v_min = 0: the car CAN stop -> loitering exists
DELTA_LIM = 0.35           # steering angle limit (rad)
ACCEL_LIM = 2.0
OMEGA_LIM = 2.0
CAR_L, CAR_W = 0.42, 0.19  # rectangular footprint (length, width)
CLAMP_G, CLAMP_L = 3.0, 1.0
G_SCALE = 0.5
#: `l` MUST stay UNCLAMPED across the whole reachable domain, or the reach
#: gradient is zero where the car spawns and it never learns to move (measured:
#: with L_SCALE=2.0, l clamps to -1 beyond 2.4 m, but the goal spawns ~3 m out,
#: so the value is flat -1 at the start and the policy sits still forever).
#: l = (goal_r - dist)/L_SCALE clamps at dist = goal_r + L_SCALE. With
#: L_SCALE=4.0 and goal_r ~0.4 that is 4.4 m -- past the max start distance
#: (~3.6 m under randomization), so l stays graded everywhere the car explores.
#: Do NOT shrink it below (max_start_dist - goal_r) or the gradient dies at spawn.
L_SCALE = 4.0
#: reach-avoid VALUE ceiling at the goal center (see _l_of_dist). Bigger =
#: more value dynamic range for the certificate, but too big destabilizes the
#: PPO reach terminal. 0.1 (=goal_r/L_SCALE) recovers the original flat l.
GOAL_VALUE = 0.3
DSTB_LIM = np.array([0.05, 0.05, 0.3, 0.2, 0.2], dtype=np.float32)

#: (x, y, radius) -- a corridor of obstacles between start and goal
DEFAULT_OBSTACLES: Tuple[Tuple[float, float, float], ...] = (
  (1.0, 0.32, 0.28),
  (2.0, -0.40, 0.28),
)
DEFAULT_START = (0.0, 0.0)
DEFAULT_GOAL = (3.0, 0.0, 0.40)   # (x, y, radius)


def _l_of_dist(d, goal_r):
  """Target margin from distance-to-goal. PIECEWISE so the value has range:

  * INSIDE the goal (``d <= goal_r``): ``(goal_r - d)/goal_r`` -> +1 at the
    center, 0 at the boundary. This gives the reach-avoid VALUE real positive
    dynamic range (V can reach ~+1), so the ``V >= 0`` certificate is readable
    instead of a razor-thin sliver near 0.
  * OUTSIDE (``d > goal_r``): ``(goal_r - d)/L_SCALE`` -> a gentle negative
    gradient that stays informative out past the max spawn distance, so
    training still has a reach signal from far away.

  Continuous at the boundary (both branches give 0). A single linear scale
  cannot do both (steep-at-center needs a small divisor, far-gradient needs a
  large one) -- hence the split. ``GOAL_VALUE`` is the value at the goal center
  (the reach-avoid value ceiling); GOAL_VALUE = goal_r / L_SCALE recovers the
  original single-scale l.
  """
  inside = GOAL_VALUE * (goal_r - d) / goal_r
  outside = (goal_r - d) / L_SCALE
  return np.where(d <= goal_r, inside, outside)


def _box_sdf(px: np.ndarray, py: np.ndarray, hx: float, hy: float) -> np.ndarray:
  """Signed distance from points (in box frame) to an axis-aligned box."""
  qx, qy = np.abs(px) - hx, np.abs(py) - hy
  outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
  inside = np.minimum(np.maximum(qx, qy), 0.0)
  return outside + inside


class Bicycle5D:
  """Pure dynamics: RK4 on the 5-D kinematic bicycle."""

  def __init__(self, wheelbase: float = WHEELBASE, dt: float = DT):
    self.L, self.dt = wheelbase, dt

  def deriv(self, s: np.ndarray, u: np.ndarray, d: np.ndarray) -> np.ndarray:
    x, y, v, psi, delta = s
    return np.array([
      v * np.cos(psi) + d[0],
      v * np.sin(psi) + d[1],
      u[0] + d[2],
      v * np.tan(delta) / self.L + d[3],
      u[1] + d[4],
    ], dtype=np.float64)

  def step(self, s: np.ndarray, u: np.ndarray, d: np.ndarray) -> np.ndarray:
    dt = self.dt
    k1 = self.deriv(s, u, d)
    k2 = self.deriv(s + k1 * dt / 2, u, d)
    k3 = self.deriv(s + k2 * dt / 2, u, d)
    k4 = self.deriv(s + k3 * dt, u, d)
    s = s + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
    s[2] = np.clip(s[2], V_MIN, V_MAX)
    s[4] = np.clip(s[4], -DELTA_LIM, DELTA_LIM)
    s[3] = (s[3] + np.pi) % (2 * np.pi) - np.pi
    return s


class BicycleGoal(gym.Env):
  """Drive to the goal circle without hitting the obstacle circles.

  :param adversary: if True the action is one ``Box(2 + 5)``: ``[0:2]`` is the
      control (accel, omega) and ``[2:7]`` the disturbance, matching the
      two-player env contract (``ctrl_action_dim=2``). Use with ``IsaacsPPO``
      (avoid) or ``GameplayPPO`` (reach-avoid).
  :param randomize: jitter obstacle/goal placement each episode (default True).
      Off makes the scene memorizable — only turn it off for a fixed-scene demo.
  :param terminate_on_goal: end the episode once the goal is reached. Safe to
      leave on for BOTH problems: the avoid car never gets there, so the avoid
      problem is unchanged in practice, and it keeps demos clean.
  """

  metadata = {"render_modes": ["rgb_array"]}

  def __init__(self, adversary: bool = False, randomize: bool = True,
               obstacles: Sequence[Tuple[float, float, float]] = DEFAULT_OBSTACLES,
               goal: Tuple[float, float, float] = DEFAULT_GOAL,
               start: Tuple[float, float] = DEFAULT_START,
               timeout: int = 300, terminate_on_goal: bool = True):
    self.dyn = Bicycle5D()
    self.adversary = bool(adversary)
    self.randomize = bool(randomize)
    self._obstacles0 = np.asarray(obstacles, dtype=np.float64).reshape(-1, 3)
    self._goal0 = np.asarray(goal, dtype=np.float64)
    self._start = np.asarray(start, dtype=np.float64)
    self.timeout = int(timeout)
    self.terminate_on_goal = bool(terminate_on_goal)

    n_obs = len(self._obstacles0)
    # [v, sin psi, cos psi, delta] + goal(2, car frame) + per-obstacle(3)
    self.observation_space = gym.spaces.Box(
      -np.inf, np.inf, (4 + 2 + 3 * n_obs,), dtype=np.float32)
    if self.adversary:
      hi = np.concatenate([[ACCEL_LIM, OMEGA_LIM], DSTB_LIM]).astype(np.float32)
    else:
      hi = np.array([ACCEL_LIM, OMEGA_LIM], np.float32)
    self.action_space = gym.spaces.Box(-hi, hi, dtype=np.float32)
    self.ctrl_action_dim = 2

  # --- margins --------------------------------------------------------------
  def _g(self, s: np.ndarray) -> float:
    """Signed distance from the car RECTANGLE to the nearest obstacle circle."""
    if len(self.obstacles) == 0:
      return float(CLAMP_G)          # nothing to hit -> maximally safe
    x, y, _, psi, _ = s
    c, sn = np.cos(-psi), np.sin(-psi)
    dx = self.obstacles[:, 0] - x
    dy = self.obstacles[:, 1] - y
    px = c * dx - sn * dy          # obstacle centers in the car frame
    py = sn * dx + c * dy
    sd = _box_sdf(px, py, CAR_L / 2, CAR_W / 2) - self.obstacles[:, 2]
    return float(np.clip(sd.min() / G_SCALE, -CLAMP_G, CLAMP_G))

  def _l(self, s: np.ndarray) -> float:
    d = np.hypot(s[0] - self.goal[0], s[1] - self.goal[1])
    return float(np.clip(_l_of_dist(d, self.goal[2]), -CLAMP_L, CLAMP_L))

  def _obs(self, s: np.ndarray) -> np.ndarray:
    x, y, v, psi, delta = s
    c, sn = np.cos(-psi), np.sin(-psi)

    def to_car(gx, gy):
      dx, dy = gx - x, gy - y
      return c * dx - sn * dy, sn * dx + c * dy

    gx, gy = to_car(self.goal[0], self.goal[1])
    parts = [v, np.sin(psi), np.cos(psi), delta, gx, gy]
    for ox, oy, r in self.obstacles:
      rx, ry = to_car(ox, oy)
      parts += [rx, ry, r]
    return np.asarray(parts, dtype=np.float32)

  # --- gym ------------------------------------------------------------------
  def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    self.obstacles = self._obstacles0.copy()
    self.goal = self._goal0.copy()
    if self.randomize:
      n = len(self.obstacles)
      self.obstacles[:, 0] += self.np_random.uniform(-0.30, 0.30, n)
      self.obstacles[:, 1] += self.np_random.uniform(-0.35, 0.35, n)
      self.obstacles[:, 2] += self.np_random.uniform(-0.05, 0.08, n)
      self.goal[0] += self.np_random.uniform(-0.25, 0.25)
      self.goal[1] += self.np_random.uniform(-0.50, 0.50)
    # v starts in [0, 0.4] -- the range INCLUDES 0 on purpose. The discriminating
    # question is whether reach-avoid INITIATES from a standstill, so a share of
    # episodes must start there; but ALWAYS zero (with a fixed map) lets the
    # policy memorize one trajectory instead of learning a controller.
    self.s = np.array([
      self._start[0] + self.np_random.uniform(-0.15, 0.15),
      self._start[1] + self.np_random.uniform(-0.30, 0.30),
      self.np_random.uniform(0.0, 0.4),           # speed
      self.np_random.uniform(-0.35, 0.35),        # heading
      self.np_random.uniform(-0.10, 0.10),        # steering angle
    ], dtype=np.float64)
    self.t = 0
    return self._obs(self.s), {"l_x": self._l(self.s)}

  def step(self, action):
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    u = np.clip(a[:2], [-ACCEL_LIM, -OMEGA_LIM], [ACCEL_LIM, OMEGA_LIM])
    d = (np.clip(a[2:7], -DSTB_LIM, DSTB_LIM) if self.adversary
         else np.zeros(5))
    self.s = self.dyn.step(self.s, u, d)
    self.t += 1
    g, l = self._g(self.s), self._l(self.s)
    reached = l >= 0.0
    terminated = bool(g < 0.0) or bool(reached and self.terminate_on_goal)
    truncated = self.t >= self.timeout
    return (self._obs(self.s), float(g), terminated, truncated,
            {"l_x": l, "reached": bool(reached), "collided": bool(g < 0.0)})

  # --- rendering (optional; matplotlib only imported here) ------------------
  def render_rgb(self, trail: Optional[np.ndarray] = None, fig=None, ax=None,
                 title: Optional[str] = None) -> np.ndarray:
    """One RGB frame ``(H, W, 3)`` uint8 — stack these into a wandb GIF.

    Pass a reused ``fig``/``ax`` when rendering many frames; making a new
    figure per frame is what makes naive matplotlib animation slow.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    own = fig is None
    if own:
      fig, ax = plt.subplots(figsize=(6, 2.6), dpi=64)
    ax.clear()
    self.render_frame(ax=ax, trail=trail)
    if title:
      ax.set_title(title, fontsize=9)
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    if own:
      plt.close(fig)
    return buf

  def render_frame(self, ax=None, trail: Optional[np.ndarray] = None):
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt
    if ax is None:
      _, ax = plt.subplots(figsize=(7, 3))
    for ox, oy, r in self.obstacles:
      ax.add_patch(mp.Circle((ox, oy), r, color="#c0392b", alpha=0.85, zorder=2))
    ax.add_patch(mp.Circle(self.goal[:2], self.goal[2], color="#27ae60",
                           alpha=0.45, zorder=1))
    if trail is not None and len(trail):
      ax.plot(trail[:, 0], trail[:, 1], "-", color="#2c3e50", lw=1.2,
              alpha=0.8, zorder=3)
    x, y, _, psi, _ = self.s
    car = mp.Rectangle((-CAR_L / 2, -CAR_W / 2), CAR_L, CAR_W,
                       color="#2980b9", zorder=4)
    tr = (plt.matplotlib.transforms.Affine2D()
          .rotate(psi).translate(x, y) + ax.transData)
    car.set_transform(tr)
    ax.add_patch(car)
    ax.set_aspect("equal")
    ax.set_xlim(-0.8, self.goal[0] + 1.0)
    ax.set_ylim(-1.4, 1.4)
    return ax
