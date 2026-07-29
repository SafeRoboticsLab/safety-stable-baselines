"""MuJoCo plant for the wheeled inverted pendulum — an INDEPENDENT simulator for
cross-validating the balance controller (vs the analytical small-angle plant and the
coupled reduced plant). Implements the Plant Protocol (dt/reset/step/get_state).

The controller works in the 4-state [v, theta, theta_dot, psi_dot]; MuJoCo carries the
full 3D state (freejoint chassis + 2 wheel hinges). This plant maps between them:

  STATE (MuJoCo -> 4-state), conventions from reduced_model.py:
    theta      = body pitch about +y; theta>0 = nose-down/forward.
    theta_dot  = body pitch rate (world-frame ω·ŷ_body component).
    psi_dot    = yaw rate about world +z.
    v          = forward axle speed along the body heading (the no-slip "v"); we report
                 the chassis forward linear velocity projected on the heading.

  CONTROL (4-state -> MuJoCo), u = [tau_L, tau_R] per wheel (N·m):
    applied to the lwheel/rwheel motors. +tau_sum drives +v. CORRECTED 2026-07-29: this previously claimed
    tau_L>tau_R drives +psi_dot. It does not -- a LEFT torque surplus yaws
    NEGATIVE (magnitudes live in the private release's parity fixtures). The reduced model and the plant agree; only the
    comment was inverted. test_yaw_sign_convention pins it.

Reset accepts a 4-state x0 and places the robot at that pitch/velocity (upright-ish);
get_state returns the 4-state. The 3D residual DOFs (roll, lateral) evolve freely —
that is the POINT: deviations from the planar no-slip model surface here.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

try:
    import mujoco
except ImportError as e:  # pragma: no cover
    raise ImportError("MuJoCo not installed: pip install mujoco") from e

from .mujoco_model import build_mjcf, load_params


class MujocoPlant:
    """Independent MuJoCo plant. wheel in {'torus','cylinder'}; mu = wheel-ground friction."""

    def __init__(self, wheel: str = "torus", mu: float = 1.0, dt: float = 0.001,
                 substeps: int | None = None,
                 solref: tuple = (0.02, 1.0), margin: float = 0.0,
                 imu_pos=None, imu_quat=None, contact_geometry: bool = False,
                 xml_override: str | None = None):
        self._p = load_params()
        self._R = self._p["wheel_radius"]
        self._has_imu = imu_pos is not None
        self.contact_geometry = contact_geometry
        self.mu = mu
        # xml_override: use a caller-supplied MJCF verbatim (e.g. build_mjcf's output with extra
        # static geoms spliced in for visualization) instead of building one from scratch here.
        # Must still define the chassis body + lwheel/rwheel joints + m_left/m_right actuators
        # that the rest of this class assumes.
        xml = xml_override if xml_override is not None else build_mjcf(
            wheel=wheel, mu=mu, params=self._p,
            solref=solref, margin=margin, imu_pos=imu_pos, imu_quat=imu_quat,
            contact_geometry=contact_geometry)
        self._m = mujoco.MjModel.from_xml_string(xml)
        self._m.opt.timestep = dt
        self._d = mujoco.MjData(self._m)
        self._dt_ctrl = 0.01          # the controller's step (get_state/step cadence)
        self._sub = substeps if substeps is not None else int(round(self._dt_ctrl / dt))
        self._lwheel = self._m.actuator("m_left").id
        self._rwheel = self._m.actuator("m_right").id
        self._chassis = self._m.body("chassis").id
        self._t = 0.0

    @property
    def dt(self) -> float:
        return self._dt_ctrl

    @property
    def model(self):
        """Raw MjModel, for callers (e.g. contact_margin.py) that need geom-level access
        beyond the 4-state abstraction."""
        return self._m

    @property
    def data(self):
        """Raw MjData, see `model`."""
        return self._d

    # --- state mapping helpers ------------------------------------------------
    def _pitch_yaw(self):
        """Extract (theta, psi) from the chassis quaternion. R = Rz(psi)·Ry(theta)."""
        quat = self._d.qpos[3:7].copy()          # (w,x,y,z)
        Rm = np.zeros(9)
        mujoco.mju_quat2Mat(Rm, quat)
        Rm = Rm.reshape(3, 3)
        # For R = Rz(psi) Ry(theta) (roll≈0): theta from -R[2,0]=sinθ? Derive directly:
        #   Ry(theta) tilts body x toward -z by theta; world-z of body-x = -sinθ.
        # body x-axis in world = first column of R.
        bx = Rm[:, 0]
        theta = np.arctan2(-bx[2], np.hypot(bx[0], bx[1]))   # pitch: nose-down +
        psi = np.arctan2(bx[1], bx[0])                       # heading
        return float(theta), float(psi), Rm

    def get_state(self) -> np.ndarray:
        d = self._d
        theta, psi, Rm = self._pitch_yaw()
        # angular velocity: qvel[3:6] is the chassis angular velocity in the LOCAL body
        # frame (MuJoCo freejoint: linear world, angular LOCAL — verified). Rotate to world.
        omega_world = Rm @ d.qvel[3:6].copy()
        by = Rm[:, 1]                       # body y-axis (pitch axis) in world
        theta_dot = float(omega_world @ by)
        psi_dot = float(omega_world[2])     # yaw rate about world z
        # forward speed v: chassis linear velocity projected on the heading (body x, planar)
        vlin_world = d.qvel[0:3].copy()
        bx = Rm[:, 0]
        heading = np.array([bx[0], bx[1], 0.0])
        n = np.linalg.norm(heading)
        heading = heading / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
        v = float(vlin_world @ heading)
        return np.array([v, theta, theta_dot, psi_dot], dtype=np.float64)

    def roll_deg(self) -> float:
        """Chassis roll about the heading (deg). ~0 upright; |roll| large = wheel liftoff/tip."""
        _, _, Rm = self._pitch_yaw()
        return float(np.degrees(np.arcsin(np.clip(Rm[2, 1], -1.0, 1.0))))

    def reset(self, x0: np.ndarray) -> None:
        """Place the robot at 4-state x0 (v, theta, theta_dot, psi_dot), upright base."""
        x0 = np.asarray(x0, dtype=np.float64)
        v, theta, theta_dot, psi_dot = x0
        mujoco.mj_resetData(self._m, self._d)
        d = self._d
        # pose: axle at z=R, pitch theta about +y (quat for Ry(theta))
        half = theta / 2.0
        d.qpos[0:3] = [0.0, 0.0, self._R]
        d.qpos[3:7] = [np.cos(half), 0.0, np.sin(half), 0.0]   # Ry(theta)
        # linear vel is WORLD (freejoint): forward v along body x (~world x at psi=0)
        d.qvel[0:3] = [v, 0.0, 0.0]
        # angular vel is LOCAL (freejoint): set local = R^T @ omega_world, with
        # omega_world = [0, theta_dot, psi_dot] at the pure-pitch reset orientation Ry(theta).
        # (Round-trips with the corrected get_state; at theta=0 reduces to [0,theta_dot,psi_dot].)
        d.qvel[3:6] = [-np.sin(theta) * psi_dot, theta_dot, np.cos(theta) * psi_dot]
        # wheel spin consistent with rolling at speed v: omega_wheel = v/R (no-slip init)
        d.qvel[6] = v / self._R
        d.qvel[7] = v / self._R
        mujoco.mj_forward(self._m, d)
        self._t = 0.0

    def step(self, u: np.ndarray) -> np.ndarray:
        """Apply u=[tau_L, tau_R] for one controller dt (substepped); return 4-state."""
        u = np.asarray(u, dtype=np.float64)
        lim = self._p.get("motor_torque_limit", 20.0)
        self._d.ctrl[self._lwheel] = float(np.clip(u[0], -lim, lim))
        self._d.ctrl[self._rwheel] = float(np.clip(u[1], -lim, lim))
        for _ in range(self._sub):
            mujoco.mj_step(self._m, self._d)
        self._t += self._dt_ctrl
        return self.get_state()

    def get_time(self) -> float:
        return self._t

    # --- full-state clone/restore (for a predictive monitor's shadow rollouts) -----------
    def get_full_state(self):
        """(qpos, qvel, t): the exact MuJoCo state (not just the reduced 4-state), so a shadow
        plant can be seeded to match this one exactly -- including roll/lateral drift the 4-state
        abstraction can't represent. Cheap (numpy copies, no XML/model rebuild)."""
        return self._d.qpos.copy(), self._d.qvel.copy(), self._t

    def set_full_state(self, qpos, qvel, t: float = 0.0) -> None:
        """Inverse of get_full_state. `self` must share the same MjModel structure as the plant
        the state was taken from (e.g. both built with the same contact_geometry/wheel/mu)."""
        self._d.qpos[:] = qpos
        self._d.qvel[:] = qvel
        mujoco.mj_forward(self._m, self._d)
        self._t = t

    def apply_disturbance(self, force=(0.0, 0.0, 0.0), torque=(0.0, 0.0, 0.0)) -> None:
        """Apply an external wrench on the chassis (world frame, at the body CoM) — the
        action channel for a learned adversary (e.g. an ISAACS xfrc-force adversary). It
        persists across mj_step until overwritten; call with zeros to clear."""
        self._d.xfrc_applied[self._chassis, :3] = force
        self._d.xfrc_applied[self._chassis, 3:] = torque

    # --- simulated IMU + encoder readouts (for the estimator-in-the-loop tests) ---
    def imu_reading(self):
        """(accel, gyro) at the IMU site, in the IMU/site frame, lever arm included. Needs imu_pos set."""
        a = self._m.sensor("acc_imu").adr[0]
        g = self._m.sensor("gyr_imu").adr[0]
        return self._d.sensordata[a:a + 3].copy(), self._d.sensordata[g:g + 3].copy()

    def wheel_velocities(self):
        """(omega_L, omega_R): wheel hinge angular velocities relative to the chassis (rad/s)."""
        return float(self._d.qvel[7]), float(self._d.qvel[6])

    # --- diagnostics for the gap analysis ------------------------------------
    def wheel_slip(self) -> float:
        """Longitudinal slip proxy: (wheel surface speed - chassis fwd speed)/max(.,eps).
        Zero under perfect rolling; nonzero = the no-slip assumption is violated."""
        st = self.get_state()
        v = st[0]
        omega_w = 0.5 * (self._d.qvel[6] + self._d.qvel[7])
        v_surface = omega_w * self._R
        denom = max(abs(v_surface), abs(v), 1e-3)
        return float((v_surface - v) / denom)
