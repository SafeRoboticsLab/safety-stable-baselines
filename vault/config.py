"""Shared physical constants and ODD spec for the 4-state safety package."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from .model_release import get_model_release

PKG = Path(__file__).resolve().parent
DATA = PKG / "data"
MODELS = PKG / "models"
GRID_NPZ = DATA / "grid_reachavoid_odd.npz"

# Model access is deliberately lazy: importing the package remains possible
# while a controller checkout is changing, but first artifact access still
# performs the strict byte-lock and per-file hash gates.
_RELEASE_VALUE_NAMES = frozenset(
    {
        "MODEL_RELEASE",
        "MASS",
        "TRACK",
        "WHEEL_R",
        "GRAV",
        "COM_H",
        "A_TIP",
        "C_THETA",
        "YAW_K0",
        "YAW_KC",
        "YAW_KV",
        "YAW_EPS",
        "CONTROLLER_TAU_MAX",
    }
)


@lru_cache(maxsize=1)
def _release_values() -> dict[str, Any]:
    release = get_model_release()
    composite = release.load_json("composite_params")
    limits = release.load_json("controller_limits")
    residual = release.load_json("coupled_residual")
    mass = composite["m_b"] + 2 * composite["m_wheel"]
    track = composite["wheel_sep"]
    wheel_radius = composite["wheel_radius"]
    gravity = limits["gravity"]
    com_height = composite["h_cm"] + wheel_radius
    yaw = residual["yaw"]
    return {
        "MODEL_RELEASE": release,
        "MASS": mass,
        "TRACK": track,
        "WHEEL_R": wheel_radius,
        "GRAV": gravity,
        "COM_H": com_height,
        "A_TIP": gravity * track / (2 * com_height),
        "C_THETA": residual["pitch"]["c0"],
        "YAW_K0": yaw["k0"],
        "YAW_KC": yaw["kc"],
        "YAW_KV": yaw["kv"],
        "YAW_EPS": yaw["eps"],
        "CONTROLLER_TAU_MAX": limits["motor_torque_limit"],
    }


def __getattr__(name: str) -> Any:
    if name in _RELEASE_VALUE_NAMES:
        return _release_values()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _RELEASE_VALUE_NAMES)

# --- integration + control ---
DT = 0.01                                       # control/integration step (s)
THETA_MAX = 1.2                                 # pitch failure bound (rad)
# Safety certification was solved over an 8 N*m action set.  The controller
# release separately preserves a 20 N*m deployed clip pending reconciliation.
TAU_MAX = 8.0                                   # per-wheel torque limit (N*m)

# --- ODD: the operational envelope we certify ---
V_ODD = (-0.3, 1.5)                             # forward-speed bounds (m/s)
PSI_ODD = 2.5                                   # |yaw rate| bound (rad/s)

# --- disturbance upper bounds (empirical + margin -> modeling assumptions) ---
EBAR_PSI = 3.4                                  # yaw-accel forcing bound (rad/s^2)
TAU_ROLL_BAR = 4.0                              # roll-wrench bound (N*m)

# --- friction (mu) for the mu-aware value function ---
MU_RANGE = (0.3, 1.0)
MU_SLICES = (0.3, 0.6, 1.0)

# --- contact-based failure modes (mujoco_env.py / contact_margin.py), not part of f_cert ---
SLAM_VEL_MAX = 1.0                              # contact normal speed (m/s) above which = a "slam"
LEG_GROUND_CLEARANCE = 0.02                     # m: (a) mujoco_model.py's leg-tip standoff from
                                                 # terrain at nominal upright pose, and (b) the
                                                 # reference length contact_margin.py normalizes by,
                                                 # so a nominal upright state gives margin ~1.0 --
                                                 # comparable to f_cert.margin's O(1) scale, instead
                                                 # of a raw meters value that would swamp min().

# --- full certified-domain bounds (matches grid.AXES_FULL) for contact-training reset coverage --
# ContactSafetyEnv resets across this whole range (not just a narrow near-upright band) so
# training sees challenging / already-failed ("no-win") states too, not only typical operation.
DOMAIN_V = (-0.5, 1.7)
DOMAIN_THETA = (-1.35, 1.35)                    # deliberately exceeds THETA_MAX=1.2 on both ends
DOMAIN_THETA_DOT = (-6.0, 6.0)
DOMAIN_PSI_DOT = (-3.0, 3.0)

# --- target set (target_margin.py): "at a stop" neighborhood, for reach-avoid training ---
# CORRECTED 2026-07-24 (Jaime): the target set must ALSO bound theta tightly, not just the
# velocity-type components -- a state with v=theta_dot=psi_dot=0 but theta=1.0 rad is NOT a
# forever-safe equilibrium for an inverted-pendulum balance robot (nonzero pitch means nonzero
# gravity torque; without continued active correction the state immediately starts accelerating
# away). "At a stop" means near-VERTICAL and near-still, not just instantaneously slow at some
# arbitrary pitch. The original version omitted theta entirely (only bounded v/theta_dot/psi_dot),
# which silently let "reached" mean "slow, at any pitch up to the constraint set's THETA_MAX" --
# caught from the (v,theta) pi-safe-set plot showing every small-v state marked "reached"
# regardless of theta. Membership in the target set ALSO requires being inside the constraint set
# (odd_margin/contact_margin >= 0); that's enforced by the reach-avoid Bellman's outer min(g,...),
# not by target_margin itself.
TARGET_V_STOP = 0.1                             # m/s
TARGET_THETA_STOP = 0.1                         # rad -- near-vertical, NOT the constraint set's
                                                 # much looser THETA_MAX=1.2
TARGET_THETA_DOT_STOP = 0.1                     # rad/s -- tightened to match TARGET_THETA_STOP's
                                                 # scale (was math.radians(10)=0.175, too loose)
TARGET_PSI_DOT_STOP = 1.0                       # rad/s -- deliberately the loosest of the four:
                                                 # killing a residual yaw spin once the robot is
                                                 # upright and otherwise still is comparatively easy
