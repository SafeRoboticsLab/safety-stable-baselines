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
        "COM_H_WHOLE",
        "A_TIP",
        "N_STATIC_MIN",
        "C_THETA",
        "YAW_K0",
        "YAW_KC",
        "YAW_KV",
        "YAW_EPS",
        "CONTROLLER_TAU_MAX",
        "ODD_CONTRACT",
        "THETA_MAX",
        "TAU_MAX",
        "V_ODD",
        "PSI_ODD",
        "EBAR_PSI",
        "TAU_ROLL_BAR",
        "MU_RANGE",
        "MU_SLICES",
        "DOMAIN_V",
        "DOMAIN_THETA",
        "DOMAIN_THETA_DOT",
        "DOMAIN_PSI_DOT",
    }
)


@lru_cache(maxsize=1)
def _release_values() -> dict[str, Any]:
    release = get_model_release()
    composite = release.load_json("composite_params")
    limits = release.load_json("controller_limits")
    residual = release.load_json("coupled_residual")
    odd_contract = release.load_json("odd_contract")
    mass = composite["m_b"] + 2 * composite["m_wheel"]
    track = composite["wheel_sep"]
    wheel_radius = composite["wheel_radius"]
    gravity = limits["gravity"]
    yaw = residual["yaw"]
    # Roll-constraint quantities come from the generated roll artifact, never from
    # the reduced dynamics parameters. h_cm is a sprung-EQUIVALENT pendulum length
    # (M_total * z_com / m_sprung), not a geometric height, so h_cm + wheel_radius
    # overstates the whole-robot centre-of-mass height above ground by 8.2 mm. The
    # rollover moment balance needs the true height, and the reduction does not
    # emit it -- hence the separate artifact.
    roll = release.load_json("roll_constraint_params")
    com_height_whole = roll["whole_com_height_above_ground_m"]
    a_tip = roll["no_liftoff_lateral_acceleration_m_s2"]["symmetric_worst_case"]
    return {
        "MODEL_RELEASE": release,
        "MASS": mass,
        "TRACK": track,
        "WHEEL_R": wheel_radius,
        "GRAV": gravity,
        "COM_H_WHOLE": com_height_whole,
        "A_TIP": a_tip,
        "N_STATIC_MIN": (
            roll["static_wheel_normal_loads_n"]["conservative_lower_per_wheel"]
        ),
        "C_THETA": residual["pitch"]["c0"],
        "YAW_K0": yaw["k0"],
        "YAW_KC": yaw["kc"],
        "YAW_KV": yaw["kv"],
        "YAW_EPS": yaw["eps"],
        "CONTROLLER_TAU_MAX": limits["motor_torque_limit"],
        "ODD_CONTRACT": odd_contract,
        "THETA_MAX": odd_contract["odd"]["theta_failure"]["bounds"][1],
        "TAU_MAX": odd_contract["control"]["tau_max"]["value"],
        "V_ODD": tuple(odd_contract["odd"]["velocity"]["bounds"]),
        "PSI_ODD": odd_contract["odd"]["yaw_rate"]["bounds"][1],
        "EBAR_PSI": odd_contract["disturbances"]["yaw_acceleration"]["value"],
        "TAU_ROLL_BAR": odd_contract["disturbances"]["roll_wrench"]["value"],
        "MU_RANGE": tuple(odd_contract["friction"]["range"]),
        "MU_SLICES": tuple(odd_contract["friction"]["slices"]),
        "DOMAIN_V": tuple(odd_contract["grid_axes"]["velocity"]["bounds"]),
        "DOMAIN_THETA": tuple(odd_contract["grid_axes"]["theta"]["bounds"]),
        "DOMAIN_THETA_DOT": tuple(
            odd_contract["grid_axes"]["theta_dot"]["bounds"]
        ),
        "DOMAIN_PSI_DOT": tuple(
            odd_contract["grid_axes"]["yaw_rate"]["bounds"]
        ),
    }


def __getattr__(name: str) -> Any:
    if name in _RELEASE_VALUE_NAMES:
        return _release_values()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _RELEASE_VALUE_NAMES)

# --- integration + control ---
DT = 0.01                                       # control/integration step (s)
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
