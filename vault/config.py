"""Repo-relative paths and shared physical constants / ODD spec for the 4-state
balance-safety package. Single source: every module imports its constants from here.
"""
from __future__ import annotations

import json
from pathlib import Path

PKG = Path(__file__).resolve().parent
DATA = PKG / "data"
MODELS = PKG / "models"
GRID_NPZ = DATA / "grid_reachavoid_odd.npz"

# --- robot parameters (vendored composite_params.json) ---
_CP = json.loads((DATA / "composite_params.json").read_text())
MASS = _CP["m_b"] + 2 * _CP["m_wheel"]        # total mass (kg)
TRACK = _CP["wheel_sep"]                        # wheel separation / track width (m)
WHEEL_R = _CP["wheel_radius"]                   # wheel radius (m)
GRAV = 9.81
COM_H = _CP["h_cm"] + WHEEL_R                   # CoM height above ground (m)
A_TIP = GRAV * TRACK / (2 * COM_H)             # no-liftoff lateral-accel bound (m/s^2)

# --- coupled friction residual fit (vendored) ---
_FR = json.loads((DATA / "coupled_residual_fit.json").read_text())
C_THETA = _FR["pitch"]["c0"]                    # pitch damping coefficient
_Y = _FR["yaw"]
YAW_K0, YAW_KC, YAW_KV, YAW_EPS = _Y["k0"], _Y["kc"], _Y["kv"], _Y["eps"]

# --- integration + control ---
DT = 0.01                                       # control/integration step (s)
THETA_MAX = 1.2                                 # pitch failure bound (rad)
TAU_MAX = 8.0                                   # per-wheel torque limit (N*m)

# --- ODD: the operational envelope we certify ---
V_ODD = (-0.3, 1.5)                             # forward-speed bounds (m/s)
PSI_ODD = 2.5                                   # |yaw rate| bound (rad/s)

# --- disturbance upper bounds (empirical + margin -> modeling assumptions) ---
EBAR_PSI = 3.4                                  # yaw-accel forcing bound (rad/s^2)
TAU_ROLL_BAR = 4.0                              # roll-wrench bound (N*m)

# --- friction (mu) for the mu-aware value function ---
MU_RANGE = (0.3, 1.0)
MU_SLICES = (0.3, 0.6, 1.0)                     # certified slices (grid keys V_mu3 / V_mu6 / V_mu10)
