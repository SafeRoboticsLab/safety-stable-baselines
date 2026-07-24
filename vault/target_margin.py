"""target_margin -- the target-set ("reach") margin for reach-avoid training.

Target set: the robot is at a stop -- near-VERTICAL (small theta) AND near-still (small v,
theta_dot, psi_dot). CORRECTED 2026-07-24: an earlier version of this module only bounded the
velocity-type components (v, theta_dot, psi_dot) and left theta unconstrained, on the reasoning
that theta is a position/orientation variable, not a velocity. That was wrong as a
characterization of a forever-safe "home": for an inverted-pendulum balance robot, a state with
zero velocities but theta=1.0 rad is NOT an equilibrium -- nonzero pitch means nonzero gravity
torque, so absent continued active correction the state immediately starts accelerating away from
it. "At a stop" has to mean upright-and-still, not merely instantaneously-slow-at-any-pitch. (This
surfaced concretely in a (v,theta) pi-safe-set plot: every small-v cell was marked "reached"
regardless of theta, which is the visual signature of theta being absent from the margin.)

Each component margin is scaled to read exactly +1 at zero (its center) and 0 at its threshold, so
all four are on a comparable O(1) scale -- avoiding the same tiny-margin/units mismatch that
contact_margin.py had to fix for the avoid side (see C.LEG_GROUND_CLEARANCE). Thresholds (see
config.py): theta and theta_dot are both tight (0.1 rad, 0.1 rad/s); psi_dot is deliberately looser
(1.0 rad/s) -- killing a residual yaw spin once the robot is upright and otherwise still is
comparatively easy, so it doesn't need the same tight margin.

Membership in the target set ALSO requires being inside the constraint set (odd_margin >= 0 and
contact_margin.margin >= 0) -- that is NOT checked here. It's enforced by the reach-avoid Bellman
recursion's outer min(g(x), ...) term (see reach_avoid_sac.py), which caps the value at the
current avoid margin regardless of how close to a stop the robot is. target_margin(x) is only the
"already at a stop" component margin l(x).
"""
from __future__ import annotations

import numpy as np

from . import config as C


def target_margin(X):
    """min of the four component margins (v, theta, theta_dot, psi_dot), vectorized over
    X[..., 4] (or [...,3+], theta_dot/psi_dot only need indices 1-3). Safe (>=0) iff the robot is
    upright (|theta| small) AND still (|v|, |theta_dot|, |psi_dot| all within their thresholds)."""
    X = np.asarray(X)
    v, theta, theta_dot, psi_dot = X[..., 0], X[..., 1], X[..., 2], X[..., 3]
    m_v = 1.0 - np.abs(v) / C.TARGET_V_STOP
    m_theta = 1.0 - np.abs(theta) / C.TARGET_THETA_STOP
    m_thd = 1.0 - np.abs(theta_dot) / C.TARGET_THETA_DOT_STOP
    m_psid = 1.0 - np.abs(psi_dot) / C.TARGET_PSI_DOT_STOP
    return np.minimum(np.minimum(m_v, m_theta), np.minimum(m_thd, m_psid))
