"""f_cert — the certified one-step model for the 4-state balance robot.

Single source of the dynamics used by the RL env, the grid HJ solver, and the value
filter. State x = [v, theta, theta_dot, psi_dot]; control u = [tau_L, tau_R] (N*m);
mu = friction coefficient.

f_cert = opt6 lossless coupled dynamics
       + per-wheel friction-cone cap on the control (slip, with centrifugal load transfer)
       + measured coupled friction residual (pitch damping + yaw scrub)
       integrated one symplectic-Euler step (config.DT).

Margins (safe iff >= 0):
  margin(X)     : nominal binding of pitch-topple and roll-liftoff (env reward / nominal grid).
  odd_margin(X) : signed distance to the FULL ODD (pitch, roll under the worst roll wrench,
                  v-in-ODD, psi-in-ODD). Negative outside the envelope, so leaving the ODD is
                  failure by construction -- no out-of-domain extrapolation.
"""
from __future__ import annotations

import numpy as np

from . import config as C
from .dynamics import CoupledOpt6

_opt6 = CoupledOpt6()


def wheel_loads(v, psid):
    """Per-wheel normal forces with centrifugal load transfer (outer wheel loaded)."""
    n_tot = C.MASS * C.GRAV
    dn = C.MASS * abs(v * psid) * C.COM_H / C.TRACK
    nl, nr = (n_tot / 2 - dn, n_tot / 2 + dn) if v * psid >= 0 else (n_tot / 2 + dn, n_tot / 2 - dn)
    return n_tot, max(nl, 0.0), max(nr, 0.0)


def f_cert_step(x, u, mu):
    """One certified step. x=[v,theta,theta_dot,psi_dot], u=[tau_L,tau_R], mu -> next x."""
    v, th, thd, psid = x
    n_tot, nl, nr = wheel_loads(v, psid)
    u_cap = np.array([np.clip(u[0], -mu * nl * C.WHEEL_R, mu * nl * C.WHEEL_R),
                      np.clip(u[1], -mu * nr * C.WHEEL_R, mu * nr * C.WHEEL_R)])
    xd = np.asarray(_opt6.f(x, u_cap), float)
    xd[2] += -C.C_THETA * thd                                            # measured pitch damping
    xd[3] += -((C.YAW_K0 + C.YAW_KC * mu) * np.tanh(psid / C.YAW_EPS)
               + C.YAW_KV * psid) * (n_tot / (C.MASS * C.GRAV))          # measured yaw scrub
    v2 = v + xd[0] * C.DT
    thd2 = thd + xd[2] * C.DT
    psid2 = psid + xd[3] * C.DT
    th2 = th + thd2 * C.DT
    return np.array([v2, th2, thd2, psid2])


def margin(X):
    """Nominal safety margin (pitch-topple AND roll-liftoff). Vectorized over X[..., 4]."""
    X = np.asarray(X)
    v, th, psid = X[..., 0], X[..., 1], X[..., 3]
    g_pitch = (C.THETA_MAX - np.abs(th)) / C.THETA_MAX
    g_roll = 1.0 - np.abs(v * psid) / C.A_TIP
    return np.minimum(g_pitch, g_roll)


def odd_margin(X, tau_roll_bar=None):
    """Signed distance to the FULL ODD (negative outside). Vectorized over X[..., 4].

    Binds pitch-topple, roll-liftoff under the worst roll wrench, v-in-ODD, psi-in-ODD.
    """
    X = np.asarray(X)
    trb = C.TAU_ROLL_BAR if tau_roll_bar is None else tau_roll_bar
    v, th, psid = X[..., 0], X[..., 1], X[..., 3]
    g_pitch = (C.THETA_MAX - np.abs(th)) / C.THETA_MAX
    a_lat = np.abs(v * psid) + trb / (C.MASS * C.COM_H)
    g_roll = 1.0 - a_lat / C.A_TIP
    g_v = np.minimum(v - C.V_ODD[0], C.V_ODD[1] - v) / ((C.V_ODD[1] - C.V_ODD[0]) / 2)
    g_psi = (C.PSI_ODD - np.abs(psid)) / C.PSI_ODD
    return np.minimum(np.minimum(g_pitch, g_roll), np.minimum(g_v, g_psi))
