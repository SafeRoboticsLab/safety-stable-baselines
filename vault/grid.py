"""4D grid Hamilton-Jacobi reach-avoid value iteration — the certifiable oracle.

Computes the reach-avoid value V(x; mu) over x = [v, theta, theta_dot, psi_dot] by robust
value iteration on f_cert:

    V_{k+1}(x) = min( g_odd(x),  max_u  min_d  V_k( f_cert(x, u) + d ) )

  * g_odd  = signed distance to the ODD (f_cert.odd_margin), negative outside the envelope.
  * max_u  = the ego's best recovery control (a grid over [tau_L, tau_R]).
  * min_d  = the worst bounded disturbance (+/- EBAR_PSI yaw-accel forcing) -> robust certificate.
  * exit-unsafe boundary: transitions leaving the grid box take the unsafe value V_OOB.

Safe set = {V >= 0}. Output: data/grid_reachavoid_odd.npz (per-mu V + axes) — the exact oracle
that distill.py regresses the deployable V_mlp against (and verifies conservative to).

Run:
  python -m vault.grid --smoke                 # tiny grid, single mu, foreground
  python -m vault.grid                         # full solve -> data/grid_reachavoid_odd.npz
  python -m vault.grid --mu 0.6 --iters 400    # one mu, no save (inspection)
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from . import config as C
from . import f_cert as F

# Grid resolution; theta (the failure axis) is finest. Box strictly contains the ODD.
AXES_FULL = [np.linspace(-0.5, 1.7, 13), np.linspace(-1.35, 1.35, 29),
             np.linspace(-6.0, 6.0, 17), np.linspace(-3.0, 3.0, 19)]
AXES_SMOKE = [np.linspace(-0.5, 1.7, 9), np.linspace(-1.35, 1.35, 15),
              np.linspace(-6.0, 6.0, 11), np.linspace(-3.0, 3.0, 11)]
V_OOB = -1.0  # value assigned to transitions that exit the grid box (exit-unsafe BC)


def control_grid(n):
    """n x n grid over [tau_L, tau_R] in [-TAU_MAX, TAU_MAX]^2."""
    g = np.linspace(-C.TAU_MAX, C.TAU_MAX, n)
    return np.array([[a, b] for a in g for b in g])


def _stencil(axes, strides, points):
    """Multilinear-interpolation indices + weights for `points` on `axes` (16 corners/point)."""
    n = len(points)
    cell_i, frac = [], []
    for d in range(4):
        a = axes[d]
        x = np.clip(points[:, d], a[0], a[-1])
        i = np.clip(np.searchsorted(a, x, side="right") - 1, 0, len(a) - 2)
        cell_i.append(i.astype(np.int64))
        frac.append((x - a[i]) / (a[i + 1] - a[i]))
    idx = np.zeros((n, 16), np.int64)
    w = np.ones((n, 16))
    for c in range(16):
        bits = [(c >> d) & 1 for d in range(4)]
        flat = np.zeros(n, np.int64)
        for d in range(4):
            flat += (cell_i[d] + bits[d]) * strides[d]
            w[:, c] *= frac[d] if bits[d] else (1 - frac[d])
        idx[:, c] = flat
    return idx, w.astype(np.float32)


def solve(mu, axes, controls, max_iters=400, tol=1e-3, verbose=True):
    """Robust reach-avoid value iteration for a single mu. Returns (V[grid-shape], n_iters)."""
    dims = [len(a) for a in axes]
    strides = np.array([dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1])
    lo = np.array([a[0] for a in axes])
    hi = np.array([a[-1] for a in axes])
    cells = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 4)
    nc = len(cells)
    g = F.odd_margin(cells)

    xn = np.empty((nc, len(controls), 4))                       # precompute transitions
    for j, u in enumerate(controls):
        for i in range(nc):
            xn[i, j] = F.f_cert_step(cells[i], u, mu)
    dpsi = C.EBAR_PSI * C.DT                                     # robust: +/- yaw-accel forcing
    disturbed = np.concatenate([(xn + np.array([0, 0, 0, s * dpsi])).reshape(-1, 4)
                                for s in (+1, -1)], 0)
    oob = (disturbed < lo).any(1) | (disturbed > hi).any(1)
    idx, w = _stencil(axes, strides, np.clip(disturbed, lo, hi))

    V = g.copy()
    for k in range(max_iters):
        vals = (V[idx] * w).sum(1)
        vals[oob] = V_OOB                                       # exit-unsafe
        vals = vals.reshape(2, nc, len(controls)).min(0).max(1)  # min_d then max_u
        Vn = np.minimum(g, vals)
        dv = np.max(np.abs(Vn - V))
        V = Vn
        if verbose and k % 50 == 0:
            print(f"  mu={mu} iter {k:3d}: dV={dv:.4f} safe={100 * np.mean(V >= 0):.1f}%", flush=True)
        if dv < tol:
            break
    return V.reshape(dims), k + 1


def main():
    ap = argparse.ArgumentParser(description="4D grid HJ reach-avoid value iteration")
    ap.add_argument("--smoke", action="store_true", help="small grid, single mu, foreground")
    ap.add_argument("--mu", type=float, default=None, help="solve a single mu (does not save)")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--controls", type=int, default=5, help="per-axis control samples (n x n)")
    args = ap.parse_args()

    axes = AXES_SMOKE if args.smoke else AXES_FULL
    controls = control_grid(3 if args.smoke else args.controls)
    mus = [args.mu] if args.mu is not None else ([0.3] if args.smoke else list(C.MU_SLICES))
    dims = [len(a) for a in axes]
    print(f"grid {'x'.join(map(str, dims))} = {int(np.prod(dims))} cells, {len(controls)} controls, "
          f"robust ODD reach-avoid (exit-unsafe BC, min_d max_u)\n")

    values = {}
    for mu in mus:
        t0 = time.time()
        V, iters = solve(mu, axes, controls, max_iters=args.iters)
        values[mu] = V
        print(f"mu={mu}: safe-set {100 * np.mean(V >= 0):4.1f}% of grid | {iters} iters {time.time() - t0:.0f}s")

    if not args.smoke and args.mu is None:
        np.savez(C.GRID_NPZ, **{f"V_mu{int(m * 10)}": V for m, V in values.items()},
                 axes=np.array(axes, dtype=object))
        print(f"\nsaved -> {C.GRID_NPZ}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
