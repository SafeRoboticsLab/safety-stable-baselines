"""4D grid Hamilton-Jacobi AVOID-ONLY safety value iteration — the certifiable oracle.

NAMING (corrected 2026-07-29): this module, its output `grid_reachavoid_odd.npz`, and
the `robust_odd_*` grids are all called "reach-avoid" for historical reasons. They are
NOT. The operator below is min(g, max_u min_d V) with NO target margin l(x): it
certifies "never fails", not "reaches a stop". The distinction is not cosmetic -- on the
same plant an avoid-only fallback leaves ~42% of the (v, theta) plane safe-but-never-
arriving, which a true reach-avoid objective collapses to zero. The reach-avoid work
lives in reach_avoid_sac.py and is a separate, learned object.

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
from collections.abc import Iterator, Sequence
from functools import cached_property
import time

import numpy as np

from . import config as C
from . import f_cert as F
from .model_release import get_model_release

_AXIS_NAMES = ("velocity", "theta", "theta_dot", "yaw_rate")


def _axes_from_contract(smoke: bool) -> list[np.ndarray]:
    axes = []
    for name in _AXIS_NAMES:
        specification = C.ODD_CONTRACT["grid_axes"][name]
        count_key = "smoke_nodes" if smoke else "nodes"
        axes.append(
            np.linspace(
                *specification["bounds"],
                int(specification[count_key]),
            )
        )
    # The v/yaw grids include the declared ODD boundaries exactly. The theta
    # grid strictly contains the pitch-failure set so exit-unsafe interpolation
    # has an exterior band on the failure axis.
    assert axes[0][0] <= C.V_ODD[0] <= C.V_ODD[1] <= axes[0][-1]
    assert axes[3][0] <= -C.PSI_ODD < C.PSI_ODD <= axes[3][-1]
    assert axes[1][0] < -C.THETA_MAX < C.THETA_MAX < axes[1][-1]
    return axes


class _ContractAxes(Sequence[np.ndarray]):
    """Lazily materialize grid axes after the release lock is first accessed."""

    def __init__(self, smoke: bool):
        self._smoke = smoke

    @cached_property
    def _axes(self) -> list[np.ndarray]:
        return _axes_from_contract(self._smoke)

    def __getitem__(self, index):
        return self._axes[index]

    def __len__(self) -> int:
        return len(self._axes)

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(self._axes)


AXES_FULL = _ContractAxes(smoke=False)
AXES_SMOKE = _ContractAxes(smoke=True)
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


def solve(mu, axes, controls, max_iters=1000, tol=1e-3, verbose=True):
    """Run one robust solve and return value, iterations, terminal dV, convergence."""
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
    return V.reshape(dims), k + 1, float(dv), bool(dv < tol)


def main():
    ap = argparse.ArgumentParser(description="4D grid HJ reach-avoid value iteration")
    ap.add_argument("--smoke", action="store_true", help="small grid, single mu, foreground")
    ap.add_argument("--mu", type=float, default=None, help="solve a single mu (does not save)")
    ap.add_argument(
        "--iters",
        type=int,
        default=1000,
        help="iteration budget only; convergence tolerance remains 1e-3",
    )
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
        V, iters, terminal_dv, converged = solve(
            mu, axes, controls, max_iters=args.iters
        )
        if not converged:
            print(
                f"GATE FAIL: mu={mu} did not converge after {iters} iterations "
                f"(dV={terminal_dv:.6g} >= 0.001); no grid saved",
                flush=True,
            )
            return 2
        values[mu] = V
        print(
            f"mu={mu}: CONVERGED dV={terminal_dv:.6g} | "
            f"safe-set {100 * np.mean(V >= 0):4.1f}% of grid | "
            f"{iters} iters {time.time() - t0:.0f}s"
        )

    if not args.smoke and args.mu is None:
        np.savez(C.GRID_NPZ, **{f"V_mu{int(m * 10)}": V for m, V in values.items()},
                 axes=np.array(axes, dtype=object))
        sidecar = get_model_release().write_external_artifact_sidecar(C.GRID_NPZ)
        print(f"\nsaved -> {C.GRID_NPZ}")
        print(f"model provenance -> {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
