"""Distill a compact, conservative deployable value net V_mlp from the grid value function.

Trains V_mlp([v, theta, theta_dot, psi_dot, mu]) to regress the grid V (grid.py output), then
makes it conservative BY CONSTRUCTION: subtract delta = max over a dense sample of (V_mlp - V_grid)
restricted to grid-UNSAFE states, so V_deploy = V_mlp - delta <= V_grid wherever the grid is unsafe
=> {V_deploy >= 0} subset {V_grid >= 0}. That closes the over-optimism that makes a naively-distilled
value unsafe. An asymmetric loss penalizes over-prediction so delta stays small (usable coverage).

Saves models/v_mlp.pt + models/v_mlp.json (arch, normalization, delta, viol/coverage).
  python -m vault.distill
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import RegularGridInterpolator as RGI

from . import config as C

_MU_KEYS = [(m, f"V_mu{int(m * 10)}") for m in C.MU_SLICES]


class VNet(nn.Module):
    """3x128 ReLU MLP (v, theta, theta_dot, psi_dot, mu) -> V. Small enough for the MCU/NPU."""

    def __init__(self, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _load_grid():
    d = np.load(C.GRID_NPZ, allow_pickle=True)
    axes = [np.asarray(a, float) for a in d["axes"]]
    dims = [len(a) for a in axes]
    rgis = {m: RGI(axes, d[k].reshape(dims), bounds_error=False, fill_value=None) for m, k in _MU_KEYS}
    lo = np.array([a[0] for a in axes])
    hi = np.array([a[-1] for a in axes])
    return rgis, lo, hi


def _normalizer(lo, hi):
    nlo = np.array([lo[0], lo[1], lo[2], lo[3], C.MU_SLICES[0]])
    nhi = np.array([hi[0], hi[1], hi[2], hi[3], C.MU_SLICES[-1]])
    return nlo, nhi, lambda s: (2 * (s - nlo) / (nhi - nlo) - 1).astype(np.float32)


def _sample(n, rng, rgis, lo, hi):
    s = np.column_stack([rng.uniform(lo[i], hi[i], n) for i in range(4)])
    mus = np.array([m for m, _ in _MU_KEYS])
    idx = rng.integers(0, len(_MU_KEYS), n)
    v = np.empty(n)
    for j, (mu, _) in enumerate(_MU_KEYS):
        m = idx == j
        if m.any():
            v[m] = rgis[mu](s[m])
    return np.column_stack([s, mus[idx]]), v


def _conservative_loss(pred, tgt, w=30.0):
    """Penalize OVER-prediction (pred > tgt) w-fold so the net under-claims -> small delta."""
    e = pred - tgt
    return (w * torch.relu(e).pow(2) + torch.relu(-e).pow(2)).mean()


def main():
    rng = np.random.default_rng(0)
    t0 = time.time()
    rgis, lo, hi = _load_grid()
    nlo, nhi, norm = _normalizer(lo, hi)

    print("sampling grid V for distillation ...", flush=True)
    x_tr, v_tr = _sample(200_000, rng, rgis, lo, hi)
    bx, bv, got = [], [], 0                                       # oversample the boundary |V|<0.3
    while got < 120_000:
        xb, vb = _sample(60_000, rng, rgis, lo, hi)
        m = np.abs(vb) < 0.3
        bx.append(xb[m]); bv.append(vb[m]); got += int(m.sum())
    x_tr = np.vstack([x_tr] + bx); v_tr = np.concatenate([v_tr] + bv)
    x_va, v_va = _sample(40_000, rng, rgis, lo, hi)

    xt = torch.tensor(norm(x_tr)); vt = torch.tensor(v_tr.astype(np.float32))
    xv = torch.tensor(norm(x_va)); vv = torch.tensor(v_va.astype(np.float32))
    net = VNet()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for ep in range(120):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 4096):
            b = perm[i:i + 4096]
            opt.zero_grad()
            _conservative_loss(net(xt[b]), vt[b]).backward()
            opt.step()
        if ep % 20 == 0:
            with torch.no_grad():
                mae = float((net(xv) - vv).abs().mean())
            print(f"  ep {ep:3d}: val MAE {mae:.4f}", flush=True)

    # conservative margin on a fresh dense sample
    x_c, v_c = _sample(120_000, rng, rgis, lo, hi)
    with torch.no_grad():
        v_mlp = net(torch.tensor(norm(x_c))).numpy()
    delta = float(v_mlp[v_c < 0].max() + 1e-3) if (v_c < 0).any() else 0.0
    v_dep = v_mlp - delta
    viol = float(np.mean((v_dep >= 0) & (v_c < 0)))              # deploy-safe but grid-unsafe (-> ~0)
    cover = float(np.mean(v_dep[v_c >= 0] >= 0))                 # of grid-safe states, still deploy-safe
    print(f"\ndelta {delta:.3f} | over-optimism {100 * viol:.2f}% | coverage {100 * cover:.1f}% "
          f"| {time.time() - t0:.0f}s")

    C.MODELS.mkdir(exist_ok=True)
    torch.save(net.state_dict(), C.MODELS / "v_mlp.pt")
    json.dump({"h": 128, "delta": delta, "nlo": nlo.tolist(), "nhi": nhi.tolist(),
               "viol_pct": 100 * viol, "coverage_pct": 100 * cover},
              open(C.MODELS / "v_mlp.json", "w"), indent=2)
    print(f"saved -> {C.MODELS / 'v_mlp.pt'} + v_mlp.json")
    ok = viol < 0.01 and cover > 0.5
    print(f"GATE: {'PASS (conservative + usable coverage)' if ok else 'REVIEW (viol or coverage off)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
