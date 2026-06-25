"""CBF-QP least-restrictive value filter.

Treats a value function V as a control barrier function on f_cert. Admits the task control
u_task iff the worst-case one-step value stays >= eps; otherwise overrides with the control that
maximizes it (least-restrictive), with an exit-the-domain guard:

  u = u_task                                   if  min_d V(f_cert(x, u_task) + d) >= eps
    = argmax_{u in U}  min_d V(f_cert(x, u) + d)    otherwise

min_d is over the bounded yaw-accel disturbance (+/- EBAR_PSI); transitions leaving the value's
domain take the unsafe value. This is the offline/Python reference for the deployed C value filter
(which ships in the vault-controller repo). Two value backends:

  ValueFilter.from_mlp()   conservative deployable V_mlp (distill.py)  -- matches deployment
  ValueFilter.from_grid()  exact grid V (grid.py)                      -- oracle baseline
"""
from __future__ import annotations

import numpy as np

from . import config as C
from . import f_cert as F


def control_grid(n=5):
    g = np.linspace(-C.TAU_MAX, C.TAU_MAX, n)
    return np.array([[a, b] for a in g for b in g])


class ValueFilter:
    """Least-restrictive CBF-QP filter over a value function ``value_fn(X[N,4], mu) -> V[N]``."""

    def __init__(self, value_fn, lo, hi, controls=None, eps=0.0):
        self._V = value_fn
        self.lo = np.asarray(lo, float)
        self.hi = np.asarray(hi, float)
        self.U = control_grid() if controls is None else np.asarray(controls, float)
        self.eps = eps

    def _worst_next_value(self, x, controls, mu):
        xn = np.array([F.f_cert_step(x, u, mu) for u in controls])
        oob = (xn < self.lo).any(1) | (xn > self.hi).any(1)        # exit-domain = unsafe
        dpsi = np.array([0.0, 0.0, 0.0, C.EBAR_PSI * C.DT])
        v = np.minimum(self._V(np.clip(xn + dpsi, self.lo, self.hi), mu),
                       self._V(np.clip(xn - dpsi, self.lo, self.hi), mu))
        v[oob] = -1.0
        return v

    def filter(self, x, u_task, mu):
        """Return (u_safe, overridden)."""
        if self._worst_next_value(x, np.array([u_task]), mu)[0] >= self.eps:
            return np.asarray(u_task, float), False
        vs = self._worst_next_value(x, self.U, mu)
        return self.U[int(np.argmax(vs))].copy(), True

    # --- value backends ----------------------------------------------------
    @classmethod
    def from_mlp(cls, models_dir=None, eps=0.0, controls=None):
        """The conservative deployable V_mlp (matches what the robot would carry)."""
        import json
        import torch
        from .distill import VNet
        md = models_dir or C.MODELS
        cfg = json.loads((md / "v_mlp.json").read_text())
        net = VNet(cfg["h"])
        net.load_state_dict(torch.load(md / "v_mlp.pt"))
        net.eval()
        nlo, nhi, delta = np.array(cfg["nlo"]), np.array(cfg["nhi"]), cfg["delta"]

        def value_fn(X4, mu):
            X = np.column_stack([X4, np.full(len(X4), mu)])
            Xn = (2 * (X - nlo) / (nhi - nlo) - 1).astype(np.float32)
            with torch.no_grad():
                return net(torch.tensor(Xn)).numpy() - delta

        return cls(value_fn, nlo[:4], nhi[:4], controls=controls, eps=eps)

    @classmethod
    def from_grid(cls, npz=None, eps=0.0, controls=None):
        """The exact grid V (nearest certified mu slice). Oracle baseline."""
        from scipy.interpolate import RegularGridInterpolator as RGI
        d = np.load(npz or C.GRID_NPZ, allow_pickle=True)
        axes = [np.asarray(a, float) for a in d["axes"]]
        dims = [len(a) for a in axes]
        rgis = {m: RGI(axes, d[f"V_mu{int(m * 10)}"].reshape(dims), bounds_error=False, fill_value=None)
                for m in C.MU_SLICES}
        slices = np.array(C.MU_SLICES)

        def value_fn(X4, mu):
            nearest = float(slices[np.argmin(np.abs(slices - mu))])
            return rgis[nearest](X4)

        lo = np.array([a[0] for a in axes]); hi = np.array([a[-1] for a in axes])
        return cls(value_fn, lo, hi, controls=controls, eps=eps)
