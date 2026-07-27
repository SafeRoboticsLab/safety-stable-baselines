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

from pathlib import Path

import numpy as np

from . import config as C
from . import f_cert as F
from .model_release import get_model_release


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
    def from_mlp(cls, models_dir=None, eps=0.0, controls=None, mode="unladen"):
        """Load the conservative release MLP or a sidecar-pinned external MLP."""
        import json
        import torch
        from .distill import VNet

        release = get_model_release()
        if mode not in ("unladen", "laden"):
            raise ValueError(f"mode must be 'unladen' or 'laden', got {mode!r}")
        if models_dir is None:
            cfg_path = release.artifact_path(f"v_mlp_{mode}_json")
            weights_path = release.artifact_path(f"v_mlp_{mode}_pt")
        else:
            md = Path(models_dir)
            cfg_path = release.verify_external_artifact(md / "v_mlp.json")
            weights_path = release.verify_external_artifact(md / "v_mlp.pt")
        cfg = json.loads(cfg_path.read_text())
        net = VNet(cfg["h"])
        net.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True)
        )
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
        """Load Jaime's ODD reach-avoid grid, regenerated for the v2.2 model."""
        release = get_model_release()
        path = C.GRID_NPZ if npz is None else Path(npz)
        path = release.verify_external_artifact(path)
        return cls._from_grid_path(path, eps=eps, controls=controls)

    @classmethod
    def from_capability_grid(cls, mode="unladen", eps=0.0, controls=None):
        """Explicitly load a controller capability grid.

        Capability grids encode the controller release's tip/capability
        contract, not Jaime's SSB ODD reach-avoid contract. Keeping this as a
        separate constructor prevents a model-release update from silently
        changing the meaning of ``from_grid()``.
        """
        release = get_model_release()
        if mode not in ("unladen", "laden"):
            raise ValueError(f"mode must be 'unladen' or 'laden', got {mode!r}")
        path = release.artifact_path(f"grid_{mode}")
        metadata = (0.0, 0.0, 0.0) if mode == "unladen" else (0.3, 0.2, 0.0)
        return cls._from_grid_path(
            path, eps=eps, controls=controls, expected_metadata=metadata
        )

    @classmethod
    def _from_grid_path(
        cls, path, eps=0.0, controls=None, expected_metadata=None
    ):
        """Build a filter from one already provenance-checked grid path."""
        from scipy.interpolate import RegularGridInterpolator as RGI

        with np.load(path, allow_pickle=True) as data:
            axes = [np.asarray(axis, float) for axis in data["axes"]]
            dims = [len(axis) for axis in axes]
            slices = (
                np.asarray(data["mus"], float)
                if "mus" in data.files
                else np.asarray(C.MU_SLICES, float)
            )
            values = {}
            for mu in slices:
                release_key = f"V_{float(mu):.1f}"
                legacy_key = f"V_mu{int(round(float(mu) * 10))}"
                key = release_key if release_key in data.files else legacy_key
                if key not in data.files:
                    raise ValueError(
                        f"grid {path} has no value array for mu={float(mu):g}"
                    )
                values[float(mu)] = np.asarray(data[key], float).reshape(dims)
            if expected_metadata is not None:
                metadata = tuple(float(data[name]) for name in ("dh", "dm", "bump"))
                if not np.allclose(
                    metadata, expected_metadata, atol=1e-12, rtol=0.0
                ):
                    raise ValueError(
                        f"release grid {path} metadata {metadata} does not match "
                        f"expectation {expected_metadata}"
                    )
        rgis = {
            mu: RGI(
                axes, value, bounds_error=False, fill_value=None
            )
            for mu, value in values.items()
        }

        def value_fn(X4, mu):
            nearest = float(slices[np.argmin(np.abs(slices - mu))])
            return rgis[nearest](X4)

        lo = np.array([a[0] for a in axes]); hi = np.array([a[-1] for a in axes])
        return cls(value_fn, lo, hi, controls=controls, eps=eps)
