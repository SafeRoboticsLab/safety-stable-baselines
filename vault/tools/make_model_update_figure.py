"""Historical figure: v2.1 -> v2.2 model update on the pre-widening ODD.

Same solver, same ODD contract, same axes, same friction slices -- only the
robot model changed. The v2.1 grid is read from git (Jaime's committed grid at
2c4f73d); the v2.2 grid is pinned at 272406c, where the old-domain v2.2
regeneration was introduced, before the 2026-07-27 +/-4 domain widening.
This script intentionally does not read the active grid: safe-set fractions
on different domains are not comparable.

Run from the repo root:
    PYTHONPATH=$PWD:../vault-controller python -m vault.tools.make_model_update_figure
or directly:
    python vault/tools/make_model_update_figure.py
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap

REPO = Path(__file__).resolve().parents[2]
V21_REF = "2c4f73d:vault/data/grid_reachavoid_odd.npz"
V22_REF = "272406cac12bee123cb1f5b2090a3b3c6fd2c7b9:vault/data/grid_reachavoid_odd.npz"
OUT = REPO / "vault/docs/model_update_safeset_v2_1_to_v2_2.pdf"

SAFE, UNSAFE = "#2e9e4f", "#d1443c"
CMAP = ListedColormap([UNSAFE, SAFE])
MUS = [("V_mu3", 0.3), ("V_mu6", 0.6), ("V_mu10", 1.0)]
L, W = 0.065, 0.87


def load(path):
    z = np.load(path, allow_pickle=True)
    axes = [np.asarray(a, float) for a in z["axes"]]
    return axes, {k: np.asarray(z[k], float).reshape([len(a) for a in axes])
                  for k, _ in MUS}


def load_git_ref(reference):
    with tempfile.NamedTemporaryFile(suffix=".npz") as temporary:
        temporary.write(
            subprocess.run(
                ["git", "show", reference],
                cwd=REPO,
                capture_output=True,
                check=True,
            ).stdout
        )
        temporary.flush()
        return load(temporary.name)


def slice_vtheta(axes, V):
    """V on the (v, theta) plane at theta_dot = psi_dot = 0."""
    return V[:, :, int(np.argmin(np.abs(axes[2]))), int(np.argmin(np.abs(axes[3])))]


def lean_at_rest(axes, V):
    sl = slice_vtheta(axes, V)
    band = np.degrees(axes[1][sl[int(np.argmin(np.abs(axes[0])))] >= 0.0])
    return (band.min(), band.max()) if band.size else (float("nan"),) * 2


def main() -> int:
    ax21, g21 = load_git_ref(V21_REF)
    ax22, g22 = load_git_ref(V22_REF)
    assert [len(a) for a in ax21] == [len(a) for a in ax22], "axes differ; not comparable"

    fig = plt.figure(figsize=(8.5, 6.4))
    fig.patch.set_facecolor("white")
    fig.text(L, 0.955, "Certified safe set under the v2.1 -> v2.2 model update",
             fontsize=13, fontweight="bold", va="top")
    fig.text(L, 0.918, "Same solver, same ODD contract, same axes. Only the robot model changed.",
             fontsize=9.5, color="#555555", va="top")

    rows = []
    for key, mu in MUS:
        f21, f22 = 100 * (g21[key] >= 0).mean(), 100 * (g22[key] >= 0).mean()
        rows.append((mu, f21, f22, lean_at_rest(ax21, g21[key]), lean_at_rest(ax22, g22[key])))

    body = ("The camera-inclusive v2.2 robot is 1.4% heavier with a 4% higher centre of mass. The "
            "certified volume shrinks slightly at every\nfriction — "
            + ", ".join(f"$\\mu$={m}: {a:.1f}% $\\rightarrow$ {b:.1f}%" for m, a, b, _, _ in rows)
            + " — but this is not a uniform inward shift.\nAt $\\mu\\geq$0.6 the at-rest lean band "
              "actually widens (e.g. $\\mu$=0.6: [-11°, +55°] $\\rightarrow$ [-17°, +61°]) while the "
              "set loses volume elsewhere,\nso the boundary moves both ways. The overall shape is "
              "preserved, which is what a mass and inertia update alone should look like.")
    fig.patches.append(mpatches.FancyBboxPatch(
        (L - 0.008, 0.752), W + 0.016, 0.112, transform=fig.transFigure,
        boxstyle="round,pad=0.006", facecolor="#f4f5f7", edgecolor="#dcdee3", linewidth=0.8))
    fig.text(L, 0.856, body, fontsize=8.2, color="#1f2430", va="top", linespacing=1.6)

    fig.legend(handles=[mpatches.Patch(color=SAFE, label="certified safe, v2.2  (V $\\geq$ 0)"),
                        mpatches.Patch(color=UNSAFE, label="not certified, v2.2"),
                        plt.Line2D([0], [0], ls="--", color="#1a1a1a", label="v2.1 safe-set boundary")],
               loc="upper left", bbox_to_anchor=(L, 0.738), ncol=3, frameon=False, fontsize=8.4)

    gs = fig.add_gridspec(1, 3, left=0.075, right=0.96, top=0.62, bottom=0.135, wspace=0.30)
    v_ax, th_ax = ax22[0], ax22[1]
    dv, dt = (v_ax[1] - v_ax[0]) / 2, (th_ax[1] - th_ax[0]) / 2
    extent = [v_ax[0] - dv, v_ax[-1] + dv, th_ax[0] - dt, th_ax[-1] + dt]

    for col, ((key, mu), (_, f21, f22, l21, l22)) in enumerate(zip(MUS, rows)):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow((slice_vtheta(ax22, g22[key]) >= 0).T.astype(float), origin="lower",
                  aspect="auto", extent=extent, cmap=CMAP, vmin=0, vmax=1,
                  interpolation="nearest")
        ax.contour(v_ax, th_ax, slice_vtheta(ax21, g21[key]).T, levels=[0.0],
                   colors="#1a1a1a", linewidths=1.2, linestyles="--")
        ax.set_title(f"$\\mu$ = {mu}", fontsize=9.5, fontweight="bold", loc="left", pad=18)
        ax.text(0, 1.015, f"{f21:.1f}% $\\rightarrow$ {f22:.1f}% safe   ·   lean at rest "
                         f"[{l22[0]:+.0f}°, {l22[1]:+.0f}°]",
                transform=ax.transAxes, fontsize=7, color="#555555", va="bottom")
        ax.set_xlabel("v  (m/s)", fontsize=8)
        ax.set_ylabel(r"$\theta$  (rad)", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.text(L, 0.075,
             "Slices at $\\dot\\theta=\\dot\\psi=0$. Grids pinned at 2c4f73d and 272406c on the "
             "same historical contract; not the active +/-4 domain.\nReproduce: "
             "python vault/tools/make_model_update_figure.py",
             fontsize=7.4, color="#666666", va="top", linespacing=1.5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f"wrote {OUT}")
    for mu, f21, f22, l21, l22 in rows:
        print(f"  mu={mu}: safe {f21:.2f}% -> {f22:.2f}%   lean@rest {l21} -> {l22}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
