"""Figure: the certified safe set on the ACTIVE +-4 ODD contract.

Reads the committed grid and the ODD contract from the pinned model release, so
the picture and the contract cannot drift apart. Companion to
make_model_update_figure.py, which is deliberately pinned to the older, narrower
contract and must not be reused for this domain.

Run:
    PYTHONPATH=$PWD:../vault-controller python vault/tools/make_odd_safeset_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap

from vault import config as C
from vault.model_release import get_model_release

REPO = Path(__file__).resolve().parents[2]
GRID = REPO / "vault/data/grid_reachavoid_odd.npz"
OUT = REPO / "vault/docs/odd_safeset_v2_2_pm4.pdf"
SAFE, UNSAFE = "#2e9e4f", "#d1443c"
CMAP = ListedColormap([UNSAFE, SAFE])
MUS = [("V_mu3", 0.3), ("V_mu6", 0.6), ("V_mu10", 1.0)]
L, W = 0.062, 0.878


def main() -> int:
    contract = json.loads(
        get_model_release().artifact_path("odd_contract").read_text())
    v_lo, v_hi = contract["odd"]["velocity"]["bounds"]
    psi = contract["odd"]["yaw_rate"]["bounds"][1]
    tier = contract["odd"]["velocity"]["tier"]

    z = np.load(GRID, allow_pickle=True)
    axes = [np.asarray(a, float) for a in z["axes"]]
    dims = [len(a) for a in axes]
    grids = {k: np.asarray(z[k], float).reshape(dims) for k, _ in MUS}
    v_ax, th_ax = axes[0], axes[1]
    i_thd, i_psd = int(np.argmin(abs(axes[2]))), int(np.argmin(abs(axes[3])))

    fig = plt.figure(figsize=(8.5, 6.6))
    fig.patch.set_facecolor("white")
    fig.text(L, 0.958, "Certified safe set on the active $\\pm$4 ODD contract",
             fontsize=13, fontweight="bold", va="top")
    fig.text(L, 0.921, f"v $\\in$ [{v_lo:g}, {v_hi:g}] m/s, $|\\dot\\psi| \\leq$ {psi:g} rad/s "
                       f"— read from the locked contract (velocity tier: {tier})",
             fontsize=9.3, color="#555555", va="top")

    stats = []
    for key, mu in MUS:
        V = grids[key]
        sl = V[:, :, i_thd, i_psd]
        row = sl[:, int(np.argmin(abs(th_ax)))]
        band = v_ax[row >= 0]
        stats.append((mu, 100 * (V >= 0).mean(), band.min(), band.max()))

    note = ("The velocity edges are the CONTRACT, not the dynamics. The four-state model has no "
            "speed-dependent term — pitch\nacceleration is identical at v = 0 through 10 m/s — so "
            "along v the value function is exactly the ODD margin ramp and the\ncertified set ends "
            "wherever the declared bound is placed. What the solve genuinely determines is the "
            "boundary in pitch,\npitch rate, and the $|v\\dot\\psi| \\leq a_{tip}$ roll coupling: "
            "those edges are dynamics. Widening v further would only move the wall.")
    fig.patches.append(mpatches.FancyBboxPatch(
        (L - 0.008, 0.735), W + 0.016, 0.118, transform=fig.transFigure,
        boxstyle="round,pad=0.006", facecolor="#fdf6ec", edgecolor="#e3d5bd", linewidth=0.9))
    fig.text(L, 0.845, note, fontsize=8.1, color="#1f2430", va="top", linespacing=1.6)

    fig.legend(handles=[mpatches.Patch(color=SAFE, label="certified safe  (V $\\geq$ 0)"),
                        mpatches.Patch(color=UNSAFE, label="not certified"),
                        plt.Line2D([0], [0], ls="--", color="#1a1a1a",
                                   label="declared ODD velocity bound")],
               loc="upper left", bbox_to_anchor=(L, 0.722), ncol=3, frameon=False, fontsize=8.4)

    gs = fig.add_gridspec(1, 3, left=0.075, right=0.96, top=0.60, bottom=0.135, wspace=0.30)
    dv, dt = (v_ax[1] - v_ax[0]) / 2, (th_ax[1] - th_ax[0]) / 2
    extent = [v_ax[0] - dv, v_ax[-1] + dv, th_ax[0] - dt, th_ax[-1] + dt]

    for col, ((key, mu), (_, frac, blo, bhi)) in enumerate(zip(MUS, stats)):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow((grids[key][:, :, i_thd, i_psd] >= 0).T.astype(float), origin="lower",
                  aspect="auto", extent=extent, cmap=CMAP, vmin=0, vmax=1,
                  interpolation="nearest")
        for edge in (v_lo, v_hi):
            ax.axvline(edge, ls="--", lw=1.1, color="#1a1a1a")
        ax.set_title(f"$\\mu$ = {mu}", fontsize=9.5, fontweight="bold", loc="left", pad=18)
        ax.text(0, 1.015, f"{frac:.1f}% of grid safe   ·   at rest, v $\\in$ "
                          f"[{blo:+.1f}, {bhi:+.1f}] m/s",
                transform=ax.transAxes, fontsize=7, color="#555555", va="bottom")
        ax.set_xlabel("v  (m/s)", fontsize=8)
        ax.set_ylabel(r"$\theta$  (rad)", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.text(L, 0.075,
             "Slices at $\\dot\\theta=\\dot\\psi=0$. Grid and contract both read from the pinned "
             "release, so figure and contract cannot drift.\nNot comparable with "
             "make_model_update_figure.py, which is pinned to the older narrower contract.  "
             "Reproduce: python vault/tools/make_odd_safeset_figure.py",
             fontsize=7.3, color="#666666", va="top", linespacing=1.5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f"wrote {OUT}")
    for mu, frac, blo, bhi in stats:
        print(f"  mu={mu}: {frac:.2f}% safe, at-rest v in [{blo:+.2f}, {bhi:+.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
