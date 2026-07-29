"""Figure: v2.2 model-layer results — avoid-only certificate, reach-avoid rollouts,
and the (v, psi_dot) turn-authority plane.

Companion to docs/CHANGES_TO_THE_MODEL_LAYER.md sections 1-13. Reads the committed
evaluation slice (data/reach_avoid_slice_v22_evalonly.json) and the release grids
from the pinned vault-controller checkout, so the figure cannot drift from the
artifacts it summarises.

The reach-avoid panel plots MuJoCo rollouts of the EVALUATION-ONLY checkpoint
(manifest entry reach_avoid_safety_sac_v22*): 300k steps, not converged, critic
still carrying the inherited entropy term (section 11). The avoid-only panels are
the reduced 4-state grid certificate. The two are different objects on different
plants -- the caveat blocks on both pages say so.

Run:
    VAULT_CONTROLLER_ROOT=~/Documents/vault-controller \
        python vault/tools/make_v22_results_figure.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap

REPO = Path(__file__).resolve().parents[2]
SLICE = REPO / "vault/data/reach_avoid_slice_v22_evalonly.json"
OUT = REPO / "vault/docs/v22_model_layer_and_reach_avoid_results.pdf"
GRIDS = Path(os.environ["VAULT_CONTROLLER_ROOT"]) / "models/generated/safety/grids"

FAIL, LIMBO, REACH = "#d1443c", "#e9b949", "#2e9e4f"
SAFE, UNSAFE = "#2e9e4f", "#d1443c"
CMAP2 = ListedColormap([UNSAFE, SAFE])
L = 0.055

MODEL_NOTE = (
    "MODEL LAYER (what changed under the math): a_tip 6.998922 → 6.924595 (the old value used h_cm + wheel_radius — the SPRUNG body's\n"
    "axle height mixed with the WHOLE robot's ground offset; corrected height 0.188559, not 0.196756, and the 7.276 mm lateral CoM offset\n"
    "is now carried). a_tip_laden 6.32652 → 5.326611, read from the laden grid's own metadata. ODD velocity ±4 → ±6 by operator ruling\n"
    "(monotone in g: everything previously certified stays certified). Speed governor now evaluates the SIGNED state — the |v|,|ψ̇| fold\n"
    "assumed a symmetry the plant does not have and erred both ways (316 permissive / 510 conservative at reserve 0.35, 40k states/μ)."
)
CAVEATS = (
    "READ BEFORE COMPARING WITH THE ORIGINAL 3-PANEL FIGURE — the domains differ. Your panels ran v ∈ [−0.5, 1.5]; this is the full ±6 ODD\n"
    "(6× wider), and the checkpoint is 300k steps and NOT converged: 17.3% limbo is the signature of an undertrained reach policy (your\n"
    "converged arms collapse limbo to 0%). Authorized for EVALUATION ONLY — the critic still carries the inherited max-entropy term inside\n"
    "the DRABE backup (§11; your ISAACS code already excludes it: base_block.py:406). Measured on this run: α auto-tuned to 1.4e-4, β < 0 in\n"
    "99.3% of states — contamination ran conservative here, by sign luck, not by guarantee.\n"
    "\n"
    "LEFT PANEL LIMITS (§10): min_d over disturbance-box corners is optimistic (3/1,161,508 states left the set under denser sampling; an\n"
    "exact fix was attempted and REVERTED — the bad-mask discontinuity defeats the affine-per-cell argument). Early-stopping residual is\n"
    "UNBOUNDED (iters/decade degrade 63 → 3,153; no geometric tail exists — your non-contraction result, measured). Node-exhaustive is not a\n"
    "continuum proof. Roll constraint is quasi-static on a roll-free model; the validation plant drops the 7.276 mm lateral offset (§13).\n"
    "\n"
    "PLANTS DIFFER BY DESIGN: the left panel is the reduced 4-state f_cert (opt6 kernel — the same kernel the iLQR linearizes); the right\n"
    "panel is full MuJoCo contact physics. Green-left = \"the reduced model certifies it\"; green-right = \"one MuJoCo rollout reached the\n"
    "target\". VERIFICATION: 28 release gates, 9/9 bounds, frame hygiene both trees, 15/15 + 124 controller, 18/18 SSB, 152 toolchain — all\n"
    "green on quiescent trees. Preconditions satisfied BY CONSTRUCTION (§12.1): finite control set, V₀ = g. Warm-starting VI destroys both."
)
TURN_NOTE = (
    "TURN AUTHORITY — the (v, ψ̇) plane at θ = θ̇ = 0 answers \"how hard can it turn at speed\". The dotted curves are the ANALYTIC roll\n"
    "hyperbola |ψ̇| = a_tip/|v| from the corrected bound; the certified edge should hug it wherever roll (not friction) binds. Checked\n"
    "exhaustively: the certified yaw edge never exceeds the hyperbola at any of 606 speed-slices across both grids — at μ ≥ 0.6 it sits\n"
    "exactly on the last grid node beneath it. At low μ, friction binds first at low speed and roll takes over by v ≈ 3 m/s. Laden: the\n"
    "5.327 bound costs turn authority (−15% at v = 2, μ=0.3), NOT straight-line speed — a_tip bounds the PRODUCT v·ψ̇, so tightening it\n"
    "squeezes turning. Any laden derate belongs on the yaw channel, not the speed channel. Dashed lines: ±6 ODD (v), ±4 ODD (ψ̇)."
)


def vpsi_panel(fig, gpos, z, ax4, key, a_tip, tag, sub):
    dims = [len(a) for a in ax4]
    V = np.asarray(z[key], float).reshape(dims)
    ith = int(np.argmin(abs(ax4[1])))
    ithd = int(np.argmin(abs(ax4[2])))
    mask = V[:, ith, ithd, :] >= 0
    axp = fig.add_subplot(gpos)
    dv = (ax4[0][1] - ax4[0][0]) / 2
    dp = (ax4[3][1] - ax4[3][0]) / 2
    axp.imshow(mask.T.astype(float), origin="lower", aspect="auto",
               extent=[ax4[0][0] - dv, ax4[0][-1] + dv, ax4[3][0] - dp, ax4[3][-1] + dp],
               cmap=CMAP2, vmin=0, vmax=1, interpolation="nearest")
    vv = np.linspace(0.3, ax4[0][-1], 400)
    for s in (1, -1):
        axp.plot(vv, s * a_tip / vv, color="#1a1a1a", lw=1.3, ls=":")
        axp.plot(-vv, s * a_tip / vv, color="#1a1a1a", lw=1.3, ls=":")
    axp.set_ylim(ax4[3][0] - dp, ax4[3][-1] + dp)
    for e in (-6.0, 6.0):
        axp.axvline(e, ls="--", lw=1.0, color="#1a1a1a")
    for e in (-4.0, 4.0):
        axp.axhline(e, ls="--", lw=1.0, color="#666666")
    axp.set_title(tag, fontsize=9.5, fontweight="bold", loc="left", pad=15)
    axp.text(0, 1.016, sub, transform=axp.transAxes, fontsize=6.8, color="#555555", va="bottom")
    axp.set_xlabel("v (m/s)", fontsize=8.5)
    axp.set_ylabel("ψ̇ (rad/s)", fontsize=8.5)
    axp.tick_params(labelsize=7.5)
    outs = []
    for probe in (2.0, 4.0, 6.0):
        j = int(np.argmin(abs(ax4[0] - probe)))
        adm = ax4[3][mask[j]]
        outs.append(f"{probe:g}: {abs(adm).max():.2f}" if adm.size else f"{probe:g}: —")
    axp.text(0, -0.24, "max |ψ̇| at v = " + " · ".join(outs) + "  rad/s",
             transform=axp.transAxes, fontsize=7, color="#333333", va="top")


def main() -> int:
    d = json.loads(SLICE.read_text())
    vg, tg = np.array(d["v_grid"]), np.array(d["theta_grid"])
    code = {"reached": 2, "safe_no_reach": 1, "failed": 0}
    M = np.array([[code[c] for c in row] for row in d["grids"]["reach_avoid"]])

    zu = np.load(GRIDS / "robust_odd_dh0_dm0_bump0.npz", allow_pickle=True)
    zl = np.load(GRIDS / "robust_odd_dh30_dm20_bump0.npz", allow_pickle=True)
    axu = [np.asarray(a, float) for a in zu["axes"]]
    axl = [np.asarray(a, float) for a in zl["axes"]]
    at_u = float(zu["roll_a_tip_m_s2"])
    at_l = float(zl["roll_a_tip_m_s2"])
    Vu10 = np.asarray(zu["V_1.0"], float).reshape([len(a) for a in axu])
    A = Vu10[:, :, int(np.argmin(abs(axu[2]))), int(np.argmin(abs(axu[3])))] >= 0
    fr, fl, ff = 100 * np.mean(M == 2), 100 * np.mean(M == 1), 100 * np.mean(M == 0)

    with PdfPages(OUT) as pdf:
        # ---- page 1 ----
        fig = plt.figure(figsize=(11, 8.6))
        fig.patch.set_facecolor("white")
        fig.text(L, 0.968, "v2.2 model layer — corrected bounds, avoid-only baseline, and reach-avoid on the corrected model",
                 fontsize=13, fontweight="bold", va="top")
        fig.text(L, 0.937, "2026-07-29 · all quantities from hash-locked release artifacts · "
                           "companion to CHANGES_TO_THE_MODEL_LAYER.md §§1–13 · page 1/2",
                 fontsize=8.8, color="#555555", va="top")
        fig.patches.append(mpatches.FancyBboxPatch((L - 0.008, 0.808), 0.9, 0.108, transform=fig.transFigure,
                           boxstyle="round,pad=0.006", facecolor="#fdf6ec", edgecolor="#e3d5bd", linewidth=0.9))
        fig.text(L, 0.908, MODEL_NOTE, fontsize=7.6, color="#1f2430", va="top", linespacing=1.55)
        gs = fig.add_gridspec(1, 2, left=0.06, right=0.965, top=0.765, bottom=0.46, wspace=0.22)

        axp = fig.add_subplot(gs[0, 0])
        dv = (axu[0][1] - axu[0][0]) / 2
        dt = (axu[1][1] - axu[1][0]) / 2
        axp.imshow(A.T.astype(float), origin="lower", aspect="auto",
                   extent=[axu[0][0] - dv, axu[0][-1] + dv, axu[1][0] - dt, axu[1][-1] + dt],
                   cmap=CMAP2, vmin=0, vmax=1, interpolation="nearest")
        for e in (-6.0, 6.0):
            axp.axvline(e, ls="--", lw=1.1, color="#1a1a1a")
        axp.set_title("Avoid-only GRID certificate (μ=1.0, θ̇=ψ̇=0)",
                      fontsize=10, fontweight="bold", loc="left", pad=14)
        axp.text(0, 1.014, "reduced 4-state f_cert · 1,161,508 nodes · corrected a_tip = 6.9246 · window ±7 (dashed = ±6 ODD)",
                 transform=axp.transAxes, fontsize=7, color="#555555", va="bottom")
        axp.set_xlabel("v (m/s)", fontsize=8.5)
        axp.set_ylabel("θ (rad)", fontsize=8.5)
        axp.tick_params(labelsize=7.5)

        axp = fig.add_subplot(gs[0, 1])
        dv = (vg[1] - vg[0]) / 2
        dt = (tg[1] - tg[0]) / 2
        axp.imshow(M.T, origin="lower", aspect="auto",
                   extent=[vg[0] - dv, vg[-1] + dv, tg[0] - dt, tg[-1] + dt],
                   cmap=ListedColormap([FAIL, LIMBO, REACH]), vmin=0, vmax=2, interpolation="nearest")
        axp.set_title("REACH-AVOID rollouts on the corrected v2.2 model",
                      fontsize=10, fontweight="bold", loc="left", pad=14)
        axp.text(0, 1.014, "MuJoCo contact physics · 300k steps, scratch_raw_sat · μ=0.8, 3 s horizon, 41×41",
                 transform=axp.transAxes, fontsize=7, color="#555555", va="bottom")
        axp.set_xlabel("v (m/s)", fontsize=8.5)
        axp.set_ylabel("θ (rad)", fontsize=8.5)
        axp.tick_params(labelsize=7.5)

        fig.legend(handles=[mpatches.Patch(color=REACH, label=f"reached ({fr:.1f}%)"),
                            mpatches.Patch(color=LIMBO, label=f"limbo — safe, no reach ({fl:.1f}%)"),
                            mpatches.Patch(color=FAIL, label=f"failed ({ff:.1f}%)"),
                            mpatches.Patch(color=SAFE, label="certified safe (left)")],
                   loc="center", bbox_to_anchor=(0.5, 0.402), ncol=4, frameon=False, fontsize=7.8,
                   columnspacing=1.4, handlelength=1.4, handletextpad=0.5)
        fig.patches.append(mpatches.FancyBboxPatch((L - 0.008, 0.035), 0.9, 0.305, transform=fig.transFigure,
                           boxstyle="round,pad=0.006", facecolor="#f2f4f8", edgecolor="#c9d2e0", linewidth=0.9))
        fig.text(L, 0.328, CAVEATS, fontsize=7.3, color="#1f2430", va="top", linespacing=1.5)
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 2 ----
        fig = plt.figure(figsize=(11, 8.6))
        fig.patch.set_facecolor("white")
        fig.text(L, 0.968, "Forward velocity vs yaw velocity — certified turn authority and the roll hyperbola",
                 fontsize=13, fontweight="bold", va="top")
        fig.text(L, 0.937, "avoid-only capability grids, corrected a_tip · θ = θ̇ = 0 slices · page 2/2",
                 fontsize=8.8, color="#555555", va="top")
        fig.patches.append(mpatches.FancyBboxPatch((L - 0.008, 0.775), 0.9, 0.142, transform=fig.transFigure,
                           boxstyle="round,pad=0.006", facecolor="#fdf6ec", edgecolor="#e3d5bd", linewidth=0.9))
        fig.text(L, 0.910, TURN_NOTE, fontsize=7.5, color="#1f2430", va="top", linespacing=1.55)
        gs = fig.add_gridspec(2, 2, left=0.06, right=0.965, top=0.73, bottom=0.09, wspace=0.22, hspace=0.52)
        vpsi_panel(fig, gs[0, 0], zu, axu, "V_0.3", at_u, "UNLADEN μ=0.3 (ice)",
                   f"a_tip = {at_u:.3f} · friction binds at low v, roll by v ≈ 3")
        vpsi_panel(fig, gs[0, 1], zu, axu, "V_1.0", at_u, "UNLADEN μ=1.0 (grip)",
                   f"a_tip = {at_u:.3f} · edge on the hyperbola at all speeds")
        vpsi_panel(fig, gs[1, 0], zl, axl, "V_0.3", at_l, "LADEN μ=0.3 (ice, dh=0.30 dm=0.20)",
                   f"a_tip = {at_l:.3f} · tightest configuration in the release")
        vpsi_panel(fig, gs[1, 1], zl, axl, "V_1.0", at_l, "LADEN μ=1.0 (grip)",
                   f"a_tip = {at_l:.3f} · roll-bound everywhere")
        pdf.savefig(fig)
        plt.close(fig)

    print(f"wrote {OUT}")
    print(f"  reach {fr:.1f}%  limbo {fl:.1f}%  failed {ff:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
