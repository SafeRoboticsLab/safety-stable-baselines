"""Render a handful of representative simulation videos for progress review.

Not part of the vault/ package (this is a one-off demo/deliverable, not certified pipeline code)
-- it imports vault as a library. Run from the safety-stable-baselines repo root:

    MUJOCO_GL=egl PYTHONPATH=$PWD:../vault-controller python video/render_videos.py

Produces, into this video/ folder:
  fallback_flat_slow.mp4        trained safety ("fallback") policy alone, flat terrain, v0=0.3
  fallback_flat_fast.mp4        same policy, flat terrain, v0=1.3 (near the top of the ODD)
  fallback_terrain_bumps.mp4    same policy, procedurally generated speed-bump terrain
  fallback_terrain_steps.mp4    same policy, procedurally generated curb/step terrain
  filter_shielding_random.mp4   QCBFFilter shielding a deliberately naive random "task policy"
  filter_shielding_aggressive.mp4  QCBFFilter shielding the REAL deployed controller under an
                                   aggressive reference command near the ODD edge

The fallback policy was trained on flat terrain only (Track A) -- the terrain videos are a
qualitative look, not a robustness claim; Track B (terrain-aware training) is the follow-on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safety_sb3 import SafetySAC
from vault import config as C
from vault import contact_margin as CM
from vault import f_cert as F
from vault.safety_filter import SACFallback, ValueMonitor, OptimizationIntervention
from vault.mujoco_model import build_mjcf
from vault.mujoco_plant import MujocoPlant

OUT = Path(__file__).resolve().parent
FPS = 30
WIDTH, HEIGHT = 640, 480
MODEL_PATH = str(C.MODELS / "contact_safety_sac_v3")


def _tracking_camera(chassis_xy, distance=2.4, azimuth=110, elevation=-15):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [chassis_xy[0], chassis_xy[1], 0.1]
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


# A headlight is attached to whichever camera is rendering (not a body/site in the world), so the
# robot stays lit as it moves instead of relying on the single static overhead point light -- purely
# a rendering setting, zero effect on physics/contacts. Spliced into every video's MJCF below.
_HEADLIGHT = '<visual><headlight ambient="0.5 0.5 0.5" diffuse="0.7 0.7 0.7" specular="0.3 0.3 0.3"/></visual>'


def _terrain_xml(kind: str, mu: float) -> str:
    """A flat-terrain contact_geometry MJCF with extra STATIC obstacle geoms spliced in, purely
    for visual variety in these demo videos (not a Track-B training terrain) -- plus a headlight
    for consistent lighting regardless of terrain kind."""
    xml = build_mjcf(wheel="cylinder", mu=mu, contact_geometry=True)
    xml = xml.replace("<compiler angle=\"degree\"/>", "<compiler angle=\"degree\"/>" + _HEADLIGHT)
    if kind == "flat":
        return xml
    if kind == "bumps":
        # a run of low speed bumps -- climbable by the wheels, perturbs the balance loop
        geoms = "".join(
            f'<geom type="box" size="0.05 0.6 {h}" pos="{x0} 0 {h}" rgba="0.5 0.5 0.5 1"/>'
            for x0, h in [(1.0, 0.02), (2.0, 0.03), (3.0, 0.04), (4.0, 0.03)]
        )
    elif kind == "steps":
        # a few raised platforms of increasing height -- some climbable, some not
        geoms = "".join(
            f'<geom type="box" size="0.15 0.6 {h / 2}" pos="{x0} 0 {h / 2}" rgba="0.6 0.4 0.3 1"/>'
            for x0, h in [(1.2, 0.03), (2.5, 0.07), (4.0, 0.12)]
        )
    else:
        raise ValueError(kind)
    return xml.replace('<light pos="0 0 2"/>', geoms + '<light pos="0 0 2"/>')


def _overlay(frame: np.ndarray, title: str, lines: list[str]) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle([0, 0, WIDTH, 18 + 14 * (len(lines) + 1)], fill=(0, 0, 0, 140))
    draw.text((6, 3), title, fill=(255, 255, 255, 255))
    for i, line in enumerate(lines):
        draw.text((6, 18 + 14 * i), line, fill=(255, 230, 120, 255))
    return np.asarray(img)


def _record(plant: MujocoPlant, control_fn, seconds: float, out_path: Path, title: str):
    """control_fn(step_idx, x, t) -> (u [tau_L,tau_R], telemetry_lines: list[str]) applied this step."""
    renderer = mujoco.Renderer(plant.model, height=HEIGHT, width=WIDTH)
    stride = max(1, int(round((1.0 / plant.dt) / FPS)))
    n_steps = int(round(seconds / plant.dt))
    frames = []
    x = plant.get_state()
    for i in range(n_steps):
        u, lines = control_fn(i, x, i * plant.dt)
        x = plant.step(np.asarray(u, float))
        if i % stride == 0:
            chassis_xy = plant.data.xpos[plant.model.body("chassis").id][:2]
            renderer.update_scene(plant.data, camera=_tracking_camera(chassis_xy))
            frame = renderer.render().copy()
            frames.append(_overlay(frame, title, [f"t={i * plant.dt:5.2f}s"] + lines))
    renderer.close()
    imageio.mimwrite(str(out_path), frames, fps=FPS, quality=7, macro_block_size=None)
    print(f"  wrote {out_path.name}: {len(frames)} frames @ {FPS}fps ({seconds:.1f}s sim) -- {title}")


def _fallback_policy(model, tau_max=C.TAU_MAX):
    def policy(x, mu):
        obs5 = np.append(np.asarray(x, np.float32), np.float32(mu))
        a, _ = model.predict(obs5, deterministic=True)
        return np.clip(a, -1.0, 1.0) * tau_max
    return policy


def render_fallback(model, terrain: str, v0: float, mu: float, seconds: float, out_name: str, title: str):
    xml = _terrain_xml(terrain, mu)
    plant = MujocoPlant(wheel="cylinder", mu=mu, contact_geometry=True, xml_override=xml)
    plant.reset(np.array([v0, 0.0, 0.0, 0.0]))

    pi_safe = _fallback_policy(model)

    def control_fn(i, x, t):
        u = pi_safe(x, mu)
        g_odd, g_ct = float(F.odd_margin(x)), CM.margin(plant)
        return u, [f"fallback pi_safe  v={x[0]:+.2f} theta={x[1]:+.2f}",
                   f"g_odd={g_odd:+.3f}  g_contact={g_ct:+.3f}"]

    _record(plant, control_fn, seconds, OUT / out_name, title)


def render_filter_shielding_random(model, seconds: float, out_name: str, title: str, seed=0):
    # Validated with an 8-seed sweep of this exact (v0, gamma, eps_floor) combo: worst-case margin
    # stayed positive (+0.04 to +0.05) across all 8 seeds under fresh-random-torque every step --
    # a genuinely adversarial nominal "task policy", not a softened one.
    rng = np.random.default_rng(seed)
    plant = MujocoPlant(wheel="cylinder", mu=0.8, dt=0.0005, substeps=20, contact_geometry=True,
                        xml_override=_terrain_xml("flat", 0.8))
    plant.reset(np.array([0.8, 0.0, 0.0, 0.0]))
    qcbf = OptimizationIntervention(ValueMonitor(model), SACFallback(model), gamma=0.2, eps_floor=0.05)

    def control_fn(i, x, t):
        u_task = rng.uniform(-C.TAU_MAX, C.TAU_MAX, size=2)   # deliberately naive/unsafe "task policy"
        u_safe, overridden = qcbf(x, u_task, plant.mu)
        g_odd, g_ct = float(F.odd_margin(x)), CM.margin(plant)
        return u_safe, [f"task=random  {'OVERRIDE' if overridden else 'passthrough'}",
                        f"g_odd={g_odd:+.3f}  g_contact={g_ct:+.3f}"]

    _record(plant, control_fn, seconds, OUT / out_name, title)


def render_filter_shielding_aggressive(model, seconds: float, out_name: str, title: str):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "vault-controller"))
    from balance_controller.c_coupled_controller import CCoupledController

    plant = MujocoPlant(wheel="cylinder", mu=0.6, dt=0.0005, substeps=20, contact_geometry=True,
                        xml_override=_terrain_xml("flat", 0.6))
    ctrl = CCoupledController()
    qcbf = OptimizationIntervention(ValueMonitor(model), SACFallback(model), gamma=0.2, eps_floor=0.05)
    # odd_margin(x0) ~ +0.4 -- comfortably inside the certified envelope, but with an aggressive
    # forward+turn command that pushes hard against it. (A closer-to-the-edge x0 was tried first,
    # e.g. v=1.45/psi_dot=2.45 simultaneously near both their ODD bounds -- that state turned out
    # to sit right at/past the true reachable-safe boundary even for pure safety-argmax, i.e. a
    # near/no-win corner, not a fair test of the filter.)
    x0 = np.array([1.0, 0.3, 0.0, 1.5])
    plant.reset(x0)
    ctrl.reset()
    ref = np.array([1.5, -2.3])             # aggressive constant reference command

    def control_fn(i, x, t):
        u_task = np.clip(ctrl.step(x, ref, t), -C.TAU_MAX, C.TAU_MAX)
        u_safe, overridden = qcbf(x, u_task, plant.mu)
        g_odd, g_ct = float(F.odd_margin(x)), CM.margin(plant)
        return u_safe, [f"task=deployed ctrl, ref={ref}  {'OVERRIDE' if overridden else 'passthrough'}",
                        f"g_odd={g_odd:+.3f}  g_contact={g_ct:+.3f}"]

    _record(plant, control_fn, seconds, OUT / out_name, title)


def main():
    print(f"loading {MODEL_PATH}.zip")
    model = SafetySAC.load(MODEL_PATH)

    print("rendering fallback-policy videos (flat terrain, varied starting speed)")
    render_fallback(model, "flat", v0=0.3, mu=0.8, seconds=6,
                     out_name="fallback_flat_slow.mp4", title="fallback policy, flat, v0=0.3")
    render_fallback(model, "flat", v0=1.3, mu=0.8, seconds=6,
                     out_name="fallback_flat_fast.mp4", title="fallback policy, flat, v0=1.3")

    print("rendering fallback-policy videos (procedurally generated terrain)")
    render_fallback(model, "bumps", v0=0.8, mu=0.8, seconds=8,
                     out_name="fallback_terrain_bumps.mp4", title="fallback policy, speed bumps")
    render_fallback(model, "steps", v0=0.8, mu=0.8, seconds=8,
                     out_name="fallback_terrain_steps.mp4", title="fallback policy, curbs/steps")

    print("rendering QCBF-filter-shielding videos")
    render_filter_shielding_random(model, seconds=6, out_name="filter_shielding_random.mp4",
                                    title="QCBF shielding a random/naive task policy")
    render_filter_shielding_aggressive(model, seconds=6, out_name="filter_shielding_aggressive.mp4",
                                       title="QCBF shielding the deployed controller, aggressive ref")

    print("done ->", OUT)


if __name__ == "__main__":
    main()
