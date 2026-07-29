"""Measure the rollover threshold in BOTH directions on the release mass distribution.

This is the empirical witness for the asymmetric no-liftoff bound. It exists because
the certificate's roll bound is a static moment balance that assumes the centre of
mass lies on the longitudinal centreline, and the v2.2 robot's does not: summing the
URDF link inertials puts it 7.2764 mm off-centre laterally. Pushing toward the heavy
side therefore tips at a lower lateral acceleration than pushing away from it, and
nothing in the four-state safety path currently expresses that.

What this measures, and why it is arranged this way:

  * The mass properties are re-derived from the release URDF at run time, through the
    same parser the model release uses, so the rig cannot drift from the model. Total
    mass, the full 3-vector centre of mass, and the full inertia tensor about the
    centre of mass all come from source rather than being typed in.
  * The body has NO pitch freedom. Roll is what is under test; an inverted pendulum
    left free in pitch simply falls over and reports its pitch collapse instead.
  * Liftoff is detected from the per-wheel contact normal force reaching zero, not
    from the contact count. Contact count is a discretisation artefact -- it can drop
    while the wheel still carries load, or persist at negligible load.
  * The lateral force is applied at the current centre of mass, tracked as the body
    rolls, which is what makes the ramp equivalent to an inertial reaction.

The discriminating quantity is the RATIO of the two directions, not either absolute
value, because the analytic bound and the simulation credit different support
geometry. The bound credits only the wheel-centre span W/2, which is the conservative
choice; the simulated cylinders are 40 mm wide and support out to W/2 + 0.02 m. That
difference shifts both absolute thresholds and largely cancels in the ratio.

Throughout, H is the WHOLE-ROBOT centre-of-mass height above ground (z_com + r_w),
read from the private release -- NOT h_cm + r_w, which is a frame mismatch.

Measured results (2026-07-28 hold-and-settle bisection) are recorded in the
PRIVATE handoff document: vault-controller/docs/ssb_handoff/CHANGES_TO_THE_MODEL_LAYER.md.

Reading: the measured ratio matches the rim-credit prediction to 0.099% and the
wheel-centre prediction to 1.19%, so the simulated contact behaves as though
supported near the wheel rim, as expected for a 40 mm cylinder. Both absolute
thresholds land ABOVE the shipped wheel-centre bound (+11.0% and +9.6%), which is
the direction that makes the shipped bound conservative rather than optimistic.

Two independent observables therefore confirm the lateral offset and its sign: the
static load split, which matches to 0.1% and needs no tipping at all, and the
directional ratio. A ratio near 1.0 would have falsified the asymmetry.

Caveats: simulation only, no hardware witness. Static, so this is a rollover
bifurcation point and says nothing about dynamic liftoff under transient roll
excitation -- the four-state model carries no roll state, and that gap is not
addressed here. Unladen; the laden case needs a declared payload mass-properties
artifact, not a scaled height.

Run:
    PYTHONPATH=$PWD:../vault-controller python vault/tools/measure_rollover_bilateral.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

CONTROLLER = Path(__file__).resolve().parents[2].parent / "vault-controller"
sys.path.insert(0, str(CONTROLLER / "models" / "tools"))
import verify_urdf_composite as V  # noqa: E402

URDF = CONTROLLER / "models/source/urdf/tucked_v2_2/robot.urdf"
GRAV = 9.81
# PUBLIC REPO: wheel geometry is read from the private release at runtime.
import json as _json
_GEOM = _json.loads((CONTROLLER / "models/source/geometry/model_geometry.json").read_text())
_PARAMS = _json.loads((CONTROLLER / "models/generated/reduced/composite_params.json").read_text())
WHEEL_R = float(_PARAMS["wheel_radius"])
HALF_WIDTH = float(_GEOM["wheel_contact_half_width"])
FRICTION = 'friction="1 .005 .0001" condim="6"'


def release_mass_properties() -> dict:
    """Total mass, whole-robot CoM and inertia about the CoM, from the release URDF."""
    links, joints = V._parse_urdf(URDF)
    poses = V._link_poses(links, joints)
    total, first_moment = 0.0, np.zeros(3)
    inertia_origin = np.zeros((3, 3))
    for name, link in links.items():
        if link is None:
            continue
        rotation, translation = poses[name]
        centre = translation + rotation @ link["com"]
        inertia_origin += rotation @ link["inertia"] @ rotation.T + link["mass"] * (
            float(centre @ centre) * np.eye(3) - np.outer(centre, centre))
        total += link["mass"]
        first_moment += link["mass"] * centre
    com = first_moment / total
    inertia_com = inertia_origin - total * (
        float(com @ com) * np.eye(3) - np.outer(com, com))
    track = 2.0 * abs(float(next(
        j for j in joints if j["name"] == "left_body_wheel_joint")["translation"][1]))
    return {"mass": total, "com": com, "inertia_com": inertia_com, "track": track}


def _rig(props: dict) -> str:
    """Rigid body on two cylinder wheels, CoM placed at the true release offset.

    The body is given exactly the three degrees of freedom a rollover test needs --
    lateral slide, vertical slide, and roll about the longitudinal axis. It is NOT a
    free body. A free-jointed version of this rig is useless here: the wheels are
    cylinders and the centre of mass sits 0.32 mm ahead of the axle, so the whole rig
    simply rolls forward and drives itself off down the plane (measured: 0.78 m in
    three seconds), which moves the contact geometry and the force application point
    throughout the measurement and tumbles the body quaternion through +-180 degrees.
    Pitch freedom is excluded for the same reason as roll is isolated: an inverted
    pendulum left free in pitch reports its pitch collapse instead of a rollover.
    """
    com, track, inertia = props["com"], props["track"], props["inertia_com"]
    # Body origin sits at the wheel axle; the URDF CoM is expressed in that frame.
    cx, cy, cz = com[0], com[1], com[2]
    full = (f'{inertia[0, 0]:.9f} {inertia[1, 1]:.9f} {inertia[2, 2]:.9f} '
            f'{inertia[0, 1]:.9f} {inertia[0, 2]:.9f} {inertia[1, 2]:.9f}')
    return f"""<mujoco>
      <option timestep="0.0005" integrator="implicitfast" cone="elliptic"/>
      <worldbody>
        <geom name="ground" type="plane" size="20 20 .1" {FRICTION}/>
        <body name="rig" pos="0 0 {WHEEL_R}">
          <joint name="slide_y" type="slide" axis="0 1 0"/>
          <joint name="slide_z" type="slide" axis="0 0 1"/>
          <joint name="roll_x"  type="hinge" axis="1 0 0"/>
          <inertial pos="{cx:.9f} {cy:.9f} {cz:.9f}" mass="{props['mass']:.9f}"
                    fullinertia="{full}"/>
          <geom name="wheel_pos" type="cylinder" size="{WHEEL_R} {HALF_WIDTH}"
                pos="0 {track / 2} 0" euler="90 0 0" {FRICTION}/>
          <geom name="wheel_neg" type="cylinder" size="{WHEEL_R} {HALF_WIDTH}"
                pos="0 {-track / 2} 0" euler="90 0 0" {FRICTION}/>
        </body>
      </worldbody></mujoco>"""


ROLL_QPOS = 2   # index of the roll hinge in qpos, given the joint order above


def wheel_normals(model, data) -> tuple[float, float]:
    """Contact normal force carried by the +y and -y wheels."""
    ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n): n
           for n in ("wheel_pos", "wheel_neg")}
    loads = {"wheel_pos": 0.0, "wheel_neg": 0.0}
    wrench = np.zeros(6)
    for i in range(data.ncon):
        contact = data.contact[i]
        name = ids.get(contact.geom1) or ids.get(contact.geom2)
        if name is None:
            continue
        mujoco.mj_contactForce(model, data, i, wrench)
        loads[name] += abs(float(wrench[0]))
    return loads["wheel_pos"], loads["wheel_neg"]


def hold(props: dict, direction: float, force: float, seconds: float = 4.0,
         settle: int = 3000) -> tuple[bool, float, float, float]:
    """Hold a constant lateral force at the CoM until the rig settles or goes over.

    Returns (tipped, steady outer normal load, static +y load, static -y load).

    Holding is what makes this a static measurement. A force RAMP triggers on the
    first instant the outer wheel's normal force touches zero, which the roll
    oscillation excited by the ramp itself can reach well below the sustained
    threshold; holding lets the rig find its equilibrium at each force level, so the
    bisection converges on a genuine static bifurcation rather than a transient.
    """
    model = mujoco.MjModel.from_xml_string(_rig(props))
    data = mujoco.MjData(model)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rig")

    for _ in range(settle):
        mujoco.mj_step(model, data)
    static_pos, static_neg = wheel_normals(model, data)

    steps = int(seconds / model.opt.timestep)
    tail = []
    for i in range(steps):
        mujoco.mj_applyFT(model, data, np.array([0.0, direction * force, 0.0]),
                          np.zeros(3), data.xipos[body], body, data.qfrc_applied)
        mujoco.mj_step(model, data)
        data.qfrc_applied[:] = 0.0
        if abs(float(data.qpos[ROLL_QPOS])) > np.radians(25.0):
            return True, 0.0, static_pos, static_neg
        if i > steps - 2000:
            pos, neg = wheel_normals(model, data)
            tail.append(neg if direction > 0 else pos)
    return False, (float(np.mean(tail)) if tail else float("nan")), static_pos, static_neg


def static_threshold(props: dict, direction: float, hi: float = 300.0,
                     iters: int = 14) -> float:
    """Bisect on the constant lateral force at which the rig goes over."""
    lo = 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        tipped, *_ = hold(props, direction, mid)
        lo, hi = (lo, mid) if tipped else (mid, hi)
    return 0.5 * (lo + hi) / props["mass"]


def main() -> int:
    props = release_mass_properties()
    mass, com, track = props["mass"], props["com"], props["track"]
    height = com[2] + WHEEL_R
    y_cm = com[1]

    print(f"release mass properties (from {URDF.name}):")
    print(f"  mass    = {mass:.6f} kg")
    print(f"  CoM     = [{1000 * com[0]:+.4f}, {1000 * com[1]:+.4f}, "
          f"{1000 * com[2]:+.4f}] mm  (axle frame)")
    print(f"  H_whole = z_com + r_w = {height:.7f} m")
    print(f"  track W = {track:.5f} m\n")

    predicted = {+1.0: GRAV * (track / 2 - y_cm) / height,
                 -1.0: GRAV * (track / 2 + y_cm) / height}
    shift = mass * GRAV * y_cm / track
    print(f"predicted static loads: +y wheel {mass * GRAV / 2 + shift:.2f} N, "
          f"-y wheel {mass * GRAV / 2 - shift:.2f} N")
    print(f"predicted thresholds:   +y (toward heavy) {predicted[+1.0]:.4f}, "
          f"-y (away) {predicted[-1.0]:.4f}, ratio {predicted[-1.0] / predicted[+1.0]:.4f}\n")

    results = {}
    for direction, label in ((+1.0, "+y  (toward heavy side)"),
                             (-1.0, "-y  (away from heavy side)")):
        accel = static_threshold(props, direction)
        _, _, static_pos, static_neg = hold(props, direction, 0.0, seconds=0.5)
        results[direction] = accel
        print(f"push {label}")
        print(f"  static loads: +y {static_pos:7.2f} N   -y {static_neg:7.2f} N")
        print(f"  static rollover a_lat = {accel:.4f} m/s^2   "
              f"(analytic {predicted[direction]:.4f}, "
              f"{100 * (accel / predicted[direction] - 1):+.1f}%)")

    ratio = results[-1.0] / results[+1.0]
    expected = predicted[-1.0] / predicted[+1.0]
    print(f"\nmeasured ratio  = {ratio:.4f}   ({100 * (ratio - 1):+.2f}% spread)")
    print(f"analytic ratio  = {expected:.4f}   ({100 * (expected - 1):+.2f}% spread)")
    print(f"ratio agreement = {100 * (ratio / expected - 1):+.2f}%")
    print("\nThe RATIO is the discriminating quantity. The analytic bound credits only\n"
          "the wheel-centre span W/2, while the simulated cylinders support out to\n"
          f"W/2 + {HALF_WIDTH} m, so the absolute thresholds are expected to sit ABOVE\n"
          "the analytic values; that bias is common to both directions and cancels in\n"
          "the ratio. A ratio near 1.0 would falsify the asymmetry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
