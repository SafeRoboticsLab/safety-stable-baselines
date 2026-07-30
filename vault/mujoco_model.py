"""Generate an MJCF for the wheeled inverted pendulum from the single-source params.

The robot is a 2-wheel balancing robot (the 4-state model's physical embodiment):
  - chassis: the sprung body (mass m_b, CoM at h_cm above the wheel axle, pitch inertia
    I_pitch about CoM, yaw inertia I_yaw). Pitches about the axle (y).
  - 2 wheels: hinge joints about the axle (y), at y=∓d, each mass m_wheel + spin inertia
    I_wheel. Geometry = SDF torus (radius1=wheel_radius, radius2=tube) or native cylinder
    (mesh tori inject ~mm rolling ripple; SDF torus ~5µm, cylinder smooth but flat-rim).
  - ground plane with friction mu; 2 wheel-torque motors (u = [tau_L, tau_R]).

State/sign conventions match f_cert: v fwd along +x, theta = pitch about +y (theta>0
nose-down/forward), psi about +z; R = Rz(psi)*Ry(theta), roll ~= 0 nominal.

Masses, inertias, and actuator limits come from the provenance-pinned
vault-controller release. The chassis yaw inertia subtracts the wheels'
parallel-axis contribution so the REASSEMBLED whole-robot yaw inertia equals
the composite value (no double-counting). Roll inertia is not a 4-state param
(roll is a wheel constraint), so it is reconstructed to a physically-valid
value; it does not enter the certified planar dynamics. Run this module to
print the assembly invariants.
"""
from __future__ import annotations

from pathlib import Path

from . import config as C
from .model_release import get_model_release


def load_params() -> dict:
    """Load structural parameters plus locked controller limits."""
    release = get_model_release()
    params = release.load_json("composite_params")
    params.update(release.load_json("controller_limits"))
    params.update(release.load_json("model_geometry"))
    return params


#  Non-wheel contact-geometry approximation (contact_geometry=True) --------------------
#  Robert's reduced 4-state model has no full-robot geometry (by design). The
#  articulated stand-in below is a COARSE VISUAL/CONTACT approximation of the
#  tucked legs: a single rigid capsule per side, outboard of the wheel, spanning
#  from axle height to just short of the ground at upright. The real linkage
#  geometry lives in the PRIVATE release URDF (vault-controller sibling); only a
#  dimensionless placement ratio is used here, and it is an approximation, not a
#  released dimension.
_LEG_Y_RATIO = 1.91                        # approx hip/wheel lateral placement (dimensionless, visual)
_LEG_RADIUS = 0.025                        # m, slender capsule (real leg links are thin)
# Ground clearance: C.LEG_GROUND_CLEARANCE (config.py) -- shared with contact_margin.py, which
# normalizes by the SAME constant so a nominal upright state's margin is ~1.0, not ~0.02 m.


def _visual_mesh_overlay() -> dict | None:
    """Release-mesh visual overlay, resolved from the PRIVATE sibling at runtime.

    Returns None (silently) when vault-controller is not checked out or the assets
    are missing -- the plant then renders its primitive geoms exactly as before.
    Everything here is read at runtime: mesh paths, link names, and placements come
    from the private articulated URDF, so this PUBLIC file carries no dimensions.

    ALL mesh-bearing links are attached, not just chassis+wheels: leg linkages,
    knee wheels and feet are posed by forward kinematics at the TUCKED display
    configuration (knees at 180 deg, all else zero) and fixed to the chassis body, since the reduced plant has no leg
    joints. The two body-wheel meshes attach to the spinning wheel bodies.

    The overlay is VISUAL ONLY: contype=0 conaffinity=0, display group 2; every
    body keeps its explicit <inertial>. Dynamics equality is pinned by
    test_visual_meshes_do_not_change_dynamics.
    """
    import os
    import xml.etree.ElementTree as ET

    import numpy as np

    root = Path(
        os.environ.get("VAULT_CONTROLLER_ROOT")
        or Path(__file__).resolve().parents[2] / "vault-controller"
    )
    urdf = root / "models/source/urdf/articulated_v2_2/robot.urdf"
    meshdir = root / "models/assets/meshes_decimated"   # highres exceeds MuJoCo's decoder
    if not (urdf.is_file() and meshdir.is_dir()):
        return None
    try:
        tree = ET.parse(urdf).getroot()
    except ET.ParseError:
        return None

    def rot(rpy):
        r, p_, y = rpy
        cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p_), np.sin(p_), np.cos(y), np.sin(y)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def vec(s, n=3):
        return np.array([float(x) for x in (s or " ".join(["0"] * n)).split()])

    # Tucked display configuration: knees at 180 deg (operator-provided joint
    # constant, matching the retract lane's knee = 180deg hold). All other joints
    # at zero. This is a VISUAL pose only.
    TUCKED_Q = {"right_knee_joint": np.pi, "left_knee_joint": np.pi}

    def axis_rot(axis, angle):
        a = axis / (np.linalg.norm(axis) or 1.0)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    # joint tree: child link -> (parent link, R, t) at the DISPLAY configuration
    parent_of: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}
    for j in tree.iter("joint"):
        o = j.find("origin")
        R_origin = rot(vec(o.get("rpy") if o is not None else None))
        angle = TUCKED_Q.get(j.get("name"), 0.0)
        if angle:
            ax = j.find("axis")
            R_origin = R_origin @ axis_rot(
                vec(ax.get("xyz") if ax is not None else "1 0 0"), angle)
        parent_of[j.find("child").get("link")] = (
            j.find("parent").get("link"),
            R_origin,
            vec(o.get("xyz") if o is not None else None),
        )

    def fk(link):
        """Pose of link frame in base_link frame at zero configuration."""
        R, t = np.eye(3), np.zeros(3)
        chain = []
        while link in parent_of:
            chain.append(parent_of[link])
            link = parent_of[link][0]
        if link != "base_link":
            return None
        for _, Rj, tj in reversed(chain):
            t = R @ tj + t
            R = R @ Rj
        return R, t

    def quat(R):
        """Rotation matrix -> MJCF quaternion, valid for ALL rotations.

        The naive w = sqrt(1+trace)/2 form divides by w, which is EXACTLY ZERO for
        any rotation by pi -- i.e. for the tucked knee fold itself. An earlier
        version short-circuited that case to the identity, which discarded the
        fold's orientation while keeping its translation: every pi-rotated link
        rendered unrotated at the correct position (feet looked right, lower legs
        pointed the wrong way, and the kinematic chain appeared broken). Use the
        standard largest-component branch instead, which is well-conditioned
        everywhere.
        """
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0:
            s = 2 * np.sqrt(1 + tr)
            w, x, y, z = s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
            s = 2 * np.sqrt(max(1e-12, 1 + R[0, 0] - R[1, 1] - R[2, 2]))
            w, x, y, z = (R[2, 1] - R[1, 2]) / s, s / 4, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] >= R[2, 2]:
            s = 2 * np.sqrt(max(1e-12, 1 + R[1, 1] - R[0, 0] - R[2, 2]))
            w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, s / 4, (R[1, 2] + R[2, 1]) / s
        else:
            s = 2 * np.sqrt(max(1e-12, 1 + R[2, 2] - R[0, 0] - R[1, 1]))
            w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4
        return f"{w:.8f} {x:.8f} {y:.8f} {z:.8f}"

    WHEELS = {"right_body_wheel_link": "right_wheel", "left_body_wheel_link": "left_wheel"}
    assets, chassis_geoms, wheel_geoms = [], [], {"right_wheel": "", "left_wheel": ""}
    for link in tree.iter("link"):
        name = link.get("name")
        vis = link.find("visual")
        mesh = vis.find("geometry/mesh") if vis is not None else None
        if mesh is None:
            continue
        stl = meshdir / Path(mesh.get("filename", "")).name
        if not stl.is_file():
            return None
        o = vis.find("origin")
        Rv = rot(vec(o.get("rpy") if o is not None else None))
        tv = vec(o.get("xyz") if o is not None else None)
        mid = f"vis_{name}"
        assets.append(f'<mesh name="{mid}" file="{stl}"/>')
        if name in WHEELS:
            # wheel link frame == wheel body frame (joint at the axle)
            wheel_geoms[WHEELS[name]] = (
                f'<geom type="mesh" mesh="{mid}" pos="{tv[0]} {tv[1]} {tv[2]}" '
                f'quat="{quat(Rv)}" contype="0" conaffinity="0" group="2" '
                f'rgba="0.25 0.25 0.27 1"/>')
            continue
        pose = fk(name)
        if pose is None:
            return None
        Rl, tl = pose
        Rg, tg = Rl @ Rv, Rl @ tv + tl
        rgba = "0.88 0.88 0.90 1" if name == "base_link" else "0.72 0.74 0.78 1"
        chassis_geoms.append(
            f'<geom type="mesh" mesh="{mid}" pos="{tg[0]:.8f} {tg[1]:.8f} {tg[2]:.8f}" '
            f'quat="{quat(Rg)}" contype="0" conaffinity="0" group="2" rgba="{rgba}"/>')

    if not chassis_geoms or not all(wheel_geoms.values()):
        return None
    return {
        "asset": "<asset>" + "".join(assets) + "</asset>",
        "chassis": "".join(chassis_geoms),
        "right_wheel": wheel_geoms["right_wheel"],
        "left_wheel": wheel_geoms["left_wheel"],
    }


def build_mjcf(wheel: str = "torus", mu: float = 1.0,
               params: dict | None = None, solref: tuple = (0.02, 1.0), margin: float = 0.0,
               imu_pos: tuple | None = None, imu_quat: tuple | None = None,
               contact_geometry: bool = False, visual_meshes: str = "auto") -> str:
    """Return an MJCF string. wheel in {'torus','cylinder'}; mu = wheel-ground friction.

    solref = (timeconst, dampratio): contact compliance. Default (0.02, 1.0) is MuJoCo's stiff
    critically-damped contact; larger timeconst = softer (real-tire-like, prevents rigid-point
    lift-off flicker). margin (m): contact activates early (a soft cushion), another anti-flicker
    lever. imu_pos/imu_quat (optional): add an IMU site (+ accel/gyro sensors) at a chassis-frame
    mount offset for estimator-in-the-loop tests; visual/sensor only, no dynamics change.

    contact_geometry (default False, backward compatible): give the chassis real collision
    (instead of the visual-only box) and add a leg capsule per side, so bodies other than the
    wheels can actually contact the terrain. Only the wheels remain "permitted contact" geometry.
    Used by the contact/slam-margin training path (mujoco_env.py); the plain balance-controller
    cross-validation path (mujoco_plant.py's default use) is unaffected.
    """
    p = params or load_params()
    R = p["wheel_radius"]
    wheel_contact_half_width = p["wheel_contact_half_width"]
    d = p["wheel_sep"] / 2.0
    m_b, h_cm = p["m_b"], p["h_cm"]
    I_pitch, I_yaw = p["I_pitch"], p["I_yaw"]
    m_w, I_w = p["m_wheel"], p["I_wheel"]
    I_w_t = 0.5 * I_w                                  # wheel transverse inertia ~ half the spin
    # Composite I_yaw is the WHOLE-ROBOT yaw inertia (incl. the wheels at y=∓d). The MuJoCo
    # wheels carry their own (I_w_t + m_w*d^2 via parallel axis), so the chassis-only yaw must
    # subtract that, else the reassembled total would double-count.
    I_yaw_chassis = I_yaw - 2.0 * (I_w_t + m_w * d * d)
    # Roll is not a 4-state param (roll ~= 0 is a wheel constraint). Pick a physically-valid
    # value satisfying the principal-inertia triangle inequality; it does not affect the
    # planar dynamics we certify.
    I_roll = max(I_yaw_chassis, abs(I_yaw_chassis - I_pitch) + 1e-3)

    overlay = _visual_mesh_overlay() if visual_meshes == "auto" else None
    # When meshes are active, primitive geoms move to display group 3 (hidden by
    # default in viewer and Renderer) -- PURELY visual; contact and inertial
    # behaviour is untouched, which test_visual_meshes_do_not_change_dynamics pins.
    hide = ' group="3"' if overlay else ""

    if wheel == "torus":
        ext = ('<extension><plugin plugin="mujoco.sdf.torus">'
               f'<instance name="wsdf"><config key="radius1" value="{R}"/>'
               f'<config key="radius2" value="{wheel_contact_half_width}"/>'
               '</instance></plugin></extension>')
        asset = '<asset><mesh name="wmesh"><plugin instance="wsdf"/></mesh></asset>'

        def wheel_geom(name):
            return (f'<geom name="{name}" type="sdf" mesh="wmesh" euler="90 0 0"{hide} '
                    f'friction="{mu} 0.005 0.0001" condim="6" '
                    f'solref="{solref[0]} {solref[1]}" margin="{margin}"/>')
    elif wheel == "cylinder":
        ext = asset = ""

        def wheel_geom(name):
            return (f'<geom name="{name}" type="cylinder"{hide} '
                    f'size="{R} {wheel_contact_half_width}" euler="90 0 0" '
                    f'friction="{mu} 0.005 0.0001" condim="6" '
                    f'solref="{solref[0]} {solref[1]}" margin="{margin}"/>')
    else:
        raise ValueError(f"wheel must be 'torus' or 'cylinder', got {wheel!r}")

    if contact_geometry:
        # Chassis: same box, now with real collision (permitted-contact set excludes it).
        base_vis = (f'<geom name="chassis_geom" type="box" size="0.06 {d} 0.06" pos="0 0 {h_cm}" '
                    f'friction="{mu} 0.005 0.0001" condim="6" solref="{solref[0]} {solref[1]}" '
                    f'rgba="0.4 0.5 0.8 0.4"/>')
        leg_y = d * _LEG_Y_RATIO
        # Capsule "fromto" endpoint is the segment end, not the rounded surface -- the actual
        # lowest point of the capsule is one more _LEG_RADIUS below that. Solve for the endpoint
        # so the SURFACE sits C.LEG_GROUND_CLEARANCE above the floor at nominal upright pose
        # (chassis-local z=0 is axle height, i.e. world height R).
        leg_bottom = C.LEG_GROUND_CLEARANCE + _LEG_RADIUS - R
        leg_geoms = "".join(
            f'<geom name="{side}_leg_geom" type="capsule" '
            f'fromto="0 {sign * leg_y} 0  0 {sign * leg_y} {leg_bottom}" size="{_LEG_RADIUS}" '
            f'friction="{mu} 0.005 0.0001" condim="6" solref="{solref[0]} {solref[1]}" '
            f'rgba="0.6 0.3 0.3 0.6"/>'
            for side, sign in (("right", -1), ("left", 1))
        )
    else:
        base_vis = (f'<geom type="box" size="0.06 {d} 0.06" pos="0 0 {h_cm}" '
                    f'contype="0" conaffinity="0"{hide} rgba="0.4 0.5 0.8 0.4"/>')
        leg_geoms = ""

    imu_site = sensor_block = ""
    if imu_pos is not None:
        q = imu_quat if imu_quat is not None else (1.0, 0.0, 0.0, 0.0)
        imu_site = (f'<site name="imu" pos="{imu_pos[0]} {imu_pos[1]} {imu_pos[2]}" '
                    f'quat="{q[0]} {q[1]} {q[2]} {q[3]}"/>')
        sensor_block = ('<sensor><accelerometer name="acc_imu" site="imu"/>'
                        '<gyro name="gyr_imu" site="imu"/></sensor>')

    # Chassis frame origin AT the wheel axle; upright => axle at height R above ground.
    overlay_asset = overlay["asset"] if overlay else ""
    overlay_chassis = overlay["chassis"] if overlay else ""
    overlay_right = overlay["right_wheel"] if overlay else ""
    overlay_left = overlay["left_wheel"] if overlay else ""
    return f"""
<mujoco model="wheeled_ip_{wheel}">
  <option timestep="0.001" integrator="implicitfast" cone="elliptic"/>
  <compiler angle="degree"/>
  {ext}
  {asset}{overlay_asset}
  <default>
    <joint damping="0.0"/>
    <motor ctrlrange="-{p['motor_torque_limit']} {p['motor_torque_limit']}"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05" friction="{mu} 0.005 0.0001" condim="6" solref="{solref[0]} {solref[1]}"/>
    <light pos="0 0 2"/>
    <body name="chassis" pos="0 0 {R}">
      <freejoint name="base"/>
      <!-- sprung body: CoM at h_cm above the axle -->
      <inertial pos="0 0 {h_cm}" mass="{m_b}" diaginertia="{I_roll} {I_pitch} {I_yaw_chassis}"/>
      {base_vis}{leg_geoms}{imu_site}{overlay_chassis}
      <body name="right_wheel" pos="0 {-d} 0">
        <joint name="rwheel" type="hinge" axis="0 1 0"/>
        <inertial pos="0 0 0" mass="{m_w}" diaginertia="{I_w_t} {I_w} {I_w_t}"/>
        {wheel_geom("right_wheel_geom")}{overlay_right}
      </body>
      <body name="left_wheel" pos="0 {d} 0">
        <joint name="lwheel" type="hinge" axis="0 1 0"/>
        <inertial pos="0 0 0" mass="{m_w}" diaginertia="{I_w_t} {I_w} {I_w_t}"/>
        {wheel_geom("left_wheel_geom")}{overlay_left}
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="m_left"  joint="lwheel"/>
    <motor name="m_right" joint="rwheel"/>
  </actuator>
  {sensor_block}
</mujoco>
""".strip()


if __name__ == "__main__":
    # Assembly invariants: the reassembled MuJoCo model must match the composite params
    # (total mass, CoM height above axle, whole-robot yaw inertia about the axle vertical).
    import mujoco
    import numpy as np

    p = load_params()
    R = p["wheel_radius"]
    for wheel in ("cylinder", "torus"):
        m = mujoco.MjModel.from_xml_string(build_mjcf(wheel=wheel, mu=1.0))
        data = mujoco.MjData(m)
        mujoco.mj_forward(m, data)
        total = float(np.sum(m.body_mass[1:]))
        com_above_axle = float(data.subtree_com[m.body("chassis").id][2] - R)
        i_yaw = sum(float(m.body_inertia[b][2] + m.body_mass[b] * (data.xipos[b][0] ** 2 + data.xipos[b][1] ** 2))
                    for b in range(1, m.nbody))
        print(
            f"[{wheel:8s}] v2.2 total_mass {total:.6f} "
            f"(param {p['m_b'] + 2 * p['m_wheel']:.6f}) | "
            f"CoM_above_axle {com_above_axle:.6f} | "
            f"I_yaw {i_yaw:.6f} (param {p['I_yaw']:.6f})"
        )
