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
#  Robert's reduced 4-state model has no full-robot geometry (by design, see CLAUDE.md);
#  this repo has no chassis/leg CAD either. The dimensions below are read directly from
#  the real robot's URDF (SafeRoboticsLab/VaultRobot_IsaacLab, delivery_robot_v2), which
#  is the actual embodiment behind the reduced model (its half-wheel-separation of
#  0.140375 m matches OPT6_DH in dynamics.py exactly):
#    right_hip_joint     parent=base_link            xyz="0 -0.26780 -0.05100"
#    right_...->knee     xyz="0 -0.01577500 0.20350000"   (upper leg span ~0.2035 m)
#    right_lower->foot   xyz="0 -0.02215000 0.20000"      (lower leg span ~0.2000 m)
#  i.e. hip mount ~0.05 m below the body_wheel/axle plane, at ~1.91x the wheel's lateral
#  offset (0.2678 / 0.140375), with a ~0.40 m fully-extended leg. The physical robot's
#  nominal stance folds the hip/knee so the foot sits near the wheel-contact height
#  (composite_params.json's own derivation notes describe the legs as "locked ...
#  tucked" for the reduced model) -- there is no folded-leg kinematics in this planar
#  plant, so we approximate the tucked leg as a single rigid capsule fixed to the
#  chassis: outboard of the wheel (no self-overlap), spanning from axle height down to
#  just short of the ground at nominal upright pose (contacts only once the chassis
#  pitches/rolls). This is a coarse stand-in for upper_leg_link + lower_leg_link, not a
#  faithful reproduction of the coupled hip-knee linkage.
_LEG_Y_RATIO = 0.26780 / 0.140375          # hip lateral offset / body-wheel lateral offset
_LEG_RADIUS = 0.025                        # m, slender capsule (real leg links are thin)
# Ground clearance: C.LEG_GROUND_CLEARANCE (config.py) -- shared with contact_margin.py, which
# normalizes by the SAME constant so a nominal upright state's margin is ~1.0, not ~0.02 m.


def build_mjcf(wheel: str = "torus", mu: float = 1.0,
               params: dict | None = None, solref: tuple = (0.02, 1.0), margin: float = 0.0,
               imu_pos: tuple | None = None, imu_quat: tuple | None = None,
               contact_geometry: bool = False) -> str:
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

    if wheel == "torus":
        ext = ('<extension><plugin plugin="mujoco.sdf.torus">'
               f'<instance name="wsdf"><config key="radius1" value="{R}"/>'
               f'<config key="radius2" value="{wheel_contact_half_width}"/>'
               '</instance></plugin></extension>')
        asset = '<asset><mesh name="wmesh"><plugin instance="wsdf"/></mesh></asset>'

        def wheel_geom(name):
            return (f'<geom name="{name}" type="sdf" mesh="wmesh" euler="90 0 0" '
                    f'friction="{mu} 0.005 0.0001" condim="6" '
                    f'solref="{solref[0]} {solref[1]}" margin="{margin}"/>')
    elif wheel == "cylinder":
        ext = asset = ""

        def wheel_geom(name):
            return (f'<geom name="{name}" type="cylinder" '
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
                    f'contype="0" conaffinity="0" rgba="0.4 0.5 0.8 0.4"/>')
        leg_geoms = ""

    imu_site = sensor_block = ""
    if imu_pos is not None:
        q = imu_quat if imu_quat is not None else (1.0, 0.0, 0.0, 0.0)
        imu_site = (f'<site name="imu" pos="{imu_pos[0]} {imu_pos[1]} {imu_pos[2]}" '
                    f'quat="{q[0]} {q[1]} {q[2]} {q[3]}"/>')
        sensor_block = ('<sensor><accelerometer name="acc_imu" site="imu"/>'
                        '<gyro name="gyr_imu" site="imu"/></sensor>')

    # Chassis frame origin AT the wheel axle; upright => axle at height R above ground.
    return f"""
<mujoco model="wheeled_ip_{wheel}">
  <option timestep="0.001" integrator="implicitfast" cone="elliptic"/>
  <compiler angle="degree"/>
  {ext}
  {asset}
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
      {base_vis}{leg_geoms}{imu_site}
      <body name="right_wheel" pos="0 {-d} 0">
        <joint name="rwheel" type="hinge" axis="0 1 0"/>
        <inertial pos="0 0 0" mass="{m_w}" diaginertia="{I_w_t} {I_w} {I_w_t}"/>
        {wheel_geom("right_wheel_geom")}
      </body>
      <body name="left_wheel" pos="0 {d} 0">
        <joint name="lwheel" type="hinge" axis="0 1 0"/>
        <inertial pos="0 0 0" mass="{m_w}" diaginertia="{I_w_t} {I_w} {I_w_t}"/>
        {wheel_geom("left_wheel_geom")}
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
