"""Measure the lateral acceleration at which a wheel lifts, versus wheel width.

The certificate's roll bound a_tip = g*W/(2H) assumes POINT contact in the wheel
plane. A finite-width cylinder tips about its outer rim, and MuJoCo's contacts are
compliant (solref time constant 20 ms, slower than the 10 ms control step), so the
simulated robot lifts a wheel later than the certificate says. This quantifies that
gap, which bounds how much the "zero liftoff" results from the MuJoCo batteries are
actually worth.

Method: a RIGID body on two cylinder wheels -- no pitch freedom, so roll is isolated
from the inverted-pendulum collapse -- settled, then a lateral force ramped at the
CoM until a wheel loses contact.

Measured 2026-07-28 (unladen, centred, quasi-static):

    half-width   full width   a_tip measured   vs analytic 6.999
      2.5 mm        5 mm          7.470            +6.7%
     10   mm       20 mm          7.382            +5.5%
     20   mm       40 mm          7.512            +7.3%   <- shipped geometry
     40   mm       80 mm          7.982           +14.0%

Reading: at the shipped 40 mm wheel the simulator is ~7% harder to tip than the
certificate assumes, so MuJoCo liftoff results are mildly optimistic. Most of that
offset is NOT wheel width -- a near-point-contact 5 mm wheel still measures +6.7%,
and the spread across the first three rows is within the ~1.5% run-to-run scatter.
The residual is attributable to contact compliance, which is why narrowing the wheel
does not fix it. Width only matters clearly above ~40 mm half-width.

Caveats: simulation only, no hardware witness; unladen and centred, so it excludes
the 9.74 mm lateral CoM offset that makes a_tip 5.5% optimistic in one turn
direction; the rig's rotational inertia is representative rather than derived from
the release; quasi-static, so it is a static tipping threshold, not a dynamic one.

Run:
    PYTHONPATH=$PWD:../vault-controller python vault/tools/measure_tipping_threshold.py
"""
from __future__ import annotations

import mujoco
import numpy as np

from vault import config as C

MASS = 19.731467
WIDTHS = (0.0025, 0.01, 0.02, 0.04)


def _rig(half_width: float) -> str:
    """Rigid body on two cylinder wheels: no pitch DOF, so roll is isolated."""
    r, w, h = C.WHEEL_R, C.TRACK, C.COM_H
    friction = 'friction="1 .005 .0001" condim="6"'
    return f"""<mujoco>
      <option timestep="0.0005" integrator="implicitfast" cone="elliptic"/>
      <worldbody>
        <geom type="plane" size="20 20 .1" {friction}/>
        <body name="rig" pos="0 0 {r}"><freejoint/>
          <inertial pos="0 0 {h - r}" mass="{MASS}" diaginertia="0.42 0.42 0.59"/>
          <geom type="cylinder" size="{r} {half_width}" pos="0 {w / 2} 0" euler="90 0 0" {friction}/>
          <geom type="cylinder" size="{r} {half_width}" pos="0 {-w / 2} 0" euler="90 0 0" {friction}/>
        </body>
      </worldbody></mujoco>"""


def tipping_accel(half_width: float, force_rate: float = 0.01) -> float:
    """Ramp a lateral force at the CoM; return |a_lat| when a wheel loses contact."""
    model = mujoco.MjModel.from_xml_string(_rig(half_width))
    data = mujoco.MjData(model)
    com_offset = np.array([0.0, 0.0, C.COM_H - C.WHEEL_R])

    for _ in range(4000):                       # settle
        mujoco.mj_step(model, data)

    force = 0.0
    for _ in range(600_000):
        force += force_rate
        mujoco.mj_applyFT(model, data, np.array([0.0, force, 0.0]), np.zeros(3),
                          data.xpos[1] + com_offset, 1, data.qfrc_applied)
        mujoco.mj_step(model, data)
        data.qfrc_applied[:] = 0.0
        if data.ncon < 2:                       # a wheel has left the ground
            return force / MASS
        if abs(data.qpos[1]) > 0.5:             # slid away instead of tipping
            break
    return float("nan")


def main() -> int:
    analytic = C.GRAV * C.TRACK / (2 * C.COM_H)
    print(f"analytic a_tip (point contact, used by the certificate) = {analytic:.3f} m/s^2\n")
    print(f"{'half-width':>11} {'full (m)':>9} {'measured':>10} {'vs analytic':>12}")
    for half_width in WIDTHS:
        a = tipping_accel(half_width)
        rel = f"{100 * (a / analytic - 1):+.1f}%" if np.isfinite(a) else "n/a"
        print(f"{half_width:11.4f} {2 * half_width:9.3f} {a:10.3f} {rel:>12}")
    print("\nSimulated tipping is ~7% later than the certificate at the shipped 40 mm wheel;")
    print("most of the offset is contact compliance, not wheel width. See the module docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
