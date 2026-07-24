"""contact_margin -- MuJoCo-contact-based safety margins for the vault robot.

Complements f_cert.py's analytic pitch/roll margins with two failure modes that need real
rigid-body contact and can't be expressed as a function of the reduced 4-state alone:

  contact_margin(plant)            : continuous signed distance (METERS) from every non-wheel/foot
                                     geom to the terrain. Safe (>=0) iff nothing but the wheels
                                     touches the ground. Raw physical units -- for diagnostics/HUD.
  contact_margin_normalized(plant) : contact_margin(plant) / C.LEG_GROUND_CLEARANCE, so a nominal
                                     upright state normalizes to ~1.0 -- the SAME O(1) scale as
                                     f_cert.margin. Use THIS (not the raw meters version) whenever
                                     combining with f_cert.margin via min(): a raw ~0.02 m value
                                     would dominate any min() with an O(1) quantity and crush the
                                     combined signal's dynamic range almost everywhere, regardless
                                     of how safe the pitch/roll state actually is -- exactly the bug
                                     that made an early version of this env's trained Q-values all
                                     bunch up near ~0.02-0.07 with almost no discriminative range.
  slam_margin(plant)               : 1 - (worst active contact's closing speed this step) /
                                     C.SLAM_VEL_MAX. Applies to ALL contacts (wheels included) --
                                     a slam is a slam regardless of which part hits. Already O(1).
  margin(plant)                    : min(contact_margin_normalized, slam_margin), mirroring
                                     f_cert.margin's min-of-conditions style and scale.

Unlike f_cert.margin (a pure function of the 4-state), these operate on a live MujocoPlant --
they need per-geom contact/position data that doesn't exist in the reduced state. Requires the
plant be built with contact geometry enabled: MujocoPlant(contact_geometry=True) (see
mujoco_model.build_mjcf). Assumes the current flat (z=0 plane) terrain -- revisit the
lowest-point-vs-plane assumption in contact_margin() for the Track-B non-flat terrain.
"""
from __future__ import annotations

import numpy as np
import mujoco

from . import config as C

_PERMITTED_SUBSTR = ("wheel",)   # geom names containing these substrings are "wheels/feet"


def _is_permitted(name: str) -> bool:
    return any(s in (name or "") for s in _PERMITTED_SUBSTR)


def _non_wheel_geom_ids(model) -> list[int]:
    return [i for i in range(model.ngeom)
            if model.geom(i).name not in ("", "floor") and not _is_permitted(model.geom(i).name)]


def _lowest_point_z(model, data, gid: int) -> float:
    """Lowest world-frame z of geom `gid`'s surface. Supports the box/capsule primitives used
    by mujoco_model.build_mjcf's contact_geometry=True path."""
    gtype = model.geom_type[gid]
    pos = data.geom_xpos[gid]
    mat = data.geom_xmat[gid].reshape(3, 3)          # world = mat @ local (see mujoco_plant.py)
    size = model.geom_size[gid]
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        corners = pos + signs * size @ mat.T
        return float(corners[:, 2].min())
    if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        axis = mat[:, 2]                             # capsule's local z-axis, in world
        half_len = size[1]
        endpoints_z = np.array([(pos + half_len * axis)[2], (pos - half_len * axis)[2]])
        return float(endpoints_z.min() - size[0])     # subtract the rounded-cap radius
    raise NotImplementedError(f"contact_margin: unsupported geom type {gtype!r} for geom {gid}")


def contact_margin(plant) -> float:
    """Signed distance (m) from the closest non-wheel/foot geom to the (flat, z=0) terrain.
    Positive = clear of the ground; safe iff >= 0. Raw meters -- see contact_margin_normalized
    for the O(1)-scaled version to use when combining with f_cert.margin."""
    model, data = plant.model, plant.data
    ids = _non_wheel_geom_ids(model)
    if not ids:
        return C.LEG_GROUND_CLEARANCE   # no contact_geometry -- report the nominal clearance
    return float(min(_lowest_point_z(model, data, gid) for gid in ids))


def contact_margin_normalized(plant) -> float:
    """contact_margin(plant) / C.LEG_GROUND_CLEARANCE -- ~1.0 at the nominal upright state,
    comparable to f_cert.margin's O(1) scale. Use this, not the raw meters version, in any
    min()-combination with f_cert.margin (env reward, filters)."""
    return contact_margin(plant) / C.LEG_GROUND_CLEARANCE


def slam_margin(plant) -> float:
    """1 - (worst active contact's closing speed this step) / C.SLAM_VEL_MAX. Negative iff some
    contact (any geom, including wheels) is closing faster than C.SLAM_VEL_MAX -- a slam."""
    data = plant.data
    worst_closing = 0.0
    for i in range(data.ncon):
        adr = data.contact[i].efc_address
        if adr < 0:
            continue
        closing = -float(data.efc_vel[adr])           # efc_vel>=0 is separating; negate for speed
        worst_closing = max(worst_closing, closing)
    return 1.0 - worst_closing / C.SLAM_VEL_MAX


def margin(plant) -> float:
    """Combined contact-based margin, min-of-conditions style and O(1) scale (mirrors
    f_cert.margin) -- uses the NORMALIZED contact margin, not raw meters."""
    return min(contact_margin_normalized(plant), slam_margin(plant))
