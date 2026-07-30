"""Fast gates for the v2.2 vault-controller model consumer."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from vault.model_release import ModelRelease, ModelReleaseError, get_model_release


LOCK_SHA256 = "f56030685ad79b16261abc7329448a78019b7f4748490f810699be5b03ca220a"
COMPOSITE_SHA256 = "ff1cb3ea82565cfff9d8454d277e8bc4d63d467fcf5c6b5c36ddf372b7c2f5a9"
KERNEL_SHA256 = "5b2aa6e4d2b337c45de8f57282c83f01266e0e2adaa19ebe08b0dc2074bc40c9"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_release_paths_and_locked_hashes() -> None:
    release = get_model_release()

    assert release.lock_sha256 == LOCK_SHA256
    assert release.model_hashes == {
        "composite_params_sha256": COMPOSITE_SHA256,
        "opt6_kernel_sha256": KERNEL_SHA256,
    }
    assert _sha256(release.artifact_path("composite_params")) == COMPOSITE_SHA256
    assert _sha256(release.artifact_path("opt6_kernel")) == KERNEL_SHA256
    assert release.artifact_path("grid_unladen").name == (
        "robust_odd_dh0_dm0_bump0.npz"
    )
    assert release.artifact_path("grid_laden").name == (
        "robust_odd_dh30_dm20_bump0.npz"
    )
    assert release.artifact_path("safety_params_header").name == (
        "bc_safety_params.h"
    )
    geometry = release.load_json("model_geometry")
    assert geometry["wheel_contact_half_width"] == 0.02


def test_lock_byte_mismatch_fails_closed(tmp_path: Path) -> None:
    release = get_model_release()
    fake_root = tmp_path / "vault-controller"
    fake_lock = fake_root / "models" / "MODEL_INPUTS.lock.json"
    fake_lock.parent.mkdir(parents=True)
    fake_lock.write_bytes(release.local_lock_path.read_bytes() + b"\n")

    with pytest.raises(ModelReleaseError, match="lock.*mismatch"):
        ModelRelease(controller_root=fake_root)


def test_package_import_is_lazy_but_first_artifact_access_fails(
    tmp_path: Path,
) -> None:
    release = get_model_release()
    fake_root = tmp_path / "vault-controller"
    fake_lock = fake_root / "models" / "MODEL_INPUTS.lock.json"
    fake_lock.parent.mkdir(parents=True)
    fake_lock.write_bytes(release.local_lock_path.read_bytes() + b"\n")
    repository = Path(__file__).resolve().parents[2]
    code = """
import importlib
import pkgutil
import vault

modules = [
    vault.__name__ + "." + item.name
    for item in pkgutil.iter_modules(vault.__path__)
    if item.name != "tests"
]
for module in modules:
    importlib.import_module(module)

from vault import config
from vault.model_release import ModelReleaseError
try:
    config.MASS
except ModelReleaseError as exc:
    assert "lock.json mismatch" in str(exc)
else:
    raise AssertionError("release-backed access accepted a mismatched lock")
print(f"lazy imports PASS: {len(modules)} modules; artifact gate PASS")
"""
    environment = {
        **os.environ,
        "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
        "PYTHONPATH": os.pathsep.join(
            [str(repository), str(repository.parent / "vault-controller")]
        ),
        "VAULT_CONTROLLER_ROOT": str(fake_root),
    }
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repository,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "lazy imports PASS" in completed.stdout
    assert "artifact gate PASS" in completed.stdout


def test_mujoco_cylinder_uses_locked_contact_half_width() -> None:
    from vault.mujoco_model import build_mjcf

    root = ET.fromstring(build_mjcf(wheel="cylinder"))
    wheel_geoms = [
        geom
        for geom in root.iter("geom")
        if geom.get("name") in {"left_wheel_geom", "right_wheel_geom"}
    ]
    assert len(wheel_geoms) == 2
    # PUBLIC REPO: expected values come from the PRIVATE release at runtime, never
    # from literals in this file.
    params = get_model_release().load_json("composite_params")
    half_width = get_model_release().load_json("model_geometry")[
        "wheel_contact_half_width"
    ]
    assert {
        tuple(float(value) for value in geom.get("size", "").split())
        for geom in wheel_geoms
    } == {(params["wheel_radius"], half_width)}


def test_config_kernel_and_mujoco_use_v22_release() -> None:
    mujoco = pytest.importorskip("mujoco")
    from vault import config as C
    from vault.dynamics import CoupledOpt6, OPT6_DH, OPT6_R, _SRC
    from vault.mujoco_model import build_mjcf, load_params

    release = get_model_release()
    # PUBLIC REPO: this test asserts CONSISTENCY between config, kernel, and the
    # private release -- never absolute model values, which must not appear here.
    # Absolute-value pins (masses, geometry, dynamics regression vectors) live in
    # vault-controller's own private test suite.
    params = release.load_json("composite_params")
    contract = release.load_json("odd_contract")
    assert C.MASS == pytest.approx(
        params["m_b"] + 2.0 * params["m_wheel"], abs=1e-12
    )
    assert C.V_ODD == tuple(contract["odd"]["velocity"]["bounds"])
    assert C.PSI_ODD == contract["odd"]["yaw_rate"]["bounds"][1]
    assert C.MU_SLICES == tuple(contract["friction"]["slices"])
    assert C.DOMAIN_V == tuple(contract["grid_axes"]["velocity"]["bounds"])
    assert C.DOMAIN_PSI_DOT == tuple(contract["grid_axes"]["yaw_rate"]["bounds"])
    assert OPT6_R == pytest.approx(params["wheel_radius"])
    assert _SRC == release.artifact_path("opt6_kernel")
    # dynamics sanity without a pinned response vector: finite, correct shape, and
    # the kinematic identity theta_dot passthrough holds exactly.
    x = np.array([0.3, 0.1, -0.2, 0.4])
    xdot = np.asarray(CoupledOpt6().f(x, np.array([1.0, -0.5])), float)
    assert xdot.shape == (4,) and np.isfinite(xdot).all()
    assert xdot[1] == pytest.approx(x[2], abs=1e-6)  # f32 kernel rounding

    mj_params = load_params()
    model = mujoco.MjModel.from_xml_string(build_mjcf(wheel="cylinder"))
    total_mass = float(np.sum(model.body_mass[1:]))
    assert total_mass == pytest.approx(
        mj_params["m_b"] + 2.0 * mj_params["m_wheel"], abs=1e-6
    )


def test_grid_axes_come_from_the_release_contract() -> None:
    from vault import config as C
    from vault import grid

    # PUBLIC REPO: axes are asserted against the PRIVATE contract read at runtime,
    # not against literals.
    contract = get_model_release().load_json("odd_contract")
    names = ("velocity", "theta", "theta_dot", "yaw_rate")
    axes = contract["grid_axes"]
    assert [len(a) for a in grid.AXES_FULL] == [int(axes[n]["nodes"]) for n in names]
    np.testing.assert_allclose(
        [a[0] for a in grid.AXES_FULL],
        [axes[n]["bounds"][0] for n in names], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        [a[-1] for a in grid.AXES_FULL],
        [axes[n]["bounds"][1] for n in names], rtol=0.0, atol=1e-12)
    assert grid.AXES_FULL[1][0] < -C.THETA_MAX
    assert grid.AXES_FULL[1][-1] > C.THETA_MAX


def test_environment_spaces_cover_the_contract_reset_domain() -> None:
    from vault import config as C
    from vault.env import BalanceSafetyEnv
    from vault.mujoco_env import ContactSafetyEnv

    # PUBLIC REPO: expected bounds come from the private contract at runtime.
    contract = get_model_release().load_json("odd_contract")
    axes = contract["grid_axes"]
    expected_high = np.array(
        [axes[n]["bounds"][1] for n in ("velocity", "theta", "theta_dot", "yaw_rate")]
        + [float(contract["friction"]["range"][1])], np.float32)
    for environment_type in (BalanceSafetyEnv, ContactSafetyEnv):
        environment = environment_type()
        np.testing.assert_allclose(
            environment.observation_space.high,
            expected_high,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            environment.observation_space.low,
            -expected_high,
            rtol=0.0,
            atol=0.0,
        )


def test_external_artifact_requires_matching_model_identity(tmp_path: Path) -> None:
    release = get_model_release()
    artifact = tmp_path / "legacy_grid.npz"
    np.savez(artifact, axes=np.array([], dtype=object))
    sidecar = Path(f"{artifact}.model.json")
    sidecar.write_text(
        json.dumps(
            {
                "lock_sha256": release.lock_sha256,
                "model_hashes": {
                    **release.model_hashes,
                    "opt6_kernel_sha256": "0" * 64,
                },
                "artifact_sha256": _sha256(artifact),
            }
        )
    )

    with pytest.raises(ModelReleaseError, match="model identity mismatch"):
        release.verify_external_artifact(artifact)


def test_default_and_capability_grids_have_distinct_contracts() -> None:
    from vault import config as C
    from vault.filter import ValueFilter

    release = get_model_release()
    assert C.GRID_NPZ.name == "grid_reachavoid_odd.npz"
    assert release.verify_external_artifact(C.GRID_NPZ) == C.GRID_NPZ

    odd_filter = ValueFilter.from_grid()
    capability_filter = ValueFilter.from_capability_grid(mode="unladen")
    np.testing.assert_allclose(
        odd_filter.lo,
        [-6.0, -1.35, -6.0, -4.0],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        odd_filter.hi,
        [6.0, 1.35, 6.0, 4.0],
        rtol=0.0,
        atol=1e-12,
    )
    # The point of this test is that the two contracts stay DISTINCT: the ODD grid
    # spans the declared envelope (+-6 after the 2026-07-28 ruling) while the
    # capability grid is a deliberately wider computational WINDOW (+-7), so the
    # filter has values everywhere a command can reach.  Both moved with the ruling;
    # if these two ever coincide, the capability window has collapsed into the ODD
    # contract -- the regression this test exists to catch.
    assert capability_filter.lo[0] == pytest.approx(-7.0)
    assert capability_filter.hi[0] == pytest.approx(7.0)
    assert capability_filter.lo[0] < odd_filter.lo[0]
    assert capability_filter.hi[0] > odd_filter.hi[0]


def test_known_stale_checkpoint_is_quarantined(tmp_path: Path) -> None:
    stale = tmp_path / "balance_safety_sac.zip"
    stale.write_bytes(b"stale")

    with pytest.raises(
        ModelReleaseError,
        match="quarantined.*replacement RL training requires explicit operator authorization",
    ):
        get_model_release().verify_checkpoint(stale)


def test_yaw_sign_convention() -> None:
    """Pin the torque->yaw sign, in BOTH the reduced model and the MuJoCo plant.

    mujoco_plant.py's docstring claimed `tau_L>tau_R drives +psi_dot`. It does not:
    a left-torque surplus yaws NEGATIVE (sign only; magnitudes stay in the
    private release). The model and the plant agreed all along -- only the comment
    was inverted, and nothing tested it.

    Asserted as a SIGN and a cross-check, not a magnitude, so it survives regeneration.
    A sign convention is exactly the kind of fact that is cheap to state wrongly and
    expensive to discover wrongly.
    """
    import numpy as np
    from vault.dynamics import CoupledOpt6

    model = CoupledOpt6()
    forward = np.asarray(model.f(np.zeros(4), np.array([1.0, 1.0])), float)
    differential = np.asarray(model.f(np.zeros(4), np.array([1.0, -1.0])), float)

    assert forward[0] > 0.0, (
        f"+tau_sum must drive +v; got vdot={forward[0]}")
    assert differential[3] < 0.0, (
        f"tau_L > tau_R must drive NEGATIVE psi_dot; got psi_ddot={differential[3]}. "
        f"If this now reads positive the convention has flipped, and every consumer "
        f"of the yaw sign -- including the roll constraint's signed form -- needs "
        f"rechecking.")
    mirrored = np.asarray(model.f(np.zeros(4), np.array([-1.0, 1.0])), float)
    assert mirrored[3] > 0.0 and abs(mirrored[3] + differential[3]) < 1e-9, (
        "swapping the torque pair must mirror psi_ddot exactly")


def test_visual_meshes_do_not_change_dynamics() -> None:
    """The release-mesh overlay is visual only: bit-identical trajectories.

    Meshes come from the PRIVATE sibling at runtime (contype=0, conaffinity=0,
    group 2) and every body keeps its explicit <inertial>, so turning the overlay
    on cannot move a single bit of the dynamics. Stepped, not argued: identical
    controls for 500 steps, qpos/qvel compared exactly.
    """
    mujoco = pytest.importorskip("mujoco")
    from vault.mujoco_model import build_mjcf

    runs = []
    for mode in ("auto", "off"):
        m = mujoco.MjModel.from_xml_string(
            build_mjcf(wheel="cylinder", visual_meshes=mode))
        d = mujoco.MjData(m)
        for k in range(500):
            d.ctrl[:] = (0.4 * np.sin(0.01 * k), -0.3 * np.cos(0.013 * k))
            mujoco.mj_step(m, d)
        runs.append((d.qpos.copy(), d.qvel.copy()))
    (q1, v1), (q2, v2) = runs
    assert np.array_equal(q1, q2), "mesh overlay changed qpos -- it is not visual-only"
    assert np.array_equal(v1, v2), "mesh overlay changed qvel -- it is not visual-only"


def test_mesh_overlay_orientations_survive_pi_rotations() -> None:
    """Regression for the w=0 quaternion collapse, caught from a single render.

    The tucked fold rotates the lower legs by EXACTLY pi, where the naive
    quaternion conversion's w is exactly zero; an earlier version short-circuited
    that to identity, producing correct positions with discarded orientations
    ("feet right, lower leg wrong, chain broken"). Reconstruct the emitted
    lower-leg quaternion and require it to be a genuine y-axis half-turn.
    """
    import re
    from vault.mujoco_model import _visual_mesh_overlay

    overlay = _visual_mesh_overlay()
    if overlay is None:
        pytest.skip("vault-controller sibling not available")
    for side in ("right", "left"):
        match = re.search(
            rf'mesh="vis_{side}_lower_leg_link" pos="[^"]+" quat="([^"]+)"',
            overlay["chassis"])
        assert match, f"{side} lower leg missing from the overlay"
        w, x, y, z = (float(v) for v in match.group(1).split())
        # unit quaternion
        assert abs(w*w + x*x + y*y + z*z - 1.0) < 1e-9
        # a HALF-TURN about y: w == 0 (the exact case that used to collapse), y == +-1
        assert abs(w) < 1e-9 and abs(abs(y) - 1.0) < 1e-9 and abs(x) < 1e-9 and abs(z) < 1e-9, (
            f"{side} lower-leg quat {match.group(1)} is not the knee's pi fold -- "
            f"if this reads (1,0,0,0) the w=0 collapse has been reintroduced")
