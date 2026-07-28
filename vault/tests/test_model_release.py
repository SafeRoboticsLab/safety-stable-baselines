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


LOCK_SHA256 = "0a7e7ecececf369e8eeea5e65942517951ba3910ecf87498a168d99e12173a2d"
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
    assert {
        tuple(float(value) for value in geom.get("size", "").split())
        for geom in wheel_geoms
    } == {(0.12705, 0.02)}


def test_config_kernel_and_mujoco_use_v22_release() -> None:
    mujoco = pytest.importorskip("mujoco")
    from vault import config as C
    from vault.dynamics import CoupledOpt6, OPT6_DH, OPT6_R, _SRC
    from vault.mujoco_model import build_mjcf, load_params

    release = get_model_release()
    assert C.MASS == pytest.approx(19.731467, abs=1e-12)
    assert C.C_THETA == pytest.approx(1.3960776897331257, abs=1e-14)
    assert C.V_ODD == (-0.3, 1.5)
    assert C.MU_SLICES == (0.3, 0.6, 1.0)
    assert OPT6_R == pytest.approx(0.12705)
    assert OPT6_DH == pytest.approx(0.140375)
    assert _SRC == release.artifact_path("opt6_kernel")
    np.testing.assert_allclose(
        CoupledOpt6().f(
            np.array([0.3, 0.1, -0.2, 0.4]),
            np.array([1.0, -0.5]),
        ),
        [0.06663837, -0.2, 1.82102907, -2.70039201],
        rtol=2e-6,
        atol=2e-7,
    )

    params = load_params()
    model = mujoco.MjModel.from_xml_string(build_mjcf(wheel="cylinder"))
    total_mass = float(np.sum(model.body_mass[1:]))
    assert total_mass == pytest.approx(19.731467, abs=1e-6)
    assert total_mass == pytest.approx(
        params["m_b"] + 2.0 * params["m_wheel"],
        abs=1e-12,
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
        [-0.5, -1.35, -6.0, -3.0],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        odd_filter.hi,
        [1.7, 1.35, 6.0, 3.0],
        rtol=0.0,
        atol=1e-12,
    )
    assert capability_filter.lo[0] == pytest.approx(-6.0)
    assert capability_filter.hi[0] == pytest.approx(6.0)


def test_known_stale_checkpoint_is_quarantined(tmp_path: Path) -> None:
    stale = tmp_path / "balance_safety_sac.zip"
    stale.write_bytes(b"stale")

    with pytest.raises(
        ModelReleaseError,
        match="quarantined.*replacement RL training requires explicit operator authorization",
    ):
        get_model_release().verify_checkpoint(stale)
