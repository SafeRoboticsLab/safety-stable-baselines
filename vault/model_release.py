"""Provenance-checked access to the vault-controller v2.2 model release.

This package does not carry robot dynamics or safety artifacts.  It consumes the
single release in ``vault-controller/models`` and refuses to load it unless this
repository's lock file is byte-identical to the controller lock.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


class ModelReleaseError(RuntimeError):
    """The requested artifact is absent, stale, or not provenance-compatible."""


_PKG_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_ROOT.parent
_LOCAL_LOCK = _PKG_ROOT / "data" / "MODEL_INPUTS.lock.json"
_RELEASE_SIDECAR = _PKG_ROOT / "data" / "release_artifacts.model.json"
_CHECKPOINT_MANIFEST = _PKG_ROOT / "data" / "checkpoints_manifest.json"

_ARTIFACTS = {
    "odd_contract": "models/source/odd_contract.json",
    "model_geometry": "models/source/geometry/model_geometry.json",
    "composite_params": "models/generated/reduced/composite_params.json",
    "controller_limits": "models/generated/reduced/controller_limits.json",
    "roll_constraint_params": (
        "models/generated/safety/roll_constraint_params.json"
    ),
    "opt6_kernel": (
        "models/generated/reduced/reduced_struct_fjac_f32_opt6.c"
    ),
    "coupled_residual": (
        "models/generated/safety/residuals/coupled_residual_fit.json"
    ),
    "grid_unladen": (
        "models/generated/safety/grids/robust_odd_dh0_dm0_bump0.npz"
    ),
    "grid_laden": (
        "models/generated/safety/grids/robust_odd_dh30_dm20_bump0.npz"
    ),
    "v_mlp_unladen_json": (
        "models/generated/safety/mlp/v_mlp_unladen.json"
    ),
    "v_mlp_unladen_pt": (
        "models/generated/safety/mlp/v_mlp_unladen.pt"
    ),
    "v_mlp_laden_json": "models/generated/safety/mlp/v_mlp_laden.json",
    "v_mlp_laden_pt": "models/generated/safety/mlp/v_mlp_laden.pt",
    "safety_params_header": (
        "balance_controller/c/include/balance_controller/bc_safety_params.h"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ModelReleaseError(f"required model release file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ModelReleaseError(f"invalid JSON in model release file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelReleaseError(f"expected a JSON object in {path}")
    return value


class ModelRelease:
    """Validated paths and metadata for one vault-controller model release."""

    def __init__(
        self,
        controller_root: str | Path | None = None,
        local_lock: str | Path = _LOCAL_LOCK,
        release_sidecar: str | Path = _RELEASE_SIDECAR,
        checkpoint_manifest: str | Path = _CHECKPOINT_MANIFEST,
    ) -> None:
        default_root = _REPO_ROOT.parent / "vault-controller"
        configured_root = controller_root or os.environ.get("VAULT_CONTROLLER_ROOT")
        self.controller_root = Path(configured_root or default_root).expanduser().resolve()
        self.local_lock_path = Path(local_lock).resolve()
        self.controller_lock_path = (
            self.controller_root / "models" / "MODEL_INPUTS.lock.json"
        )
        self.release_sidecar_path = Path(release_sidecar).resolve()
        self.checkpoint_manifest_path = Path(checkpoint_manifest).resolve()

        try:
            local_bytes = self.local_lock_path.read_bytes()
            controller_bytes = self.controller_lock_path.read_bytes()
        except FileNotFoundError as exc:
            raise ModelReleaseError(
                "vault-controller v2.2 model release is unavailable; set "
                "VAULT_CONTROLLER_ROOT to the pinned controller checkout "
                f"(missing {exc.filename})"
            ) from exc
        if local_bytes != controller_bytes:
            raise ModelReleaseError(
                "MODEL_INPUTS.lock.json mismatch: safety-stable-baselines and "
                "vault-controller are not pinned to the same model release"
            )

        self.lock_sha256 = hashlib.sha256(local_bytes).hexdigest()
        self.lock = _read_json(self.local_lock_path)
        if self.lock.get("release") != "vault-robot-v2.2-camera-inclusive-reduced-balance":
            raise ModelReleaseError(
                f"unsupported model release {self.lock.get('release')!r}"
            )
        self._release_files = self.lock.get("release_files")
        if not isinstance(self._release_files, dict):
            raise ModelReleaseError("model lock has no release_files hash table")

        self.model_hashes = {
            "composite_params_sha256": self._locked_hash(
                _ARTIFACTS["composite_params"]
            ),
            "opt6_kernel_sha256": self._locked_hash(_ARTIFACTS["opt6_kernel"]),
        }
        self._verify_release_sidecar()

    def _locked_hash(self, relative_path: str) -> str:
        expected = self._release_files.get(relative_path)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ModelReleaseError(
                f"model lock has no SHA-256 for release artifact {relative_path}"
            )
        return expected

    def _verify_model_hashes(self, value: Any, source: Path) -> None:
        if value != self.model_hashes:
            raise ModelReleaseError(
                f"model identity mismatch in {source}: expected {self.model_hashes}, "
                f"got {value}"
            )

    def _verify_release_sidecar(self) -> None:
        sidecar = _read_json(self.release_sidecar_path)
        if sidecar.get("lock_sha256") != self.lock_sha256:
            raise ModelReleaseError(
                f"lock hash mismatch in release sidecar {self.release_sidecar_path}"
            )
        self._verify_model_hashes(
            sidecar.get("model_hashes"), self.release_sidecar_path
        )
        artifacts = sidecar.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ModelReleaseError(
                f"release sidecar has no artifact map: {self.release_sidecar_path}"
            )
        if artifacts != _ARTIFACTS:
            raise ModelReleaseError(
                "release artifact map differs from the consumer's expected roles"
            )

    def artifact_path(self, role: str) -> Path:
        """Return an artifact path after lock, role, and file-hash validation."""
        try:
            relative_path = _ARTIFACTS[role]
        except KeyError as exc:
            raise ModelReleaseError(f"unknown model release artifact role: {role}") from exc
        path = self.controller_root / relative_path
        if not path.is_file():
            raise ModelReleaseError(f"model release artifact is missing: {path}")
        expected = self._locked_hash(relative_path)
        actual = _sha256(path)
        if actual != expected:
            raise ModelReleaseError(
                f"model release artifact hash mismatch for {path}: "
                f"expected {expected}, got {actual}"
            )
        return path

    def load_json(self, role: str) -> dict[str, Any]:
        return _read_json(self.artifact_path(role))

    def safety_grid_mus(self, mode: str = "unladen") -> tuple[float, ...]:
        try:
            values = self.lock["phase5_safety_artifacts"]["gates"][
                "robust_grid_convergence"
            ]["terminal_dv"][mode]
        except (KeyError, TypeError) as exc:
            raise ModelReleaseError(
                f"model lock has no robust-grid metadata for mode {mode!r}"
            ) from exc
        return tuple(float(value) for value in values)

    def verify_external_artifact(
        self, artifact: str | Path, sidecar: str | Path | None = None
    ) -> Path:
        """Verify an external artifact against an adjacent model-hash sidecar."""
        path = Path(artifact).expanduser().resolve()
        sidecar_path = (
            Path(sidecar).expanduser().resolve()
            if sidecar is not None
            else Path(f"{path}.model.json")
        )
        metadata = _read_json(sidecar_path)
        self._verify_model_hashes(metadata.get("model_hashes"), sidecar_path)
        if metadata.get("lock_sha256") != self.lock_sha256:
            raise ModelReleaseError(
                f"model lock hash mismatch in artifact sidecar {sidecar_path}"
            )
        expected = metadata.get("artifact_sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ModelReleaseError(
                f"artifact sidecar has no valid artifact_sha256: {sidecar_path}"
            )
        if not path.is_file():
            raise ModelReleaseError(f"artifact referenced by sidecar is missing: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ModelReleaseError(
                f"artifact hash mismatch for {path}: expected {expected}, got {actual}"
            )
        return path

    def write_external_artifact_sidecar(self, artifact: str | Path) -> Path:
        """Write provenance for a newly generated, non-release artifact."""
        path = Path(artifact).expanduser().resolve()
        if not path.is_file():
            raise ModelReleaseError(f"cannot sidecar missing artifact: {path}")
        sidecar_path = Path(f"{path}.model.json")
        payload = {
            "schema_version": 1,
            "lock_sha256": self.lock_sha256,
            "model_hashes": self.model_hashes,
            "artifact_sha256": _sha256(path),
        }
        sidecar_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return sidecar_path

    def verify_checkpoint(self, checkpoint: str | Path) -> Path:
        """Require a matching sidecar or an explicit compatible manifest entry."""
        requested = Path(checkpoint).expanduser()
        path = requested if requested.suffix == ".zip" else requested.with_suffix(".zip")
        path = path.resolve()
        sidecar = Path(f"{path}.model.json")
        if sidecar.is_file():
            return self.verify_external_artifact(path, sidecar)

        manifest = _read_json(self.checkpoint_manifest_path)
        if manifest.get("release_lock_sha256") != self.lock_sha256:
            raise ModelReleaseError(
                f"checkpoint manifest lock mismatch: {self.checkpoint_manifest_path}"
            )
        entries = manifest.get("checkpoints")
        if not isinstance(entries, list):
            raise ModelReleaseError(
                f"checkpoint manifest has no checkpoints list: {self.checkpoint_manifest_path}"
            )
        names = (path.name, path.stem)
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            pattern = entry.get("pattern")
            if not isinstance(pattern, str) or not any(
                fnmatch.fnmatch(name, pattern) for name in names
            ):
                continue
            status = entry.get("status")
            if status == "compatible":
                self._verify_model_hashes(
                    entry.get("model_hashes"), self.checkpoint_manifest_path
                )
                if entry.get("lock_sha256") != self.lock_sha256:
                    raise ModelReleaseError(
                        f"checkpoint manifest lock mismatch for {path.name}"
                    )
                expected = entry.get("artifact_sha256")
                if not path.is_file() or _sha256(path) != expected:
                    raise ModelReleaseError(
                        f"compatible checkpoint hash mismatch for {path}"
                    )
                return path
            raise ModelReleaseError(
                f"checkpoint {path.name} is quarantined as incompatible with "
                f"model release {self.lock_sha256}; replacement RL training requires "
                "explicit operator authorization"
            )
        raise ModelReleaseError(
            f"checkpoint {path.name} has no model-hash sidecar or compatible manifest "
            "entry; refusing an unpinned policy. Replacement RL training requires "
            "explicit operator authorization"
        )


@lru_cache(maxsize=1)
def get_model_release() -> ModelRelease:
    """Return the process-wide validated release."""
    return ModelRelease()
