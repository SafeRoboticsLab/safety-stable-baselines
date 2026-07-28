"""Re-pin this repository to the vault-controller model release, in one pass.

Every recorded hash is recomputed from disk. Doing this by hand does not work: the
lock, the release sidecar and the per-artifact `.model.json` sidecars all record
overlapping digests, so patching one leaves another stale and the failure surfaces
as a mismatch in a different file than the one that was edited.

There are two kinds of record here and they must not be treated alike.

WHAT RELEASE WE ARE PINNED TO -- rewritten by this tool:

  * `vault/data/MODEL_INPUTS.lock.json` -- copied byte-for-byte from the controller
    checkout. The consumer requires byte equality, not semantic equality.
  * `vault/data/release_artifacts.model.json` -- lock digest, the model hashes the
    consumer re-derives from the lock, and the artifact role map.
  * `release_lock_sha256` in `vault/data/checkpoints_manifest.json`.

WHAT RELEASE AN ARTIFACT WAS PRODUCED UNDER -- reported, never rewritten:

  * per-checkpoint `lock_sha256` entries in the checkpoint manifest;
  * the `.model.json` sidecars beside generated grids and eval artifacts.

Restamping those would assert that a grid solved against the previous bound, or a
policy trained against it, is valid under the new one. They are supposed to mismatch
until the artifact is actually regenerated -- that mismatch is what quarantines a
stale checkpoint and what makes a stale grid fail its gate. This tool lists them so
the regeneration backlog is visible, and leaves them alone.

The artifact role map is taken from `model_release._ARTIFACTS`, so adding a role
there is all that is needed to bring a new generated artifact under the lock.

Run:
    PYTHONPATH=$PWD:../vault-controller python vault/tools/relock.py [--check]

`--check` reports what would change and exits non-zero if anything would, which is
what CI should call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from vault.model_release import _ARTIFACTS  # noqa: E402

LOCAL_LOCK = REPO / "vault/data/MODEL_INPUTS.lock.json"
SIDECAR = REPO / "vault/data/release_artifacts.model.json"
MANIFEST = REPO / "vault/data/checkpoints_manifest.json"
DATA = REPO / "vault/data"
MODEL_HASH_ROLES = {
    "composite_params_sha256": "composite_params",
    "opt6_kernel_sha256": "opt6_kernel",
}


def controller_root() -> Path:
    import os
    configured = os.environ.get("VAULT_CONTROLLER_ROOT")
    root = Path(configured).expanduser() if configured else REPO.parent / "vault-controller"
    if not root.is_dir():
        raise SystemExit(f"vault-controller checkout not found: {root}")
    return root.resolve()


def build_sidecar(lock_bytes: bytes, lock: dict) -> dict:
    release_files = lock["release_files"]
    missing = [r for r in _ARTIFACTS.values() if r not in release_files]
    if missing:
        raise SystemExit(
            "controller lock does not cover these artifact roles:\n  "
            + "\n  ".join(missing))
    return {
        "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "model_hashes": {
            key: release_files[_ARTIFACTS[role]]
            for key, role in MODEL_HASH_ROLES.items()
        },
        "artifacts": dict(_ARTIFACTS),
    }


def stale_provenance(lock_sha: str) -> list[str]:
    """Artifacts still recording an older release. Regeneration backlog, not drift."""
    stale = []
    for path in sorted(DATA.glob("*.model.json")):
        if path == SIDECAR:
            continue
        try:
            recorded = json.loads(path.read_text()).get("lock_sha256")
        except (OSError, json.JSONDecodeError):
            continue
        if recorded != lock_sha:
            stale.append(path.name)
    if MANIFEST.is_file():
        manifest = json.loads(MANIFEST.read_text())
        for entry in manifest.get("checkpoints", []):
            # Only "compatible" entries assert a checkpoint is valid for a given
            # release. The rest are quarantine rules keyed by filename pattern; they
            # carry no lock of their own and are not artifacts.
            if entry.get("status") != "compatible":
                continue
            if entry.get("lock_sha256") != lock_sha:
                stale.append(f"checkpoint: {entry.get('pattern', '?')}")
    return stale


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="report drift and exit non-zero instead of rewriting")
    args = parser.parse_args()

    source_lock = controller_root() / "models" / "MODEL_INPUTS.lock.json"
    lock_bytes = source_lock.read_bytes()
    sidecar = build_sidecar(lock_bytes, json.loads(lock_bytes))
    lock_sha = sidecar["lock_sha256"]

    current_lock = LOCAL_LOCK.read_bytes() if LOCAL_LOCK.is_file() else b""
    current_sidecar = json.loads(SIDECAR.read_text()) if SIDECAR.is_file() else {}
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {}
    lock_drift = current_lock != lock_bytes
    sidecar_drift = current_sidecar != sidecar
    manifest_drift = manifest.get("release_lock_sha256") != lock_sha

    if args.check:
        for label, drifted in (("MODEL_INPUTS.lock.json", lock_drift),
                               ("release_artifacts.model.json", sidecar_drift),
                               ("checkpoints_manifest.json", manifest_drift)):
            print(f"  {'DRIFT' if drifted else 'ok   '}  {label}")
        if lock_drift or sidecar_drift or manifest_drift:
            print("\nrun without --check to re-pin", file=sys.stderr)
            return 1
        print("\nin sync with the controller release")
        return 0

    shutil.copyfile(source_lock, LOCAL_LOCK)
    SIDECAR.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    print(f"lock      <- {source_lock}")
    print(f"           sha256 {lock_sha}")
    print(f"sidecar   <- {len(sidecar['artifacts'])} artifact roles")
    for key, value in sidecar["model_hashes"].items():
        print(f"           {key} {value[:16]}...")
    if manifest:
        manifest["release_lock_sha256"] = lock_sha
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print("manifest  <- release_lock_sha256 updated")

    stale = stale_provenance(lock_sha)
    if stale:
        print(f"\n{len(stale)} artifact(s) still record an older release. These are NOT")
        print("re-stamped: each must be regenerated before it can be trusted again.")
        for name in stale:
            print(f"  STALE  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
