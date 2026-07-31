"""PUBLIC REPO GUARD: no robot model values may be committed under vault/.

WHY THIS EXISTS
---------------
On 2026-07-28 the controller lock (MODEL_INPUTS.lock.json, 464 lines: mass, track, wheel
radius, CoM height, roll constraints, static wheel loads, the full ODD contract) and a
19 MB certified safety grid were committed to this PUBLIC repository. They were removed
the next day by the SANITIZE commit -- but a removal is a new commit, not an erasure, so
both blobs stayed reachable in the history of a pushed branch and were still retrievable
by SHA through the GitHub API three days later.

The rule was already known and written down when that happened. What was missing was
anything that ENFORCES it. `.gitignore` listed three exact paths, which stops those three
filenames and nothing else -- a differently-named file carrying the same values passes.

This test is the enforcement. It reads the git INDEX (tracked files only), so it fails at
the point the value is staged, not after it reaches a remote.

THE CONTRACT IT PROTECTS
------------------------
SSB owns the METHOD. vault-controller owns the ROBOT.
This repo holds machinery plus a sha256 DIGEST of the controller lock; the values
themselves are read at runtime from the private sibling via VAULT_CONTROLLER_ROOT.
A byte-identical lock is exactly what the digest asserts, and a digest reveals nothing.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

_VAULT = Path(__file__).resolve().parent.parent
_REPO = _VAULT.parent

# Artifacts that carry model values by construction. Sidecars (.model.json) are hashes.
_BANNED_SUFFIXES = {".npz", ".npy", ".pt", ".pth", ".pkl", ".onnx", ".stl", ".urdf"}

# Distinctive keys of MODEL_INPUTS.lock.json and the generated model chain.
#
# IMPORTANT DISTINCTION, and the reason this check is narrow: naming one of these keys is
# NOT a disclosure. `config.py` does roll["static_wheel_normal_loads_n"][...] and
# `dynamics.py`/`relock.py` reference "composite_params" by name -- that is the contract
# working as designed, reading the private lock at runtime. What must never be committed
# is the key bound to a NUMBER. So this check looks only at tracked DATA files (.json),
# and only flags a key that carries a numeric value. Code that merely names a key passes.
_LOCK_KEYS = (
    # top-level sections of MODEL_INPUTS.lock.json, verified against the real leaked blob
    # (f251e45^): 61 numeric leaves including controller_limits.motor_torque_limit,
    # theta_max, wheel_speed_limit, geometry_contract widths and the int8 deployment gates.
    "full_mass_properties_and_roll_constraint",
    "geometry_contract",
    "controller_limits",
    "int8_deployment",
    "reduced_generation",
    "source_urdfs",
)
# The distinction is the FILE TYPE, not the syntax around the key.
#   *.py   -- naming a key is a runtime lookup. Allowed, and REQUIRED by the contract:
#             config.py does odd_contract["grid_axes"][...] to read the private lock.
#   *.json -- a data file does not "reference" a key, it CONTAINS it. Disclosure.
# Two earlier drafts of this guard were wrong and both are pinned by self-tests below:
#   (1) it scanned *.py too, and flagged config.py/dynamics.py/relock.py -- the very
#       mechanism that keeps values OUT of this repo;
#   (2) it required `"key": <number>`, which MISSED the real leak because these keys map
#       to nested OBJECTS. A gate that passes on the known disclosure is decoration.
_LOCK_KEY_RE = re.compile(r"\"(" + "|".join(_LOCK_KEYS) + r")\"")
_SIDECAR_SUFFIX = ".model.json"

# These names must resolve through config.__getattr__ at runtime. Assigning any of them a
# numeric literal in this repo means the value has been inlined.
_RELEASE_NAMES = (
    "MASS", "TRACK", "WHEEL_R", "COM_H_WHOLE", "A_TIP", "N_STATIC_MIN",
    "C_THETA", "YAW_K0", "YAW_KC", "YAW_KV", "YAW_EPS", "CONTROLLER_TAU_MAX",
    "THETA_MAX", "TAU_MAX", "EBAR_PSI", "TAU_ROLL_BAR",
)
_ASSIGN = re.compile(
    r"^\s*(" + "|".join(_RELEASE_NAMES) + r")\s*(?::[^=]+)?=\s*[-+]?\d+\.?\d*",
    re.MULTILINE,
)


def _tracked_under_vault() -> list[Path]:
    out = subprocess.run(
        ["git", "-C", str(_REPO), "ls-files", "vault/"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    return [_REPO / p for p in out]


def test_no_model_artifacts_tracked() -> None:
    """No binary/model artifact may be tracked under vault/."""
    bad = [
        p for p in _tracked_under_vault()
        if p.suffix.lower() in _BANNED_SUFFIXES
    ]
    assert not bad, (
        "model artifacts committed to a PUBLIC repo:\n  "
        + "\n  ".join(str(p.relative_to(_REPO)) for p in bad)
        + "\nThese belong in the private vault-controller sibling. Track a sha256 "
          "sidecar instead (see vault/data/*.model.json)."
    )


def test_no_lock_values_in_tracked_data() -> None:
    """No tracked data file may bind a lock key to a NUMBER (naming a key is fine)."""
    offenders: list[str] = []
    for path in _tracked_under_vault():
        if path.suffix.lower() != ".json" or path.name.endswith(_SIDECAR_SUFFIX):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        hits = sorted({m.group(1) for m in _LOCK_KEY_RE.finditer(text)})
        if hits:
            offenders.append(f"{path.relative_to(_REPO)}: {', '.join(hits)}")
    assert not offenders, (
        "private model VALUES found in tracked data of a PUBLIC repo:\n  "
        + "\n  ".join(offenders)
        + "\nThese belong in the private controller lock, read at runtime through "
          "vault.config. Track a sha256 sidecar here instead."
    )


def test_release_values_are_not_inlined() -> None:
    """Release values must resolve via config.__getattr__, never be literals here."""
    offenders: list[str] = []
    for path in _tracked_under_vault():
        if path.suffix != ".py" or path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for m in _ASSIGN.finditer(text):
            line = text[: m.start()].count("\n") + 1
            offenders.append(f"{path.relative_to(_REPO)}:{line}: {m.group(0).strip()}")
    assert not offenders, (
        "robot model values inlined in a PUBLIC repo:\n  "
        + "\n  ".join(offenders)
        + "\nThese names are resolved lazily by vault/config.py from the private "
          "controller lock; assigning them here defeats that."
    )


def test_lock_pin_is_a_digest_not_the_lock() -> None:
    """The tracked pin must be a bare sha256, never the lock's contents."""
    pin = _VAULT / "data" / "controller_lock.sha256"
    assert pin.exists(), f"missing lock pin: {pin}"
    first = pin.read_text().split()[0].strip().lower()
    assert re.fullmatch(r"[0-9a-f]{64}", first), (
        f"{pin} must hold a bare sha256 digest; found {first[:80]!r}. "
        "The lock's semantic fields carry model values -- track the digest only."
    )


def test_guard_actually_rejects(tmp_path: Path) -> None:
    """A gate that cannot demonstrate it rejects is decoration."""
    assert _ASSIGN.search("MASS = 41.5\n"), "should reject an inlined float"
    assert _ASSIGN.search("TAU_MAX: float = 8\n"), "should reject an annotated literal"
    assert not _ASSIGN.search('"MASS",\n'), "should ignore a name listed as a string"
    assert not _ASSIGN.search("MASS = C.MASS\n"), "should allow resolution via config"
    assert Path("x.npz").suffix in _BANNED_SUFFIXES
    assert Path("x.model.json").suffix not in _BANNED_SUFFIXES, "sidecars are hashes, allowed"

    # The reference-vs-value distinction this guard turns on. Getting it backwards is how
    # the first draft flagged config.py/dynamics.py/relock.py -- the very mechanism that
    # keeps values OUT of this repo.
    # A nested object, which is the shape in the REAL lock and which the first draft missed.
    assert _LOCK_KEY_RE.search('"controller_limits": {"motor_torque_limit": 20.0}'), \
        "must reject the nested-object shape -- this is what the real leak looked like"
    # NOTE the actual mechanism: the pattern matches a quoted key in EITHER file type.
    # What makes a .py runtime lookup safe is the extension filter in
    # test_no_lock_values_in_tracked_data, which scans *.json only. Asserting the pattern
    # itself distinguishes them would be wrong -- and was, in a previous draft of this test.
    assert _LOCK_KEY_RE.search('cfg["controller_limits"]'), \
        "pattern matches the key regardless of file type -- the .json filter is the guard"


def test_guard_catches_the_real_leak() -> None:
    """Pin the guard against the ACTUAL 2026-07-28 disclosure, not a synthetic one."""
    import subprocess
    blob = subprocess.run(
        ["git", "-C", str(_REPO), "show", "f251e45^:vault/data/MODEL_INPUTS.lock.json"],
        capture_output=True, text=True)
    if blob.returncode != 0:
        pytest.skip("pre-sanitize commit not present in this checkout")
    assert _LOCK_KEY_RE.search(blob.stdout), (
        "the guard does not detect the real MODEL_INPUTS.lock.json that was committed to "
        "this public repo on 2026-07-28 -- it would not have prevented the incident")
