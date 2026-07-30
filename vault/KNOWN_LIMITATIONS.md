# Known limitations of the v2.2 consumer (read before trusting a demo run)

## The joystick/teleop demo has no v2.2-valid monitor checkpoint

`vault/teleop.py` is unmodified and its wiring works, but on the v2.2 release there
is no correct policy to give it. A partial run can easily look like a working demo,
so this is stated explicitly.

- The default monitor, `contact_safety_sac_v3`, is quarantined by
  `vault/data/checkpoints_manifest.json` as incompatible with this release. That
  quarantine is **correct and pre-existing**: it is a v2.0/v2.1-era contact policy
  that was never retrained against v2.2. `_dry_run` loads a policy
  unconditionally, so with the default model the demo does not start at all.
- The only checkpoint authorized under this release,
  `reach_avoid_safety_sac_v22`, is **not a substitute**:
  1. **Value semantics differ.** `ValueMonitor` (`vault/safety_filter.py`) reads the
     critic as an avoid-only safety value. `ReachAvoidSafetySAC`
     (`vault/reach_avoid_sac.py`) deliberately changes the backup to
     `V(x) = min(g(x), max(l(x), V(x_next)))`, so reaching the target set "cashes
     in" value. Those numbers do not mean the same thing. `teleop` loads via
     `load_safety_sac` (the parent class); `load_reach_avoid_safety_sac` exists but
     is unused here, and **nothing gates monitor class against checkpoint class**.
  2. It is **undertrained and evaluation-only** (300k steps; 61.7% reached / 17.3%
     limbo / 21.0% failed at mu=0.8). It has never been claimed as a control policy.

### Correction to commit 50d1c49's message

That commit reports `teleop --dry-run 150` completing with "margins positive
throughout". That validated the **control-loop wiring only** and overstates what was
shown. Two corrections:

- The run used `reach_avoid_safety_sac_v22` as the monitor, i.e. the semantic
  mismatch above. The resulting "filter overrode 100% of steps" is an **artifact of
  that mismatch** — it is not evidence the filter is working hard, nor that it is
  broken. Do not cite that number in either direction.
- The margins stayed positive largely because the robot barely moves in the run's
  1.5 s window (150 steps x 0.01 s). With the filter disabled the same scripted
  command only reaches about 0.05 m/s, so filtered-vs-unfiltered is a far smaller
  difference than "override on every step" suggests.

### What would fix it

Operator/maintainer decision — replacement RL training is authorized-only and was
not performed here. Either retrain a contact/avoid-only monitor against v2.2, or
point `teleop` at a monitor whose value semantics match its checkpoint. Either way,
gating the monitor/checkpoint pairing would stop this mismatch recurring silently.

Longer form, with the release-side context: `vault-controller`
`docs/ssb_handoff/CHANGES_TO_THE_MODEL_LAYER.md` section 15.

## Not a limitation here: the missing opt6 MVE/batch4 entries

The v2.2 released opt6 kernel lacks its Arm Helium (M55) vector entries. That is a
**firmware-side** gap and does not affect this repo: `vault/dynamics.py` binds only
the scalar `reduced_struct_fjac_f32` entry, whose numerics are unchanged, and
nothing here references `batch4`. Recorded so the question does not have to be
re-derived.
