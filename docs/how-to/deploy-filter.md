# Deploy a value-based safety filter

**Goal.** Use a trained safety value function as a **runtime filter**: let a nominal
(task) controller drive, and hand control to the safety policy only when the nominal is
about to leave the certified-safe set.

**Prerequisites.** A trained safety or reach-avoid learner (its critic is the
certificate) and a nominal controller for the same environment.

## The principle

The learned value function is a certificate: `V(s) ≥ 0` marks states from which the
safety policy can stay safe (see
[Safety margins and certificates](../concepts/safety-margins.md)). A filter uses it as a
gate:

1. Run the nominal controller.
2. Look at the safety value of the **current state**, or of the **nominal's proposed next
   state**.
3. If it drops below a threshold, switch to the safety policy; otherwise let the nominal
   act.

With a SAC critic the value of a state under the safety policy is
`V(s) = min_i Q_i(s, π_safe(s))`. The SAC family gives the cleanest filters because that
value is calibrated over the action space, not just the on-policy tube — see the
[Bicycle5D value comparison](../environments/bicycle5d.md).

## Two things that matter in practice

- **Latched vs. instantaneous switching.** An un-latched filter *chatters* at the
  boundary, flipping between controllers every step. A latched filter holds the safety
  policy once engaged — but the latch **must be cleared on episode reset**.
- **Commitment.** Past a maneuver-specific point, switching to the fallback is *worse*
  than continuing — a mid-flight stop command can be a crash. Measure the committed
  region with **live-switch tests** (switch policies mid-episode and record outcomes)
  before trusting an intervention rule. Respawn-based state-restoration tests silently
  break observation-history buffers, so they do not measure this correctly.

## Do not optimize against a learned certificate

A value/success certificate that is reliable as an **evaluator** becomes unsound the
moment a policy is *rewarded* for entering its acceptance set — PPO will find adversarial
corners of the network (observed: 88% → 4% true-success-given-accepted). Keep the
certificate as a **gate**, use geometric/physical conditions for reward shaping, and pay
large rewards only on *realized* outcomes (an actual witness rollout succeeding). More in
[best practices → curricula and evaluation](../best-practices.md).

## Robustness: filter with a two-player certificate

A `*2P` critic is a **robust** certificate — its `{V ≥ 0}` set accounts for a worst-case
disturbance, so a filter built on it tolerates modeled disturbances a single-player
certificate would not. Train it with [an adversarial learner](train-adversarial.md); the
deployable control policy is `self.policy`.

## Related

- [Evaluate a policy](evaluate.md) — measure the certificate's soundness first.
- [Safety margins and certificates](../concepts/safety-margins.md) — what `V ≥ 0` guarantees.
- [Oracle validation](../validation-oracle.md) — how well the certificate matches ground truth.
