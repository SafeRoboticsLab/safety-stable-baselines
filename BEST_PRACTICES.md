# Best practices for margin-based safety training

Field rules distilled from training Go2 locomotion/gap-crossing safety policies at
scale (thousands of GPU envs) with this library. Each one was learned by watching a
run fail in a specific, diagnosable way; the failure signatures are included so you
can recognize them in your own curves.

## 1. Margin design

**The reward IS the margin — never normalize it.** `VecNormalize(norm_reward=True)`
(or any running reward scaler) silently destroys the backup: the value function's
zero level set is the safety boundary, and rescaling moves it. Observation
normalization is fine and recommended.

**The l/g magnitude ratio is a risk-tolerance parameter — set it deliberately.**
In the reach-avoid backup, a committed maneuver with success probability `p` is
preferred over stopping roughly when `p·l_bank > (1−p)·|g_fail|`, i.e. the break-even
attempt probability is `p* = |g|/(|g| + l)`. With `l` clamped to ±1 against `g`
anchored at ±3, `p* = 75%`: any maneuver that succeeds less than 75% of the time is
*correctly* abandoned by the optimizer, and the policy drifts to never-attempting
while retaining the skill. Most formulations set this dial by accident. Signature:
task-completion rate decays smoothly over training while the safety rate rises and
the skill remains demonstrable from pinned states.

> **Caveat — this `p*` needs re-deriving.** The break-even above compares attempting
> against a *non-negative* stop baseline, which is what a `g`-anchored reach-avoid
> backup produces. Under the correct `min(l, g)` anchor, loitering off-target is worth
> `l < 0`, so attempting only has to beat a negative baseline and the true `p*` is
> lower — possibly much lower. Treat the `75%` figure as an artifact of the old anchor
> until it is re-derived and re-measured.

**Check margin floors against your reset distribution's physics.** If a margin term
(e.g. a landing-impact limit) is violated by the *reset states themselves* (e.g.
spawn drops touch down at 3 m/s against a 2.2 m/s limit), that part of the state
space is condemned by construction and the curriculum anchor dies. Signature:
episode length collapses to a few steps at specific curriculum rungs; levels stall
near zero.

**Terminate the episode when `g < 0`.** The buffers anchor terminal steps at `g`
(avoid) or `min(l, g)` (reach-avoid); letting the sim keep running after a
violation leaks post-failure states into the value target.

## 2. Optimizer stability (on-policy fine-tuning)

**Cap the action std when fine-tuning a converged motor skill** —
`StdCapCallback(max_std=0.3–0.4)` plus `target_kl≈0.01`. Margin objectives have no
gait-quality gradient; PPO will inflate the std chasing a high-magnitude advantage
landscape and the exploration noise itself erodes the skill. Signature: `train/std`
climbing steadily (e.g. 0.3 → 0.8) while task success at fixed eval states decays.

**Never reset the log-std when warm-starting a fine-tune.** Std resets are for
*discovering new behavior* from a collapsed source. Applied to a converged policy
under a margins-only objective they destroy the skill in the first few million steps
and the safety objective rebuilds it at ~1%/40M. Signature: eval success crashes
immediately after the warm start, then rebuilds with a very shallow slope.

**Transfer the observation normalizer with the weights.** A warm start that loads
policy weights but reinitializes running obs statistics is not a warm start — the
policy sees differently-scaled inputs. Save/restore the normalizer state
(`TensorVecNormalize` state dict / `vecnormalize.pkl`) alongside the checkpoint.

## 3. Curricula and evaluation

**Curriculum promotion predicates must be the composed task success.** Promoting on
any proxy (episode timeout, partial progress) gets exploited: a policy that stands
still for the full episode "times out" and climbs the ladder. Signature: curriculum
level rises while an independent frozen benchmark stays flat — always keep such a
benchmark (a fixed bank of held-out initial states, evaluated deterministically) as
the source of truth.

**Don't use a learned certificate as an optimization target.** A value/success
certificate that is reliable as an *evaluator* becomes unsound the moment a policy
is rewarded for entering its acceptance set — PPO finds adversarial corners of the
network (observed: 88% → 4% true-success-given-accepted). Use geometric/physical
conditions for shaping, keep the certificate as a gate, and pay large rewards only
on *realized* outcomes (an actual witness rollout succeeding).

**Mind per-step reward scaling in dt-scaled stacks.** mjlab/Isaac-style pipelines
multiply reward weights by the control dt (e.g. 0.02): a "bonus of 30" pays 0.6.
Event bonuses need weights ×(1/dt), or living costs will dominate and the optimizer
may prefer ending episodes early. Signature: mean episode length collapses while
the per-step reward is net-negative.

## 4. Reach-avoid specifics

**Pick the cell, don't bend the margins.** The learners are a 2×2 over
{avoid, reach-avoid} × {single, two-player} — see the README table. Avoid is
**not** a reach-avoid instance with a degenerate `l`: the reduction needs
`l ≥ g` (for the anchor) *and* `l ≤ V'` (for the recursion), and `V' ≤ g`, so it
demands `l ≥ g ≥ V' ≥ l`. A large negative `l` gives `V ≡ l` — an empty safe set
with healthy-looking metrics. A zero/positive `l` gives `V ≡ g` — no lookahead at
all, since `max(l, ·)` clips every negative future. **If your `l` exists only to
be ignored, you have an avoid task: use `SafetyPPO` or `IsaacsPPO`.**

**The reach-avoid anchor is `min(l, g)`, not `g`.** The `(1 − γ)` anchor is the
"terminate now" payoff, and reach-avoid scores that well only if you are in the
target *and* safe. Anchoring on `g` — the *avoid* problem's anchor — makes "stay
safe forever, never reach" a fixed point at `V = g > 0`, a win, when its true
reach-avoid value is `maxₜ l(sₜ) < 0`. That is the **loiter optimum**: the policy
parks in a safe state, never attempts the task, and the critic calls it success.
If you write a custom buffer, keep the anchor `min(l, g)`; see the README table
for which anchor goes with which problem.

**Don't carry ISAACS's `g` anchor into a reach-avoid problem.** ISAACS is an
avoid game (no target set, no `l` in the paper at all), so `g` is right *there*.
Gameplay Filters extends ISAACS to reach-avoid and changes the anchor to
`min(l, g)`. The mixture — a `g` anchor with a `max(l, V')` recursion — appears
in none of the papers, and its fixed point is neither problem's value.

**`l` enters through the recursion, so target visits must be on-policy reachable.**
The reach term only banks value along trajectories that actually visit `l ≥ 0`. If
the target set is far outside the current policy's visitation, mix target-adjacent
states into the reset distribution (state banks) rather than hoping exploration
finds them.

**Avoid-only fine-tuning of a task-capable policy is a strong recipe on its own.**
The avoid gradient is near-zero on all safe episodes and sharp only at failures, so
it barely moves a competent policy except exactly where it dies — converting doomed
attempts into stopping while preserving successful ones. If your goal is "keep the
task, remove the deaths," try `SafetyPPO` fine-tuning before anything fancier.

## 5. Filters (deployment)

**Value-based shielding:** run the nominal; switch to the safety policy when the
safety value of the current state (or the nominal's proposed next state) drops
below a threshold. Two things matter in practice:

- **Latched vs. instantaneous switching:** un-latched filters chatter at the
  boundary; latched filters must clear the latch on episode reset.
- **Commitment:** past a maneuver-specific point, switching to the fallback is
  *worse* than continuing (a mid-flight stop command is a crash). Measure the
  committed region with live-switch tests (switch policies mid-episode and record
  outcomes) before trusting a filter's intervention rule — respawn-based
  state-restoration tests silently break observation-history buffers.

## 6. Reproducibility

- Log with fixed eval-state banks, not just rolling training stats.
- Record eval videos on a fixed cadence from the start — behavioral failure modes
  (idling, suicide, rocking) are obvious in ten seconds of video and invisible in
  scalars.
- Keep every experimental variant as a named config in code (registry entry), not
  as uncommitted local edits; failure-mode artifacts are worth archiving alongside
  successes.
