# What changed underneath the reach-avoid work, and why

For Jaime. Everything here is in the **model layer** — the four-state plant, its
constants and its contracts. None of it changes the reach-avoid mathematics. The
purpose of this document is that you can *audit* these corrections rather than
discover them from a diff, because several of them move results without moving a
line of your code.

Baseline for the comparison is `2c4f73d` (2026-07-24).

## The short version

The four-state model handed over in `4340e3b` contained a frame error in the
no-liftoff roll bound, repeated in five independent places including inside the
release verifier. It is corrected. The ODD velocity bound was widened from ±4 to
±6 by operator ruling. Both change the certified set; neither changes an operator.

## 1. The roll bound was computed from the wrong height

`a_tip` is the lateral acceleration at which a wheel lifts, from a static moment
balance about the contact line. The handed-over value used `h_cm + wheel_radius`
as the whole-robot centre-of-mass height above ground.

That is a **frame mismatch**, not a wrong number. `h_cm` is a correct height — of
the **sprung body**, above the **axle**. Both wheels sit at `z = 0` in the URDF, so
`sum(m_i z_i)` over all links equals the sprung sum and `h_cm = z_sprung` exactly.
Adding the wheel radius to it mixes the sprung body's axle-referenced height with
the whole robot's axle-to-ground offset, producing a height belonging to neither
body. The wheels pull the whole-robot centre of mass *below* the sprung one:

    sprung CoM above axle     h_cm       0.069706 m
    whole CoM above axle      z_whole    0.061509 m     (h_cm is 13.3% higher)
    whole CoM above ground    z + r_w    0.188559 m     <- what rollover needs
    the mixed-frame value     h_cm + r_w 0.196756 m     (4.3% high)

It survived in five places because it was silently CONSERVATIVE — an over-tall
moment arm understates `a_tip` — and because the type docstring reads "CoM height
above wheel axle" without saying whose CoM. Every author read it correctly and
added the wheel radius to reach the ground.

Also missing: the v2.2 centre of mass is 7.2764 mm off-centre laterally, so the
symmetric bound `g*W/(2H)` overstates the true worst-direction limit.

    a_tip   6.998922  ->  6.924595     (symmetric worst case, corrected height)

An earlier internal estimate put the optimism at 5.47%. That was wrong — it mixed
the sprung-equivalent length with a geometric height. The correct figure against
the worst turn direction is **1.07%**.

`models/generated/safety/roll_constraint_params.json` in vault-controller is now
the single generated source, schema v2, including an explicit sign convention
derived from the URDF joint origins and asserted in a gate.

**Verification.** A bilateral MuJoCo measurement (`vault/tools/measure_rollover_bilateral.py`)
confirms the asymmetry from two independent observables: the static per-wheel load
split matches the prediction to 0.1% and needs no tipping at all, and the
directional ratio matches rim-credit geometry to 0.099%. Both measured thresholds
sit above the shipped bound, so the shipped bound is conservative against the plant.

## 2. Wheel loads: conservative lower value on both wheels

`wheel_loads()` used an even static split. The true split is 101.80/91.77 N because
of the lateral offset. Assigning those to left and right requires a sign convention
that was not released at the time, and reversing it would credit the lighter wheel
with the heavier load, so both wheels now take the **lower** value, 91.766 N.

This understates available friction, so the friction clip in `f_cert_step` is
tighter and the ego has less recovery authority. It costs certified volume, most at
low `mu` where the clip binds hardest. It is a strict tightening.

The sign convention has since been released and gated, so the true split is now
recoverable — the right wheel's load is already exact and only the left is
understated, by 10.033 N. Not yet applied; it needs a C/Python parity test and a
re-solve.

## 3. The ODD velocity bound widened from ±4 to ±6

Operator ruling, 2026-07-28. This is the change most likely to affect your results.

`odd_margin` builds `g_v` from `odd.velocity`, so widening increases `g` pointwise,
and the solver's operator

    V_{k+1}(x) = min( g(x),  max_u min_d V_k(f(x,u)+d) )

is monotone in `g`. Therefore `V_new >= V_old` pointwise and
`{V_old >= 0} subset {V_new >= 0}`: **everything previously certified stays
certified**, and the set grows.

Be clear what it is: a **relaxation of the constraint set**. The physical failure
terms `g_pitch` and `g_roll` are untouched; only the declared-region terms moved.
The guarantee changes from "avoid topple, liftoff, and leaving ±4" to "…leaving ±6".

Why it was necessary: the deployed Layer-1 speed ceiling is 5.5 m/s, which exceeded
the declared ±4 ODD. Commands sat outside the region the contract described. The
ceiling was itself derived as "certified extent minus margin" when the working
envelope was ~6 m/s, with the derivation recorded only in a comment, so it silently
decoupled when the envelope changed.

**A structural note you may care about more than the number.** `odd_margin`
conflates *physical failure* (`g_pitch`, `g_roll`) with *ODD exit* (`g_v`, `g_psi`)
in one `min`. Since the four-state dynamics have no `v`-dependence, `g_v` is the
only thing bounding the certified set in velocity — so the velocity extent of the
reach-avoid certificate is a **policy artifact, not a dynamic result**. If it would
help the mathematics to separate those, a `failure_margin()` alongside the existing
`odd_margin()` is straightforward and non-breaking; say the word.

## 4. Changes inside your files, all non-semantic

    reach_avoid_sac.py     63 lines
    train_reach_avoid.py   16
    safety_filter.py        8
    reach_avoid_eval.py     6
    reach_avoid_slice.py    4

The only one worth checking is `reach_avoid_sac.py`. The DRABE backup was extracted
into a pure function `_drabe_target_torch(gs, ls, next_q, dones, gamma)` so it can
be unit-tested; it was verified **bit-identical** to the inline form, max |diff| =
0.0 over 200 random 64-batches. Your caveat text is preserved verbatim. The rest is
a docstring and one audit comment. **No operator of yours was altered.** The
`_worst_next_value` CBF-QP core, the value iteration and the taxonomy modules are
untouched; the only `filter.py` changes are added constructors.

## 5. A defect we found and deliberately did not fix

The maximum-entropy bonus is added to the continuation value **inside** the
discounted reach-avoid backup. That term is not part of the operator, and it is
optimistic wherever `-log pi > 0`, so the under-approximation guarantee the operator
is chosen for does not transfer to the learned critic.

On a self-looping state with constraint margin `G = +0.5` and target margin
`L = -0.5`, whose true undiscounted value is `-0.5`, write `beta` for the per-step
additive bonus actually applied to the continuation (`beta = alpha * (-log pi)`, so
`beta`, not `alpha`, is the quantity that matters). The contaminated fixed point
crosses zero at

    beta* = |L| (1 - gamma) / gamma

i.e. `5.0e-4` at the shipped `gamma = 0.999`. The structure is adverse: raising
`gamma` to tighten the approximation *lowers* the bar for breaking it.

It is inherited from the avoid-only backup in `safety_sb3`, so it is not specific to
the reach-avoid extension. We recorded it in the code and left it alone — fixing the
operator is your call, not ours. No result we ship depends on a learned critic.

## 6. Provenance, so you can check any of this

Every model quantity now comes from a generated, locked artifact rather than a
literal. `vault/tools/relock.py` re-pins this repo to the controller release in one
pass and reports which artifacts still record an older release rather than
re-stamping them.

In vault-controller:

    models/tools/verify_model_release.py      28 gates
    models/tools/check_bound_consistency.py   K subset O subset D, K <= P
    models/tools/check_frame_hygiene.py       rejects mixed-frame height expressions
    models/ROLL_CONSTRAINT_REGENERATION_20260728.md   the full account

## 7. What is NOT established

- **No hardware witness.** Everything above is simulation or source-level. The only
  real-robot fact this model has is total mass, 19.731467 kg.
- **Roll has no dynamic state.** The bound is a quasi-static bifurcation criterion
  imposed on a four-state model that cannot represent roll dynamics. There is no
  error bound against dynamic liftoff under transient roll excitation, in either
  direction. No constant fixes this; it needs a roll degree of freedom.
- **The lateral offset is nominal and unperturbed** in the robust solves; its
  variation is unmeasured.
- **`dh = 0.30`** is an unvalidated height-uncertainty assumption, not payload
  geometry. A real laden envelope needs a declared payload mass-properties artifact.
- **Conservatism is exhaustive at grid nodes**, which is not a continuum proof, for
  both the float and int8 networks.
- **The robot does not run this.** The flashed firmware carries a v2.1 lineage whose
  dynamics kernel is reconstructible from no source we have.

---

# Addendum, 2026-07-29

Everything above still stands. This records what changed since, and four findings
from a literature review aimed at the limitations in §7. Nothing here touches your
files — `reach_avoid_sac.py`, `reach_avoid_value.py`, `reach_avoid_eval.py`,
`reach_avoid_slice.py`, `train_reach_avoid.py`, `safety_filter.py` and
`tests/test_drabe_operator.py` are unmodified.

## 8. Your code runs on v2.2 unchanged

`train_reach_avoid.py` trained 300k steps on the corrected release with no edits;
`test_drabe_operator.py` passes 7/7. `config.py` reads `THETA_MAX`, `V_ODD`,
`PSI_ODD` and the domains straight out of `odd_contract.json`, so the corrected
`a_tip` and the ±6 ruling reached your trainer by construction.

The checkpoint is authorized in `checkpoints_manifest.json` as **evaluation only** —
see §11.

## 9. The speed governor no longer folds its inputs

`bc_safety_v_governor` (and its Python mirror in `joystick_demo.v_governor`)
evaluated the value function at `(|v|, 0, 0, |ψ̇|)`. That asserts a symmetry the
v2.2 plant does not have: `y_cm = 7.2764 mm ≠ 0` makes turn direction asymmetric,
and even a joint state reversal fails because the shifted roll condition
`|H·a_y − g·y_cm|` carries a term that does not flip sign.

Measured over 40,000 states per µ at reserve 0.35, it erred in **both** directions:
316 states admitted below the required reserve, 510 needlessly throttled despite
genuine margin. Now evaluated at the signed state, with only the speed magnitude
bisected. This is a correction toward the certificate, not a relaxation of it — but
it does raise governed speed in some conditions.

## 10. Four limitations, better characterised than in §7

**(a) Early stopping is anti-conservative.** Our VI starts at `V₀ = g` and is
monotone non-increasing, so every iterate has `V_k ≥ V_∞`: stopping at `dV < 1e-3`
ships a set *larger* than the discrete fixed point, compounding interpolation error
rather than opposing it. Undiscounted VI admits no residual bound — the usual
`γδ/(1−γ)` correction is infinite at `γ = 1`. Fix is to iterate to the exact
floating-point fixed point. **Not yet applied; it changes every grid.**

**(b) `min_d` is taken over the CORNERS of the disturbance box.** Restricting the
adversary is the optimistic direction. Measured against a denser sampling, 3 states
of 1,161,508 left the safe set — a lower bound twice over, since that was one
operator application and the denser sampler was itself approximate. The error is
second-order (`V(f+d) ≈ V(f) + ∇V·d` is exact to first order in `d`), which is why
it is small. Note this also breaks an assumption of the level-set theorems: Hirsch
et al. (arXiv:2607.17435, 2026) require `f(x,a,𝓑)` convex, which a finite vertex set
is not. **Not yet applied.**

**(c) The roll constraint's real exposure is cross-slope, not transients.** Our
`|v·ψ̇| ≤ a_tip` is literally the NHTSA Static Stability Factor. The automotive
"quasi-static is ~15% optimistic" literature concerns suspension compliance, which
we do not have. What the criterion is *silent* about is terrain-injected roll
energy — the mechanism behind ~95% of real rollovers. At a 2.9° cross slope (5%,
routine on driveway aprons and ADA curb-ramp flares) plus the CoM offset, effective
`a_tip` drops **11.3%** — more than our entire measured transient overshoot, and
*sustained*, which is the regime that actually topples.

Also: `I_roll` in the validation plant is a placeholder chosen to satisfy the
principal-inertia triangle inequality. Correct for balance dynamics, and exactly
wrong for roll certification — roll excursion scales as `1/I_φ`, so every
roll-margin number rests on an unidentified inertia. `I_φ` has never been measured.

**(d) A second unbounded gap we had not named: the inter-sample gap.** Our Bellman
equation is discrete-time, but safety must hold *between* ticks; the state moves up
to `M·Δt`. This is orthogonal to grid resolution and not covered by "exhaustive at
grid nodes."

One point in our favour, worth stating because reviewers conflate it: multilinear
interpolation is an *averager* (Gordon, ICML 1995) — non-negative weights summing to
one, hence non-expansive. Our scheme **is** monotone in the Barles–Souganidis sense
and converges to the viscosity solution. Monotone ≠ conservative; both are true and
they are different claims.

## 11. On the entropy term — measured, not just argued

§5 recorded that the max-entropy bonus sits inside the discounted reach-avoid
backup. Two things to add.

**Your lab's own SAC reach-avoid code already excludes it.**
`SafeRoboticsLab/ISAACS`, `agent/base_block.py:406`, applies `entropy_motives` only
`if self.mode == 'performance'` — the safety, risk and reach-avoid critics are
entropy-free. So removing it aligns with your reference implementation rather than
diverging from it. Relatedly, the RSS 2021 published code is tabular Q-learning and
DDQN; there is no SAC, so Theorem 1 was never proven for a soft backup.

**Measured on our v2.2 checkpoint.** `α` auto-tuned to 1.4e-4, and
`β = α·(−log π)` came out **negative in 99.3%** of sampled states — for
tanh-squashed Gaussians the density exceeds 1 near action saturation, so `log π > 0`.
The contamination therefore ran *conservative* here, and the critic shows no
saturation: V spans −2.52 to +0.12 with IQR 0.85. `|β|` still exceeds
`β* = |L|(1−γ)/γ = 5.005e-4` across 83% of the domain.

So the defect is real and its magnitude is above threshold; what saved this run is
the *sign*, which is an emergent property of where the policy converged rather than
a guarantee. It remains yours to fix or leave; we have not touched it.

**If the deployed monitor becomes rollout-based**, one caution: the terminal set Ω
must come from the grid oracle or a hand-certified at-rest set. If Ω is the learned
critic's zero level set, the contamination is relocated behind H steps of simulation
and made harder to see, not smaller.
