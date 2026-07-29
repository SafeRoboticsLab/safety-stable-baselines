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

**(a) Early stopping is anti-conservative, and the residual is UNBOUNDED.** Our VI
starts at `V₀ = g` and is monotone non-increasing, so every iterate has `V_k ≥ V_∞`:
stopping at any positive `dV` ships a set *larger* than the discrete fixed point,
compounding interpolation error rather than opposing it.

We tried to close this by tightening and **it cannot be closed that way.** We measured
the convergence rate on the full laden μ=0.3 grid:

    iterations per decade of dV
       0 ->  200        63
     200 ->  600       759
     600 -> 1000     2,164
    1000 -> 1400     2,854
    1400 -> 1700     3,153      still climbing at iteration 1,900

The rate is not constant — it degrades monotonically. So there is no geometric tail to
sum, and **`dV` bounds nothing.** This is your own non-contraction result showing up
empirically: ICRA 2019 states the undiscounted safety operator "does not induce a
contraction mapping on V", and a non-contraction has no uniform rate to bound with.

We briefly set the tolerance to 1e-9 on the argument that the tail was bounded by
`dV·r/(1−r) ≈ 154·dV`. **That argument was wrong** and we are recording it because the
error is instructive: it used an early-window rate as if it were global. Reaching 1e-9
would have needed >17,000 iterations and still bounded nothing. The tolerance is back
at 1e-3.

Consequence for the certificate: the shipped safe sets are supersets of the discrete
fixed point by an amount we cannot quantify. Closing this needs a different
construction — verifying the shipped `V̂` directly as a candidate invariant, cell by
cell, which does not depend on VI having converged at all.

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

## 12. Three things you need from us before you can rule on any of it

### 12.1 Two certificate preconditions we satisfy by accident, not by argument

Our robust safety operator is **not a contraction** — your own ICRA 2019 paper says so
in as many words, and Akametalu et al. (arXiv:1809.00706) exhibit the failure for an
operator structurally identical to ours: any constant vector `α·1` is a fixed point
for `α < −L`. So there is a continuum of spurious fixed points, and value iteration
is not guaranteed to find the one we want.

Two conditions close that gap (Bertsekas, *Abstract Dynamic Programming* 3rd ed.,
Prop. 4.3.13–14 and §4.3.2), and our solvers satisfy **both** — but by construction
rather than by any stated intent:

1. **The control set is finite.** `UGRID` is 9 discrete torque pairs. Note the
   asymmetry with the disturbance: discretizing the *control* is what earns this
   guarantee, while discretizing the *disturbance* is what cost us §10(b).
2. **Value iteration is initialised at `V₀ = g`.** Every solver does `V = g.copy()`,
   which yields convergence to the *greatest* fixed point below `g`.

**We are recording these as preconditions because either would be silently destroyed
by an obvious optimisation.** A continuous or optimizer-based control search breaks
the first. Warm-starting VI from a previously-converged or learned `V` breaks the
second — and that is precisely the kind of speedup someone proposes to make a 25-hour
re-solve cheaper. If either changes, the fixed-point argument has to be redone.

### 12.2 A candidate terminal set Ω, with its provenance and its limits

If the deployed monitor becomes rollout-based, it needs a terminal safe set Ω that is
invariant under the fallback. We can supply one; we cannot rule on whether it
qualifies.

What we have: the avoid-only capability grid's safe set `{V ≥ 0}`, solved on the
±7 v / ±5 ψ̇ window under the corrected `a_tip`, exhaustively conservative at grid
nodes with zero false-safe nodes in the int8 gate.

What it is **not**: it is an **avoid-only** set. It certifies "never fails", not
"reaches a stop". It is therefore not a target set in the reach-avoid sense, and
using it as Ω asserts something we have not shown — that the fallback actually drives
the state into it and stays. It also carries every limitation in §10, including that
its own `min_d` was corrected only on 2026-07-29.

The reason we raise it at all: **if Ω is taken from the learned critic's zero level
set instead, the entropy contamination in §11 is relocated behind H steps of
simulation and made harder to detect, not smaller.** A grid-oracle Ω avoids that. The
choice is yours; we are flagging the trap, not making the call.

### 12.3 An asymmetry the contract can express and the consumer cannot

`config.py` reduces two of three ODD intervals to a single endpoint:

    THETA_MAX = odd_contract["odd"]["theta_failure"]["bounds"][1]   # drops bounds[0]
    PSI_ODD   = odd_contract["odd"]["yaw_rate"]["bounds"][1]        # drops bounds[0]
    V_ODD     = tuple(odd_contract["odd"]["velocity"]["bounds"])    # keeps both

All three are symmetric today, so there is no numerical error right now. The defect is
representational: **the contract can express an asymmetric bound that the consumer
structurally cannot carry**, and it would be silently half-applied.

This is not hypothetical for v2.2. The plant is genuinely asymmetric — `y_cm =
7.2764 mm` is what makes turn direction differ, what forced `A_TIP` to carry a
symmetric worst case, and what made the governor's `|ψ̇|` fold unsound (§9). An
asymmetric yaw bound is a plausible consequence of that asymmetry, and today it would
be accepted by the contract and quietly discarded by the consumer.

We have not fixed it: promoting these to intervals changes every downstream consumer,
and the right shape depends on whether the filter is ever intended to carry
direction-dependent bounds — which is your call, not ours.

## 13. The MuJoCo validation plant drops the lateral CoM offset

`vault-controller/models/generated/mujoco/balance/balance_plant.xml` places the sprung
body's centre of mass at `pos="0 0 0.0697058"` — **y = 0**. The generated mass
properties put the whole-robot CoM at `[0.000322, 0.007276, 0.061509]`, and since the
wheels are symmetric that lateral offset comes entirely from the sprung body, so the
plant should carry `y ≈ 0.00825`.

Why it matters here specifically: `y_cm ≠ 0` is the asymmetry that forces `A_TIP` to
carry a symmetric worst case, that made the governor's `|ψ̇|` fold unsound (§9), and
that produces the upright yaw→pitch coupling the reduced model exhibits. **A validation
plant with `y_cm = 0` cannot reproduce any of it** — it is a symmetric robot validating
an asymmetric model, and it will agree most convincingly exactly where the asymmetry
matters least.

Not fixed. It is a generated artifact, so the correction belongs in its generator, and
changing it invalidates existing validation runs.

**A related claim we could NOT confirm.** An earlier internal note recorded the plant's
roll inertia as "25.9% understated". The plant's `diaginertia` is `[0.538619, 0.252561,
0.538619]` against a generated whole-robot `inertia_at_com` of `[0.787347, 0.277148,
0.590612]`, which looks like a 31.6% shortfall — but the plant body is the **sprung**
body (17.411307 kg) and the artifact tensor is the **whole robot** (19.731467 kg).
Comparing them is the same frame error this document opens with, so we are not
asserting a percentage. Confirming it needs a sprung-body inertia tensor, which the
generated artifact does not separate out.

Note also that the plant uses the same value for `Ixx` and `Izz`, and that
`firmware/validation/plants/build_mujoco_model.py` picks roll inertia as
`max(chassis_yaw, |chassis_yaw − I_pitch| + 1e-3)` — a value chosen to satisfy the
principal-inertia triangle inequality, with a comment noting roll does not enter the
planar dynamics. That is correct for balance and tracking, and **exactly wrong for roll
certification**: roll excursion scales as `1/I_φ`, so every roll-margin figure rests on
an unidentified inertia. `I_φ` has never been measured.

---

# §14. Ownership, open items, and how to pick this up

This section is the handoff proper: who owns what, every open item with its evidence
pointer, and the mechanics your coding agent needs on day one.

## 14.1 The ownership rule (restating CODEOWNERS, with the boundary cases)

**Safety-filter mathematics is yours. The four-state model and its contracts are ours.**
Model/contract changes are announced to you before landing (this document is that
announcement); math-layer defects found by us are reported with analysis, never fixed.
We held that line: none of your seven files has been modified across any commit in
this release — verifiable with
`git diff <base>..HEAD -- $(grep @jfisac CODEOWNERS | awk '{print $1}')` → empty.

Two boundary cases resolved this week, recorded so the precedent is visible:
- The speed governor's `|v|,|ψ̇|` fold (§9): an *architecture component* failing for a
  *plant-assumption* reason. We fixed it, because the wrong assumption was about OUR
  plant (`y_cm ≠ 0`). Whether command-space governing is a sound intervention scheme
  at all remains YOUR question.
- The exact-`min_d` attempt (§10b): solver mathematics. We tried it, found the gap
  ourselves, and REVERTED rather than ship an operator asserted exact on an argument
  that does not close. The finding is yours to resolve.

## 14.2 Safety theory — all yours. Status of every row

| # | item | status | evidence |
|---|------|--------|----------|
| T1 | `min_d` over disturbance-box corners is optimistic | OPEN. Exact fix attempted, REVERTED — the `bad`-mask (−1.0 override on out-of-box / wheel-overspeed successors) is a jump discontinuity inside the disturbance interval that the affine-per-cell argument does not cover. 3/1,161,508 states left the set under denser sampling (lower bound). Also breaks Hirsch et al. 2026 convexity assumption on f(x,a,D). | §10(b); solver comment at the corner loop in `robust_capability_odd_solve.py`; revert commit explains the gap |
| T2 | Early-stopping residual is UNBOUNDED | OPEN, proven unclosable by iteration: iters/decade degrade 63→759→2,164→2,854→3,153 (laden μ=0.3, full grid). No geometric tail exists — your ICRA 2019 non-contraction, measured. A 154·dV bound was briefly believed and is retracted; the failed reasoning is preserved deliberately. | §10(a); solver comment above `CONVERGENCE_TOL` |
| T3 | Operator admits spurious constant fixed points | OPEN as a stated precondition: finite control set (9 torque pairs) + `V₀ = g` close it (Bertsekas AbstractDP Prop 4.3.13–14) and we satisfy both BY CONSTRUCTION. Warm-starting VI or a continuous control search silently forfeits it. | §12.1 |
| T4 | Inter-sample gap (`M·Δt` between ticks) | OPEN, unbounded, orthogonal to grid resolution | §10(d) |
| T5 | `{V ≥ 0}` vs `{V ≥ ε}` — boundary belongs to the losing side | OPEN, cheap | §10; Hirsch et al. 2026 Thm 1 |
| T6 | Node-exhaustive ≠ continuum proof | structural; the literature path is an interval cell-wise invariance certificate on the shipped V̂, which does not require VI to have converged | §10 |
| T7 | No finite-grid two-sided error bound exists for undiscounted Isaacs + multilinear interpolation | known limit of the field, cited, not a defect | §10 |
| T8 | Multilinear interpolation is an averager ⟹ monotone ⟹ converges to viscosity solution | IN OUR FAVOUR — but monotone ≠ conservative; both statements are true and different | §10 |
| T9 | Roll constraint quasi-static on a roll-free model | OPEN. Closed-form roll-energy allowance derived (ε·τ ≤ √(2·I_φ·U)/(m·H·a_tip)); blocked on measuring I_φ, which has never been measured — the validation plant's I_roll is a placeholder | §13 |
| T10 | Max-entropy term inside the DRABE backup | OPEN, yours (inherited from safety_sb3). β* = \|L\|(1−γ)/γ = 5.005e-4 at γ=0.999, saturation at 2β*. Your ISAACS code already excludes it (`base_block.py:406`). Measured on our checkpoint: α→1.4e-4, β<0 in 99.3% of states — conservative by SIGN LUCK, not guarantee | §11; `tests/test_drabe_operator.py` has the contraction/nesting tests; the §2 trap makes a good regression test |
| T11 | D-hat margin statistics (1.25× fails resampling at n=50) | OPEN, untouched | internal task record |

## 14.3 Filter architecture — split ownership

| # | item | owner | status |
|---|------|-------|--------|
| A1 | avoid-only vs reach-avoid objective | operator | DECIDED: reach-avoid deploys; avoid-only grid parked as baseline (`AVOID_ONLY_GRID_BASELINE.md` in vault-controller) |
| A2 | value-based vs rollout monitor | **you — highest leverage** | OPEN. Decides whether T10 is in the deployed path at all: rollout monitor with a certified Ω escapes it by construction (RSS 2021: FSR "exactly 0"); value monitor inherits it |
| A3 | where Ω comes from | **you** | OPEN. Candidate supplied with disqualifiers stated (§12.2): our grid is avoid-only, so it certifies "never fails", not "reaches". Ω from the learned critic RELOCATES the contamination behind H steps of simulation |
| A4 | deployed representation | operator | DECIDED: float64 grid-direct on Orin. No grid-direct evaluator exists yet — the C path consumes MLP weights; the lossless path exists only in Python `ValueFilter` |
| A5 | locked release artifact | ours | DECIDED: the capability grid (±7/±5 window) |
| A6 | governor plant assumption | ours | FIXED: signed-state evaluation; property tests replace the pinned witness |
| A7 | is command-space governing a sound intervention scheme | **you** | OPEN |
| A8 | `bounds[0]` discard — contract can express asymmetry the consumer cannot carry | **you** (shape) | OPEN, documented §12.3. Not hypothetical: the plant is asymmetric (`y_cm ≠ 0`) |
| A9 | is any value-function approximation certifiable (δ-shift, coverage/retention) | **you** | OPEN — applies to the distilled net AND to grid interpolation; Lin & Bansal L4DC 2024 is the closest machinery, and it calibrates against the critic's own policy, so T10 must be fixed first |

## 14.4 Mechanics for your coding agent

Environment:

    conda env: safe-sb3   (SB3 2.8.0, torch 2.12.0 CPU, mujoco 3.9.0; NOT isaacs)
    export VAULT_CONTROLLER_ROOT=~/Documents/vault-controller
    # the model-release gate refuses to load anything if the lock is not byte-identical

Verification battery (all green as of this handoff):

    (SSB)        python -m pytest vault/tests -q                          # 18
    (controller) python models/tools/verify_model_release.py             # 28 gates, exit 0
                 python models/tools/check_bound_consistency.py          # 9/9
                 python models/tools/check_frame_hygiene.py [extra-root] # both trees
                 balance_controller/c/test/*.py are STANDALONE scripts — pytest
                 collects 1 of 15 and prints a false green; run them individually
    (toolchain)  cd firmware && python -m pytest -q                      # 152 + 6 xfail
                 # safety_cert/ is in testpaths NOW; archive/ is excluded — a file
                 # there poisoned sys.path at import time and 16/18 tests silently
                 # validated v2.0 parameters. Do not re-add archive to collection.

Checkpoints: `reach_avoid_safety_sac_v22` is authorized EVALUATION-ONLY via an
explicit pinned manifest entry (lock + sha256). Everything else matching
`reach_avoid_safety_sac*` is quarantined. Retraining after you fix T10: the trainer
is `python -m vault.train_reach_avoid --steps 300000 --saturate-target` (~55 min at
~100 fps CPU); slices via `python -m vault.reach_avoid_slice --n 41 --horizon 300
--model reach_avoid=<ckpt>`. The 8 remaining corrected-solver grid re-solves were
deferred to your hardware (~3 h/slice at the current tolerance).

Figures: `vault/tools/make_v22_results_figure.py` regenerates the results PDF from
the committed slice JSON + pinned grids. `make_odd_safeset_figure.py` is pinned to
the older contract; do not reuse it for this domain.

Traps we hit so you do not: pytest's 1-of-15 false green (above); `pgrep | tail -1`
returning a wrapper shell instead of the python process; generated headers going
stale against the built `.so` (`export_v_mlp_c.py` does NOT rebuild — always
`make -C balance_controller/c` after any header export); relock does not touch
`int8_deployment.gates` (hand-maintained); a smoke solve used to overwrite release
grids (now suffixed `_SMOKE`).

## 14.5 What "satisfactory" means here, stated plainly

The model layer is corrected, locked, gated, and green. The reach-avoid result is a
PIPELINE validation (your code runs unmodified on the corrected model and produces a
sane outcome map), NOT a converged benchmark — 17.3% limbo says undertrained, and
the checkpoint carries T10. The avoid-only certificate is internally consistent and
its eight limits are stated with measurements. Nothing in this release claims a
continuum guarantee, a hardware witness beyond total mass, or a certified learned
policy. Where we tried to close a theory gap and failed, the failed reasoning is in
the record (T1, T2) because the failure modes are instructive.
