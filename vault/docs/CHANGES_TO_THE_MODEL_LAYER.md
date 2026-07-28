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
