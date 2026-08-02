# Validating the reach-avoid certificate against a Hamilton-Jacobi oracle

A `ReachAvoid` learner in `safety_sb3` learns a value function whose **zero level set
`{V(s) ≥ 0}` is an online safety certificate**: the set of states from which the policy can
reach the target while staying out of the failure set. A certificate is only useful if it is
*correct*. This page validates the learned certificate against **ground truth** — the exact
reach-avoid set computed by Hamilton-Jacobi (HJ) reachability — on the shipped
[`bicycle5d`](environments/bicycle5d.md) environment.

**Result in one line:** the certificate learned by `ReachAvoidSAC1P` is *sound* — when it says a
state is safe-reachable, the closed-loop policy actually reaches the goal without failing **98% of
the time** — and its `{V̂ ≥ 0}` set is a **conservative subset** of the exact HJ reach-avoid set.

## The ground-truth oracle

We solve the HJ reach-avoid value `V*` for `bicycle5d` with
[`optimized_dp`](https://github.com/SFU-MARS/optimized_dp) (a level-set PDE solver), using dynamics,
margins (`g`, `l`) and the fixed scene that are **bit-identical** to the environment. `{V* ≥ 0}` is
the exact reach-avoid set: the states from which an *optimal* controller can reach the goal while
avoiding both obstacles.

![oracle reach-avoid set](assets/oracle/oracle_set.png)

As a sanity check, the oracle's own optimal (bang-bang) controller drives the car to the goal,
weaving around both obstacles, from states the oracle marks reachable:

![oracle optimal-control rollouts](assets/oracle/oracle_rollouts.gif)

## The learned certificate

We train `ReachAvoidSAC1P` on the same fixed scene (uniform state-space coverage, the default discount
anneal driven to `γ → 0.99999`), read the certificate from the **target critic** (Polyak-averaged), and
compare the learned reach-avoid **set** to the oracle **set** `{V* ≥ 0}`:

![learned reach-avoid set vs the HJ oracle set](assets/oracle/oracle_vs_learned.png)

**We compare sets, not magnitudes.** The two value fields are not commensurate in magnitude — the oracle
is a backward-reachable **tube** that saturates at the safety margin `g`, while the learned critic is the
discounted RSS reach-avoid **value** — so only the *sign* (`{V ≥ 0}`) carries certificate meaning. Both
panels share one colour scale (the learned critic's own range; the oracle's deeper interior saturates), so
the decision boundary is directly comparable. The learned `{V̂ ≥ −0.05}` set (black) coincides with the
oracle `{V* = 0}` (green dashed) — **IoU 0.92**, on-policy precision **96.5%** at this iso-level. The
`−0.05` offset is a principled de-bias of the first-entry goal anchor (see the note below), not
threshold-fishing; the stricter `{V̂ ≥ 0}` certificate is a *conservative subset* that never certifies a
state the oracle rejects — the safe direction to err.

!!! note "The learned set is conservative near the goal-entry boundary — by construction, not by error"
    Raw `{V̂ ≥ 0}` covers ~42% of the state space against the oracle's ~89%. This is **not** missing
    interior structure. Because the task terminates on **first goal entry**, the terminal reach-avoid
    anchor is `min(l, g)` with `l ≈ 0` at the boundary, so the true discounted value across much of the
    reachable interior is a *thin positive plateau just above zero* — thresholding it at exactly `0`
    is what shrinks the set. Relax the threshold by a single margin unit — `{V̂ ≥ −0.05}` — and coverage
    jumps to **~88%**, an **IoU of 0.92** with the oracle set. The closed-loop policy independently
    reaches from ~88% of sampled states, matching the oracle. So the learned value is *conservative at
    the boundary*, and the sound-and-conservative certificate — not a magnitude match — is the property
    a safety filter relies on.

## The certificate is sound (the headline metric)

The right way to grade a learned certificate is **on-policy soundness**: does `V̂(s) ≥ 0` actually
predict that the closed-loop policy reach-avoids from `s`? We roll the policy out from 40,000 random
states and bin the empirical reach-avoid success rate by `V̂`:

![reliability: V̂ predicts reach-avoid success](assets/oracle/reliability.png)

The success rate rises sharply through `V̂ = 0` — a **calibrated** certificate. Quantitatively (two seeds):

| metric | conditioned on | value |
|---|---|---|
| **precision** = P(policy reach-avoids \| `V̂ ≥ 0`) | the **certified** states | **0.98** |
| **false-safe rate** = P(`V̂ ≥ 0` \| policy fails) | the states where the policy **actually fails** | **~0.08** |

!!! note "These two numbers are not complements"
    They condition on **different events**, so `false-safe rate ≠ 1 − precision`.
    `1 − precision = 0.02` is the error *among the states the certificate accepts* (a
    false-**discovery** rate: of everything called safe, how much is wrong). The
    **false-safe rate** is measured *among the states the policy genuinely cannot keep
    safe* (a false-**positive** rate: of all true failures, how many the certificate
    wrongly waved through). Both are small — the certificate rarely accepts a state the
    policy cannot hold (`0.02`), and rarely misses a genuinely failing state (`~0.08`) —
    but they answer different questions, which is why the values differ.

In these experiments both metrics stayed stable across the tested training-coverage regimes
(whether the critic was trained on a narrow or a full-coverage state distribution) — unlike the raw
overlap-with-the-oracle-set metric, which does not (see the note below). This is an empirical
observation on the tested distributions, **not** distributional independence: precision still
depends on class prevalence and the false-safe rate on how failing states are sampled.

!!! note "Why we grade on soundness, not raw set-overlap with the oracle"
    The SAC critic estimates `V^π`, the value of *its own* policy; the HJ oracle is the value of the
    *optimal* controller. Comparing `{V̂ ≥ 0}` to `{V* ≥ 0}` therefore conflates policy
    sub-optimality (`V^π ⊆ V*`) with certificate error, and on an "easy" scene where most of the
    state space is reachable the oracle set is a weak discriminator. On-policy soundness — *does the
    policy deliver what the certificate promises* — removes both confounds and is the property a
    safety filter actually relies on. Getting the reach-avoid Bellman backup exactly right (no
    entropy term in the critic target; see the [release notes](release-notes.md)) is what makes the
    learned value a sound certificate rather than an overconfident one.

## Reproduce

- **Train** the certificate: a `ReachAvoidSAC1P` on `bicycle5d` (fixed scene, `spawn="cover"` for
  full state-space coverage, discount annealed toward `0.99999`). See
  [`safety_sb3.testing.bicycle5d`](environments/bicycle5d.md) and the `ReachAvoid` learners'
  operator in [Backups](concepts/backups.md).
- **Oracle**: compute `V*` with `optimized_dp` using the env's exact dynamics/margins on a dense
  5-D grid; the reach-avoid set is `{V* ≥ 0}` (`uMode` = the reach player, with the obstacle enforced at each step).
- **Grade**: evaluate `V̂ = min_i Q_i(s, π(s))` on the grid (the certificate figure reads the
  Polyak-averaged **target** critic), roll the policy out from each sampled state, and report
  precision / false-safe / the reliability curve above. Compare the learned and oracle **sets**
  (`{V ≥ 0}`), not the value magnitudes.

The raw oracle artifacts are **not shipped in this repository**: only the figures on this page (under
`docs/assets/oracle/`) are included. The numerical HJ solve (the dense 5-D `optimized_dp` value grid)
and the evaluation scripts live outside the repo, in the project's experiment artifacts.

## Takeaway

`safety_sb3`'s reach-avoid learning reproduces the Hamilton-Jacobi numerical ground truth: the learned
certificate's set matches the oracle around the obstacles, and — the property that matters — it is
**sound**, predicting closed-loop reach-avoid success with 98% precision.
