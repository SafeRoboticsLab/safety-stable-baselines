# Safety margins and certificates

## Intuition

> A **safety margin** is a number that is positive in allowed states and negative after
> a violation.

That single scalar is how you tell a safety learner what "safe" means. You do not write
a reward that trades off progress against risk; you hand the learner the margin and it
learns the largest set of states from which safety can be maintained. The learned value
function is that answer, and — because it is grounded in the margin — its **sign is a
certificate** you can trust at deployment time.

## The two margins

There are two margins, one per safety problem. Both follow the same sign convention:
**≥ 0 means "in the good set"**.

| symbol | name | `≥ 0` means | used by |
|---|---|---|---|
| `g(s)` | **safety margin** | the state is **outside the failure set** (safe) | every safety learner |
| `l(s)` | **target margin** | the state is **inside the target set** | `ReachAvoid*` only |

A natural choice for `g` is a signed distance to the failure boundary (positive outside,
negative inside), and for `l` a signed distance to the target (positive inside). The
*magnitudes* matter too — they set the value function's scale and, in reach-avoid, the
risk trade-off between committing to a maneuver and stopping (see
[best practices](../best-practices.md)).

## The certificate: `V ≥ 0`

A trained value function `V` satisfies:

> `V(s) ≥ 0` ⟺ **the policy can stay safe from `s`** — forever (avoid) or until it
> reaches the target without ever failing (reach-avoid).

This is the property that makes `safety_sb3`'s value functions usable as a **runtime
safety filter**: run your nominal controller, and switch to the safety policy whenever
the nominal's proposed next state has `V < 0`. The zero level set `{V = 0}` is the
boundary of the safe-controllable region.

For this to hold, the value function must be learned with the right Bellman operator —
a `min`/`max` recursion over margins, **not** a discounted sum of rewards. That is what
[Backups](backups.md) are about, and getting it exactly right (no entropy bonus in the
critic target) is what makes the certificate *sound* rather than overconfident — see the
[oracle validation](../validation-oracle.md) study, which measures the certificate
against a Hamilton–Jacobi ground truth.

## How the library represents it

- `g(s)` rides on the **reward channel** — the reward *is* the margin.
- `l(s)` rides on **`info["l_x"]`** (NumPy path) or is returned directly by
  `step_tensor` (tensor path).
- The episode **terminates when `g < 0`**.

The exact interface — spaces, dones/truncation, normalization rules, NumPy vs tensor —
is the [environment contract](environment-contract.md).

!!! warning "Never normalize the reward"
    `VecNormalize(norm_reward=True)`, or any reward scaler, silently corrupts the
    backup: the value function's zero level set *is* the safety boundary, and rescaling
    moves it. Observation normalization is fine and recommended.

## Consequences for users

- You get a **certificate for free** with the policy — no separate learning step.
- The certificate is only as good as the margin. A `g` that is wrong at the boundary
  moves the boundary; a margin violated by your own reset states condemns that region by
  construction.
- Avoid and reach-avoid are **different problems with different operators**. Choosing
  the problem is choosing the Mode, not bending the margins — see
  [Choose a learner](../getting-started/choose-a-learner.md).

## Related API

- [`safety_sb3.backups`](backups.md) — the value operators over margins.
- [The environment contract](environment-contract.md) — where `g` and `l` go.
- [`SafeSuccessRateEvalCallback`](../how-to/evaluate.md) — measures whether the
  certificate holds on-policy.
