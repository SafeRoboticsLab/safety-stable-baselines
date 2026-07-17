# Release notes

## v0.2.0 — the reach-avoid anchor fix (BREAKING; retraining required)

**If you trained a reach-avoid or ISAACS policy with v0.1.0, its value function is
not the value function you think it is, and the policy needs retraining.** This
release fixes a wrong Bellman operator, renames two classes, and adds the two
learners the library was missing. Read all four sections below before upgrading.

---

### 1. The bug: the reach-avoid anchor was `g`, not `min(l, g)`

Every backup here has the shape

```
target = (1 - γ)·anchor + γ·backup
```

`1 - γ` is the per-step termination probability, so **the anchor is the
"episode terminates now" payoff**. The two problems score that differently:

| problem | anchor | stopping now is a win iff |
|---|---|---|
| avoid | `g` | you are safe |
| **reach-avoid** | **`min(l, g)`** | you are **in the target _and_ safe** |

v0.1.0's PPO family (`ReachAvoidPPO`, `IsaacsPPO`, and both rollout buffers)
anchored reach-avoid on `g`. Consequences:

- **"Stay safe forever, never reach" became a fixed point at `V = g > 0` — a
  win.** Its true reach-avoid value is `maxₜ min(lₜ, min_{s≤t} gₛ) = maxₜ lₜ < 0`,
  a loss. The operator had *no term that punishes not reaching*: `l` entered only
  through `max(l, V')`, which can only ever raise value near the target. In
  practice the learner reliably converged to a safe policy that never attempts
  the task — the **loiter optimum** — and the critic scored it as success.
- **The fixed point was neither the reach-avoid value nor the avoid value.**
  RSS'21's under-approximation theorem (`RA_γ ⊆ RA`, nested increasing in γ) does
  not apply to it, so *the critic is unsound to shield or filter with*: it can
  wrongly certify reachability. RSS'21 says of this `g`-anchored form (its eq. 13)
  that it approximates "safety or liveness problems, **but not both**".
- **The SAC family (`ReachAvoidSAC`, the old `IsaacsSAC`) was already correct** and
  is unchanged by the fix. v0.1.0 shipped **two different fixed points under one
  name**: PPO and SAC disagreed, and the docs asserted the PPO convention applied
  to all backups.

Reference: `safe_adaptation_dev/utils/train.py`, `mode='reach-avoid'`; Hsu,
Rubies-Royo, Tomlin, Fisac, RSS 2021, eq. 15; Hsu et al., *Gameplay Filters*,
eq. 6a. The anchor is literally the same expression as those papers'
finite-horizon terminal condition `V_H = min(l, g)` (Gameplay Filters eq. 5b).

**Where the bad rule came from.** A 2026-07-04 pendulum validation read a stall
under `min(l,g)` as a bug and "fixed" it by switching to the `g` anchor. The stall
was almost certainly the correct operator honestly reporting infeasible spawns
(reach-avoid returns `V < 0` everywhere when the target is unreachable, the policy
goes indifferent, and episodes end early). The `g` anchor "fixed" it by handing
back a dense, always-achievable *survive* signal — `ep_len` went up, and that was
recorded as validation. The rule then propagated into the README, BEST_PRACTICES,
both buffers, and downstream projects.

### 2. BREAKING: `Isaacs*` renamed, and the name now means something else

ISAACS (Hsu, Nguyen, Fisac 2022) is a two-player **avoid** game: its paper defines
only a failure set and margin `g`, has **no target set and no `l` anywhere**, and
anchors on `g`. The string "reach-avoid" appears in it exactly once, in the
bibliography. *Gameplay Filters* is the paper that extends ISAACS to reach-avoid —
and changes the anchor when it does.

v0.1.0's `IsaacsPPO`/`IsaacsSAC` were the reach-avoid game, i.e. Gameplay Filters
wearing the ISAACS name, and the library had **no two-player avoid learner at all**.
The 2x2 is now complete and paper-faithful:

|  | avoid | reach-avoid |
|---|---|---|
| **single-player** | `SafetyPPO` / `SafetySAC` / `SafetyDQN` / `SafetyA2C` | `ReachAvoidPPO` / `ReachAvoidSAC` |
| **two-player** | `IsaacsPPO` / `IsaacsSAC` *(NEW — eq. 7)* | `GameplayPPO` / `GameplaySAC` *(renamed)* |

> ⚠️ **`IsaacsPPO` and `IsaacsSAC` silently changed meaning.** An unmodified
> `from safety_sb3 import IsaacsPPO` still imports, and now gives you the **avoid**
> game. If you were doing two-player reach-avoid, rename to `GameplayPPO` /
> `GameplaySAC`. If you were doing two-player avoid via the `l_neg` trick (below),
> the new `IsaacsPPO` is what you actually wanted.

### 3. `l_neg` / `l_zero` are dead: avoid is NOT a reach-avoid instance

A common workaround was to run an avoid task on a reach-avoid learner by pinning
`l` to a constant — usually `l ≡ -CLAMP` ("`l_neg`"), on the argument that
`max(l, γV') = γV'` reduces the backup to avoid. **That argument only ever held
under the `g` anchor**, which is the bug: it reasons about the recursion and never
mentions the anchor.

Under the correct operator, the reduction needs **both**

- the anchor to reduce: `min(l, g) = g` ⟹ `l ≥ g`, and
- the recursion to reduce: `max(l, V') = V'` ⟹ `l ≤ V'`,

and the avoid recursion caps `V' ≤ g`, so you would need `l ≥ g ≥ V' ≥ l`. **No `l`
satisfies it.** Concretely:

- **`l_neg`** (large negative) buys the recursion and destroys the anchor:
  `V ≡ l` everywhere, independent of the dynamics ⟹ `{V ≥ 0} = ∅`, an empty safe
  set — while `ep_len`, `ep_rew` and `critic_loss` all look healthy. Silent.
- **`l_zero` / `l_pos`** (≥ 0 everywhere) buys the anchor and destroys the
  recursion: the target is the whole space, so you are "already done" at `t=0`,
  `max(l, ·)` clips every negative future, and `V ≡ g` — a myopic "am I safe right
  now" with no lookahead. Coming failures never propagate. (This one was already
  known to destroy a warm-started policy within 25M steps.)

**Use the avoid learner for avoid problems.** That is what the reference does — it
switches operator (`mode='safety'`) rather than hunting for a clever `l`. The new
two-player `IsaacsPPO`/`IsaacsSAC` exist precisely so this is possible.

### 4. What you need to do

1. **Retrain** every `ReachAvoidPPO` / old-`IsaacsPPO` policy. The old value
   functions do not correspond to any reach-avoid problem; results computed from
   them are not backed by the theory. `SafetyPPO`/`SafetySAC` (avoid) and
   `ReachAvoidSAC`/old-`IsaacsSAC` (reach-avoid) are unaffected — they were
   already correct.
2. **Rename** two-player reach-avoid uses: `IsaacsPPO` → `GameplayPPO`,
   `IsaacsSAC` → `GameplaySAC`.
3. **Delete `l_neg` / `l_zero`** and move those tasks to `IsaacsPPO` / `IsaacsSAC`
   (two-player avoid) or `SafetyPPO` / `SafetySAC` (single-player avoid).
4. **Re-examine your `g` / `l` design.** Now that reach-avoid actually punishes
   not-reaching, formulations tuned against the old operator will behave
   differently — often *better*, since the pathology the old anchor produced was
   exactly "never attempt". In particular, any `p*`-style risk-dial calibration
   that assumed loitering is worth `g > 0` needs re-deriving: under the correct
   operator loitering is worth `l < 0`, so attempting only has to beat a negative
   baseline and the break-even attempt probability is lower.
5. **If you consume a reach-avoid critic as a filter/shield** (value shielding,
   Q-CBF/R-CBF), re-derive your guarantee. The old critic had no
   under-approximation property.

### Also in this release

- **`safety_sb3/backups.py`** — both operators now defined **once** and shared by
  every learner. The v0.1.0 bug survived because the backup was re-implemented at
  four call sites, which drifted; that is now structurally impossible.
- `ReachAvoid*` accept **`terminal_type`** (`"all"` → `min(l, g)`, default and the
  horizon condition; `"g"` → `g`), matching the reference's own parameter.
- Tests: reach-avoid **anchor** and **loiter fixed-point** tests, a `terminal_type`
  test, and numpy↔torch buffer parity. (v0.1.0's `tests/test_backups.py` covered
  only the numpy PPO buffers and never tested the anchor; its parity was untested,
  and its RA smoke test asserted an ordering that the avoid term alone already
  produced — so it passed without exercising reach at all.)
- Fixed: `IsaacsPPO`'s dstb rollout buffer hardcoded `ReachAvoidRolloutBuffer` on
  the numpy path instead of honouring the class's buffer attribute.
