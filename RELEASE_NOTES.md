# Release notes

## v0.4.0 — the MAP rename (BREAKING, no shims)

**Every public learner is renamed.** This release is a migration guide.

There are **no aliases, no deprecation warnings, and no compatibility shims**. Old
code will fail at import with a plain `ImportError`, which is deliberate: the
previous rename (v0.2.0) changed what `Isaacs*` *meant* while keeping the name, so
existing code kept importing and silently trained a different algorithm. A hard
break cannot do that.

**If you need the old names, pin v0.3.x:**

```
safety_sb3 @ git+https://github.com/SafeRoboticsLab/safety-stable-baselines.git@v0.3.0
```

The rename is the largest change, but v0.4.0 also lands a **certificate-soundness
fix** (the entropy bonus is removed from the safety critic target — §7, validated
against an HJ oracle), a **CUDA-graph SAC speedup** (§8), and **new docs** (§9:
the MAP convention and the oracle-validation study).

### Why

Class names encoded paper lineage (`IsaacsPPO`, `GameplaySAC`), so using the library
required knowing which paper introduced which game — and, after v0.2.0, which
*version* of which name meant which game. Names now encode the algorithm instead:

> **Here's the MAP to navigate the codebase — Mode. Algorithm. Players.**

```
M = Mode       Safety | ReachAvoid | Cumulative    (which Bellman operator)
A = Algorithm  PPO | SAC | A2C | DQN               (which RL method)
P = Players    1P | 2P                             (single | zero-sum game)
```

`SafetyPPO1P`, `ReachAvoidSAC2P`, `CumulativePPO1P` — read the name, know the
algorithm. The papers are still cited in the docstrings; you just no longer need
them to pick a class.

### 1. Rename table

| v0.3.x | v0.4.0 |
|---|---|
| `SafetyPPO` | `SafetyPPO1P` |
| `ReachAvoidPPO` | `ReachAvoidPPO1P` |
| `IsaacsPPO` | **`SafetyPPO2P`** (two-player avoid) |
| `GameplayPPO` | **`ReachAvoidPPO2P`** |
| `SafetySAC` | `SafetySAC1P` |
| `ReachAvoidSAC` | `ReachAvoidSAC1P` |
| `IsaacsSAC` | **`SafetySAC2P`** (two-player avoid) |
| `GameplaySAC` | **`ReachAvoidSAC2P`** |
| `SafetyA2C` | `SafetyA2C1P` |
| `SafetyDQN` | `SafetyDQN1P` |
| `IsaacsPolicy` | **`TwoPlayerSACPolicy`** (now in `safety_sb3.policies`) |

Buffers keep a **Mode prefix and no player count** — `SafetyRolloutBuffer`,
`ReachAvoidRolloutBuffer`, and the `Tensor*` twins are unchanged names. A buffer
cannot tell how many players filled it, and the two-player learners build two
instances of the very same class.

Module files were renamed to match; every `isaacs_*` file is gone:

| v0.3.x | v0.4.0 |
|---|---|
| `safety_ppo.py` | `ppo_base.py` + `ppo_1p.py` |
| `isaacs_ppo.py` | `ppo_2p.py` |
| `reach_avoid_ppo.py` | `reach_avoid_mixin.py` |
| `safety_sac.py`, `reach_avoid_sac.py` | `sac_1p.py` |
| `isaacs.py` | `sac_2p.py` |
| `safety_a2c.py` / `safety_dqn.py` | `a2c.py` / `dqn.py` |
| `safety_buffers.py` | `buffers_rollout.py` |
| `tensor_buffers.py` | `buffers_tensor.py` |
| `isaacs_buffers.py` | `buffers_replay.py` |
| `isaacs_policy.py` | `policies.py` |

### 2. New classes (the naming law implies them)

- **`CumulativePPO1P`, `CumulativeSAC1P`, `CumulativeA2C1P`, `CumulativeDQN1P`** —
  ordinary reward-maximizing RL as a first-class Mode, so a nominal baseline runs
  through every line of the same code as the safety learners. Not safety learners:
  the reward is a reward, and `V ≥ 0` means nothing.
- **`ReachAvoidA2C1P`** — new, and free: the Mode axis costs one line.
- **`AbstractPPO1P` / `AbstractPPO2P` / `AbstractSAC1P` / `AbstractSAC2P` /
  `AbstractA2C` / `AbstractDQN`** — the extension points. The `*1P`/`*2P` loops run
  directly with `mode=`.
- **`LeagueEvaluator`** (in `safety_sb3.leaderboard`) — the two-player league's
  scorer, lifted out of the SAC learner so all league machinery lives in one module.

**`ReachAvoidDQN1P` does not exist and will not.** SB3's discrete-action replay
buffer carries no target margin `l(s)`, so the reach-avoid operator is not computable
there. `mode="reach-avoid"` raises at construction rather than computing something
else. Use `ReachAvoidSAC1P`, or add an `l`-carrying discrete replay buffer.

### 3. Removed

- **All compatibility surface.** The v0.2.0 rename warning in `__init__.py` and every
  "before v0.2.0…" note are gone. v0.2.0 is far enough back that the archaeology cost
  more than it explained; `git log` has it.
- **`_is_reach_avoid`.** It existed only so avoid-mode subclasses could inherit
  reach-avoid machinery and switch it off. Under the new tree the avoid learners never
  receive that machinery — reach-avoid is a **mixin** (`_ReachAvoidPlumbing`) composed
  onto both player counts, not a base class — so the predicate became unreachable.
  Deleting it is a correctness win, not tidiness: previously *every* `l`-touching line
  needed a guard, and a single missed guard fed a target margin to an avoid learner.
- **`numpy_rollout_buffer_class` / `tensor_rollout_buffer_class` class attributes.**
  Buffers are now looked up from `_MODE` via
  `buffers_rollout.rollout_buffer_classes(mode)`, so a buffer can no longer disagree
  with the mode its owner declares. Pass `rollout_buffer_class=` to override.

### 4. Behavioral changes

The refactor is otherwise behavior-preserving: seeded loss and parameter traces for
**all ten learners** (4 PPO, 4 SAC, A2C, DQN) on the toy envs reproduce v0.3.x
**bitwise**, including the two-player league operation traces. Three intentional
exceptions:

- **Both two-player rollout buffers now get the same kwargs.** `terminal_type` reached
  the ctrl buffer but not the dstb buffer, so with a non-default `terminal_type` the
  two players scored terminal states differently. No effect at the default (`"all"`).
- **Two-player log keys `isaacs/*` → `game/*`** (`game/phase_is_dstb`,
  `game/rollouts_done`, `game/archived_dstb`, `game/lr_ctrl`, `game/lr_dstb`). Update
  any dashboard that plots them.
- **Default league directories renamed** — `isaacs_ppo_leaderboard` →
  `ppo_2p_leaderboard`, `isaacs_leaderboard` → `sac_2p_leaderboard`. Pass
  `leaderboard_dir=` to keep an existing archive path.

Also, `AbstractPPO` and `AbstractSAC` now **refuse to run** instead of inheriting
SB3's stock loop. Instantiating a bare base previously would have converged to the
cumulative fixed point while `_MODE` claimed a safety operator.

### 5. Migrating

A mechanical rename covers almost all call sites:

```bash
sed -i -E 's/\bIsaacsPPO\b/SafetyPPO2P/g;   s/\bGameplayPPO\b/ReachAvoidPPO2P/g;
           s/\bIsaacsSAC\b/SafetySAC2P/g;   s/\bGameplaySAC\b/ReachAvoidSAC2P/g;
           s/\bReachAvoidPPO\b/ReachAvoidPPO1P/g; s/\bReachAvoidSAC\b/ReachAvoidSAC1P/g;
           s/\bSafetyPPO\b/SafetyPPO1P/g;   s/\bSafetySAC\b/SafetySAC1P/g;
           s/\bSafetyA2C\b/SafetyA2C1P/g;   s/\bSafetyDQN\b/SafetyDQN1P/g;
           s/\bIsaacsPolicy\b/TwoPlayerSACPolicy/g' $(git ls-files '*.py')
```

Apply the two-player substitutions **before** the single-player ones (`SafetySAC2P`
contains `SafetySAC`). Saved checkpoints (`.zip`) store the class path, so a model
trained on v0.3.x will not load on v0.4.0 — pin v0.3.x to load it, or retrain.

### 6. Also in v0.4.0 — the abstract-DP refactor

This landed on the same branch and ships here. It is behavior-preserving on its own:
no default changed, and the learners reproduce v0.3.0 bit-for-bit except `SafetyDQN1P`
(~1e-6 relative, pure float reassociation) and `ReachAvoidSAC1P` when
`min_alpha`/`max_alpha` are set away from their defaults (a bug fix) — both below.
Class names below are the v0.4.0 ones.

- **Every learner now takes its backup as a parameter.** `AbstractSAC1P` holds the
  single-player SAC update loop and calls `backups.target(self._MODE, …)`;
  `SafetySAC1P`, `ReachAvoidSAC1P` and `CumulativeSAC1P` are one-line `_MODE`
  specializations of it (previously `ReachAvoidSAC.train()` was a hand-copy of
  `SafetySAC.train()`). The mode is also selectable per instance:
  `Cls(…, mode="reach-avoid")`. The two-player learners keep their own `train()`
  (twin actors, per-actor entropy) and share only the backup.
- **Bug fix — `min_alpha`/`max_alpha` now apply to `ReachAvoidSAC1P`.** The copied
  `train()` had dropped the `_clamp_entropy_temps()` call, so the entropy-
  temperature floor/ceiling silently did nothing on the single-player
  reach-avoid path (measured: α = 0.763 after 1500 steps with `min_alpha=0.9`).
  Two-player SAC was never affected (it clamps inline). If you trained RA-1P
  with a non-default `min_alpha`, α was unclamped; retrain for the intended
  behavior.
- **New mode `backups.CUMULATIVE`** — the standard discounted-return backup
  `r + γ·V'`. Plain reward-maximizing RL is now a mode of this library, so a
  nominal baseline runs through the same actor/critic/entropy code as the safety
  learners. v0.4.0 gives it first-class classes (`CumulativeSAC1P`,
  `CumulativePPO1P`, `CumulativeA2C1P`, `CumulativeDQN1P`); `mode="cumulative"`
  also still works on any learner, and `CumulativeRolloutBuffer` ≡ SB3's GAE. It is
  **not** a safety operator: its first argument is a reward and `V ≥ 0` carries no
  certificate meaning.
- **`SafetyDQN1P` no longer inlines its own backup.** It computed
  `(1 − γ·nt)·g + γ·nt·min(g, V')` by hand; that is the same expression as
  `backups.avoid_target`, which it now calls. Reassociating the terms changes
  float rounding, so long DQN runs drift ~1e-6 relative from v0.3.0.
- The on-policy buffers (`SafetyRolloutBuffer` / `ReachAvoidRolloutBuffer` and
  their tensor twins) likewise share one `_target` that dispatches on `_MODE`.

### 7. Certificate soundness — the entropy bonus is gone from the safety critic target

**This is a behavioral fix for every SAC safety/reach-avoid learner; retrain to
benefit.** The SAC critic target subtracted the maximum-entropy bonus
(`next_q -= ent_coef · log_prob`) inside the Bellman backup. That term is correct
for the *cumulative* backup (`r + γ·V'`, entropy-regularized RL), but the
safety/reach-avoid backups are a `min`/`max` of *margins*, not a discounted
reward sum — adding an entropy bonus there biases the value and **breaks the
`{V ≥ 0}` certificate**. v0.4.0 removes the entropy term from the safety and
reach-avoid critic targets (`sac_1p.py`, `sac_2p.py`); it stays in the cumulative
target, where it belongs. PPO was never affected — its entropy lives in the loss,
not the value target.

Measured against a Hamilton-Jacobi reach-avoid **oracle** on the `bicycle5d`
env: on-policy certificate precision — `P(policy reach-avoids | V̂ ≥ 0)` — is
**0.98 with the fix vs 0.05 with the bug**. The full study, with the oracle
solve and the reliability plots, is the new [Oracle validation](validation-oracle.md)
page. If you trained a SAC safety/RA twin on ≤ v0.3.x, its value function is
unsound as a certificate — retrain on v0.4.0.

### 8. Performance — the SAC update is CUDA-graph-capturable

The single- and two-player SAC update loops no longer issue in-loop host↔device
syncs and use a fused Polyak target update, so the whole gradient step can be
captured as a CUDA graph. On the GPU-resident tensor path this removes the
per-step launch overhead that dominated at small batch sizes; no default or
result changes (seeded traces are unchanged), it is purely a throughput fix.

### 9. New documentation

- **[The MAP convention](map.md)** — the canonical Mode·Algorithm·Players naming
  law, the full learner roster, the abstract class hierarchy, and how the
  environment layer composes with it. Read this first.
- **[Oracle validation](validation-oracle.md)** — the `bicycle5d` reach-avoid
  learner vs. a numerical HJ oracle: the soundness result behind §7.

## v0.3.0 — two-player SAC + reference-faithful discount annealing

Additive release (no anchor/API breakage vs v0.2.x), with **one behavioral default
change**: reach-avoid learners now **anneal the discount `γ` by default**. Retrain
to benefit; pin the schedule off (`gamma_schedule=None`) to reproduce v0.2.x exactly.

### 1. Two-player SAC on the tensor path
`GameplaySAC` (two-player **reach-avoid**, Gameplay Filters eq. 6a) and `IsaacsSAC`
(two-player **avoid**, ISAACS) — the off-policy analog of `GameplayPPO`/`IsaacsPPO`:
a twin critic over the concatenated `[ctrl, dstb]` action, two soft actors, two
entropy temperatures, the safety/reach-avoid backup, and a self-play leaderboard.

### 2. Discount (γ) annealing — ON by default
Ports `safe_adaptation_dev`'s reference-faithful schedule. `StepGammaAnneal`
(discrete jumps `0.99 → 0.999 → 0.9999`, the default) and `GeometricGammaAnneal`
(continuous), via a `GammaAnnealMixin` across the PPO **and** SAC families. On each
discrete jump the entropy temperature `α` is **reset** (re-exploration) and floored
at `min_alpha`. This is what let two-player RA-SAC recover/beat the reference on
`go2_stabilize`.

### 3. Per-agent LR / entropy controls (two-player SAC)
Independent `critic_lr` / `dstb_lr` / `ent_coef_lr` / `dstb_ent_coef_lr`, fixed or
`StepLR` schedules, and a `min_alpha`/`max_alpha` entropy-temperature clamp.

### 4. Leaderboard throughput — on-device league eval
`_eval_pair_tensor` runs the self-play pairing evaluation entirely on-device
(no numpy `VecEnv` round-trip). With throughput-tuned defaults (larger eval
interval, fewer episodes) this is ~30× faster at 1024 envs — the leaderboard no
longer stalls collection.

### 5. Eval + reference env
`SafeSuccessRateEvalCallback` logs safe-rate / success-rate to wandb. Added the
`Bicycle5D` reference reach-avoid environment + docs showreel. Full knob reference
in `docs/hyperparameters.md`.

---

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
  not apply to it, so *the critic is unsound to filter with*: it can
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
5. **If you consume a reach-avoid critic as a filter** (value filtering,
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
