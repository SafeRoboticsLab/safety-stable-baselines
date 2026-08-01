# Backups and terminal handling

This is the canonical reference for the value operators. Every operator is defined
**once** in [`safety_sb3.backups`](../reference.md#backups-the-value-operators) and every
learner routes through it — learners are written against `backups.target(mode, …)` and
specialize by setting a mode, not by owning a backup. The convention for the two safety
operators: `g ≥ 0` safe, `l ≥ 0` in-target, `V` maximized, `V ≥ 0` ⟺ in the solution set
(see [Safety margins and certificates](safety-margins.md)).

## The shared form

```
target = nt · ( (1 − γ)·anchor + γ·backup ) + (1 − nt)·terminal
```

`nt = 1` on non-terminal steps, `0` on terminal steps. The **anchor** is the
"episode terminates now" payoff (`1 − γ` is the per-step termination probability), and
**it differs by problem**:

| problem | `backup` (v_to_go) | `anchor` |
|---|---|---|
| avoid | `min(g, V')` | `g` |
| reach-avoid | `min(g, max(l, V'))` | `min(l, g)` |

- **avoid** — Fisac et al. 2019; ISAACS eq. 6/7. Stopping now scores well iff you are
  safe, so the anchor is `g`.
- **reach-avoid** — Hsu et al. RSS'21 eq. 15; Gameplay Filters eq. 6a. Stopping now
  scores well iff you are **in the target _and_ safe**, so the anchor is `min(l, g)` —
  the same expression as those papers' finite-horizon terminal condition
  `V_H = min(l, g)` (Gameplay Filters eq. 5b).

!!! danger "Do not anchor reach-avoid on `g`"
    Anchoring reach-avoid on `g` makes "stay safe forever, never reach" a fixed point at
    `V = g > 0` — a win — when its true reach-avoid value is `maxₜ lₜ < 0`, a loss. The
    result is neither problem's value: RSS'21's under-approximation theorem stops
    applying, so the critic can wrongly certify reachability. RSS'21 says of the
    `g`-anchored form (its eq. 13) that it approximates "safety or liveness problems,
    **but not both**". This was the v0.1.0 bug — see the [release notes](../release-notes.md).

## The third mode: `cumulative` (ordinary RL)

`V(s) = r + γ·V'` — the standard discounted-return backup, **not** a safety operator:
its first argument is a *reward*, not a margin, and `V ≥ 0` means nothing. It is a mode
because the learners take the operator as a parameter, so plain reward-maximizing RL
costs one branch: `AbstractSAC(…, mode="cumulative")` is SAC,
`SafetyDQN1P(…, mode="cumulative")` is DQN, and `SafetyRolloutBuffer(…, mode="cumulative")`
reduces exactly to SB3's GAE (asserted in `tests/test_abstract_dp.py`). Use it for a
nominal baseline that shares every line of the safety learners' code.

## Public functions

```python
backups.avoid_target(g, v_next, not_done, gamma)
backups.reach_avoid_target(g, l, v_next, not_done, gamma, terminal_type="all")
backups.cumulative_target(reward, v_next, not_done, gamma)
backups.target(mode, g, v_next, not_done, gamma, l=None, terminal_type="all")
backups.AVOID, backups.REACH_AVOID, backups.CUMULATIVE   # the mode strings
backups.MODES, backups.SAFETY_MODES         # all three / the two safety ones
backups.check_mode(s), backups.check_terminal_type(s)
```

All are elementwise and accept NumPy arrays or PyTorch tensors interchangeably. Full
signatures in the [API reference](../reference.md#backups-the-value-operators).

## `terminal_type` — the algorithm-side knob

How a **terminal** step is valued (the non-terminal blend is unaffected):

| `terminal_type` | terminal target | meaning |
|---|---|---|
| `"all"` (default) | `min(l, g)` | the reach-avoid horizon condition (`V_H`, eq. 5b) |
| `"g"` | `g` | the avoid terminal; also offered by the reference implementation |

`terminal_type` is the **algorithm half** of a pairing whose environment half is the
task's **`end_criterion`** (when the episode ends). The composition matters:

| `end_criterion` (env) | `terminal_type` (algo) | learned behavior |
|---|---|---|
| `failure` (don't stop on reach) | `all` | **reach *deeper*** — value climbs with `l` up to the `g` ceiling |
| `reach-avoid` (stop on reach) | `all` | **reach and stop** — terminal value capped at `min(l, g) ≈ 0` at the boundary |
| `failure` | `g` | viability-flavored; the terminal is the avoid margin |

`end_criterion` lives on the environment/task (see the
[robot-safety-sandbox](https://github.com/SafeRoboticsLab/robot-safety-sandbox) API). The
two knobs are orthogonal; all pairings are constructible.

### Where `terminal_type` applies — by family

`terminal_type` belongs to the **reach-avoid** learners. How an avoid learner treats it
differs by algorithm, because of *where the argument lives*:

- **SAC family** — `terminal_type` is a constructor kwarg on the shared SAC base, so
  **every** SAC learner accepts it and the avoid learners simply **ignore** it (mode
  dispatch never reads it). Passing it to `SafetySAC1P` is harmless.
- **PPO / A2C families** — `terminal_type` lives on the reach-avoid **mixin**
  (`_ReachAvoidPlumbing`), which the avoid learners never receive. So `SafetyPPO1P` /
  `SafetyA2C1P` **do not accept `terminal_type` at all** — passing it raises `TypeError`.
- **DQN** — no reach-avoid variant, so no `terminal_type`.

```python
ReachAvoidPPO1P("MlpPolicy", env, terminal_type="all")            # default
ReachAvoidSAC1P("MlpPolicy", env, terminal_type="g")
ReachAvoidPPO2P("MlpPolicy", env, ctrl_action_dim=2, terminal_type="all")
SafetySAC1P("MlpPolicy", env, terminal_type="g")                  # accepted, ignored
# SafetyPPO1P("MlpPolicy", env, terminal_type="g")                # TypeError
```

## Avoid is not a reach-avoid instance { #avoid-is-not-a-reach-avoid-instance }

Do not run an avoid task on a reach-avoid learner by pinning `l` to a constant. The
reduction needs the anchor to reduce (`min(l,g)=g` ⟹ `l ≥ g`) **and** the recursion to
reduce (`max(l,V')=V'` ⟹ `l ≤ V'`); since `V' ≤ g`, that requires `l ≥ g ≥ V' ≥ l`,
which nothing satisfies:

- `l ≡ −C` (large negative): `V ≡ −C` everywhere → **empty safe set**, with
  healthy-looking `ep_len`/`ep_rew`/`critic_loss`.
- `l ≡ 0` or `+C`: `V ≡ g`, **no lookahead** — coming failures never propagate.

Use the avoid Mode (`SafetyPPO1P` single-player, `SafetyPPO2P` two-player). That is what
the reference does — it switches operator, never degenerates `l`.

## Related

- [Safety margins and certificates](safety-margins.md) — what `g`, `l`, and `V ≥ 0` mean.
- [The MAP convention](../map.md) — the Mode axis is exactly the operator chosen here.
- [Oracle validation](../validation-oracle.md) — the certificate measured against a
  Hamilton–Jacobi ground truth.
