# safety-stable-baselines — API reference

The canonical contract for the **algorithm layer**. This is the source of truth
for what `safety_sb3` exposes and what it expects from an environment; the
environment side (mjlab tasks, margins, `end_criterion`) is documented in
[robot-safety-sandbox `docs/API.md`](https://github.com/SafeRoboticsLab/robot-safety-sandbox/blob/main/docs/API.md).
Keep the two in sync — the `g`/`l` channel contract below is shared, and this
document defines it.

- Orientation & rationale: [Home](index.md)
- Training recipes & pitfalls: [Best practices](best-practices.md)
- What changed in v0.2.0 (breaking): [Release notes](release-notes.md)

---

## 1. The environment contract

Every learner consumes an environment through two channels plus a done signal.
Get these right and everything works; get `g` wrong and the value function's zero
level set — the safety boundary — moves.

| channel | symbol | meaning | where |
|---|---|---|---|
| reward | `g(s)` | **safety margin**. `g ≥ 0` ⟺ outside the failure set. | the reward field |
| info | `l(s)` | **target margin**. `l ≥ 0` ⟺ inside the target set. Reach-avoid only. | `info["l_x"]` (numpy) / `step_tensor`'s 5th return (tensor) |
| dones | | terminate when `g < 0`; timeouts are never value-bootstrapped by default | standard gym |

Rules, each of which has cost real debugging (see BEST_PRACTICES):

- **Never normalize the reward.** `g` *is* the margin. `VecNormalize(norm_reward=True)`
  or any reward scaler silently destroys the backup. Observation normalization is fine.
- **Terminate on `g < 0`.** Letting the sim run past a violation leaks post-failure
  states into the value target.
- **`l` is read only by the reach-avoid column** (`ReachAvoid*`, `Gameplay*`). The
  avoid column (`Safety*`, `Isaacs*`) ignores it. An avoid task should not invent one
  — see §5.

### Numpy path vs tensor path

- **Numpy path** (standard SB3 `VecEnv`): `g` on `reward`, `l` on `info["l_x"]`.
  Used by the SAC family and by PPO on CPU envs.
- **Tensor path** (`TensorVecEnv`, GPU-resident): the env implements
  `step_tensor(actions) -> (obs, reward_g, dones, timeouts, l_x)`, all device tensors.
  `TensorVecNormalize` normalizes observations only. This is the ~50k-step/s regime for
  massively-parallel envs; the PPO family auto-detects it.

---

## 2. The learners — a 2×2 over {problem} × {players}

|  | **avoid** (stay safe forever) | **reach-avoid** (reach it, staying safe) |
|---|---|---|
| **single-player** | `SafetyPPO` `SafetySAC` `SafetyDQN` `SafetyA2C` | `ReachAvoidPPO` `ReachAvoidSAC` |
| **two-player** (ctrl max + dstb min) | `IsaacsPPO` `IsaacsSAC` | `GameplayPPO` `GameplaySAC` |

- `Isaacs*` = ISAACS (Hsu et al. 2022), the two-player **avoid** game — no target set, no `l`.
- `Gameplay*` = Gameplay Filters (Hsu et al. 2024), which extends ISAACS to reach-avoid.

> These names changed meaning in **v0.2.0**. Pre-v0.2.0 `Isaacs*` were reach-avoid.
> See RELEASE_NOTES.md before upgrading old code.

Pick the **row** by your problem (does the task have a target to reach?) and the
**column** by whether you train against a worst-case disturbance adversary.

---

## 3. The backups (`safety_sb3.backups`)

Both operators are defined **once** in `safety_sb3/backups.py` and every learner
routes through it. Convention: `g ≥ 0` safe, `l ≥ 0` in-target, `V` maximized,
`V ≥ 0` ⟺ in the solution set.

```
target = nt · ( (1 − γ)·anchor + γ·backup ) + (1 − nt)·terminal
```

`nt = 1` on non-terminal steps, `0` on terminal steps. The **anchor** is the
"episode terminates now" payoff (`1 − γ` is the per-step termination probability):

| problem | `backup` (v_to_go) | `anchor` |
|---|---|---|
| avoid | `min(g, V')` | `g` |
| reach-avoid | `min(g, max(l, V'))` | `min(l, g)` |

- **avoid** — Fisac et al. 2019; ISAACS eq. 6/7.
- **reach-avoid** — Hsu et al. RSS'21 eq. 15; Gameplay Filters eq. 6a. The anchor
  `min(l, g)` is the same expression as the finite-horizon terminal condition
  `V_H = min(l, g)` (Gameplay Filters eq. 5b).

**Do not anchor reach-avoid on `g`.** That makes "stay safe forever, never reach"
a fixed point at `V = g > 0` — a win — when its true value is `maxₜ lₜ < 0`. The
result is neither problem's value, RSS'21's under-approximation theorem stops
applying, and the critic is unsound to shield with. This was the v0.1.0 bug.

Public functions:

```python
backups.avoid_target(g, v_next, not_done, gamma)
backups.reach_avoid_target(g, l, v_next, not_done, gamma, terminal_type="all")
backups.target(mode, g, v_next, not_done, gamma, l=None, terminal_type="all")
backups.AVOID, backups.REACH_AVOID          # the two mode strings
backups.check_terminal_type(s)              # validates "all" | "g"
```

All are elementwise and accept numpy arrays or torch tensors interchangeably.

---

## 4. `terminal_type` — the algorithm-side knob

How a **terminal** step is valued (the non-terminal blend is unaffected):

| `terminal_type` | terminal target | meaning |
|---|---|---|
| `"all"` (default) | `min(l, g)` | the reach-avoid horizon condition (`V_H`, eq. 5b) |
| `"g"` | `g` | the avoid terminal; also offered by the reference impl |

It is a **first-class constructor kwarg** on every reach-avoid learner and is
ignored (harmlessly) by the avoid learners:

```python
ReachAvoidPPO("MlpPolicy", env, terminal_type="all")     # default
ReachAvoidSAC("MlpPolicy", env, terminal_type="g")
GameplayPPO("MlpPolicy", env, ctrl_action_dim=2, terminal_type="all")
```

`terminal_type` is the algorithm half of a pairing whose environment half is the
task's **`end_criterion`** (when the episode ends). The composition matters:

| `end_criterion` (env) | `terminal_type` (algo) | learned behavior |
|---|---|---|
| `failure` (don't stop on reach) | `all` | **reach *deeper*** — value climbs with `l` up to the `g` ceiling |
| `reach-avoid` (stop on reach) | `all` | **reach and stop** — terminal value capped at `min(l, g) ≈ 0` at the boundary |
| `failure` | `g` | viability-flavored; the terminal is the avoid margin |

`end_criterion` lives on the environment/task — see the sandbox API reference.
The two knobs are orthogonal; all pairings are constructible.

---

## 5. Avoid is not a reach-avoid instance

Do not run an avoid task on a reach-avoid learner by pinning `l` to a constant.
The reduction needs the anchor to reduce (`min(l,g)=g` ⟹ `l ≥ g`) **and** the
recursion to reduce (`max(l,V')=V'` ⟹ `l ≤ V'`); since `V' ≤ g`, that requires
`l ≥ g ≥ V' ≥ l`, which nothing satisfies:

- `l ≡ −C` (large negative): `V ≡ −C` everywhere → **empty safe set**, with
  healthy-looking `ep_len`/`ep_rew`/`critic_loss`.
- `l ≡ 0` or `+C`: `V ≡ g`, **no lookahead** — coming failures never propagate.

Use the avoid **column** (`SafetyPPO` single-player, `IsaacsPPO` two-player). That
is what the reference does — it switches operator, never degenerates `l`.

---

## 6. Exported symbols

```python
from safety_sb3 import (
    # single-player avoid
    SafetyPPO, SafetySAC, SafetyDQN, SafetyA2C,
    # single-player reach-avoid
    ReachAvoidPPO, ReachAvoidSAC,
    # two-player avoid (ISAACS eq. 7)
    IsaacsPPO, IsaacsSAC,
    # two-player reach-avoid (Gameplay Filters)
    GameplayPPO, GameplaySAC,
    # tensor path
    TensorVecEnv, TensorVecNormalize,
    TensorSafetyRolloutBuffer, TensorReachAvoidRolloutBuffer,
    # buffers / policy / callbacks
    SafetyRolloutBuffer, ReachAvoidRolloutBuffer,
    ReachAvoidReplayBuffer, IsaacsPolicy, StdCapCallback,
    # the operators
    backups,
)
```

### Two-player learners

`Isaacs*` / `Gameplay*` take a single concatenated action `Box(ctrl_dim + dstb_dim)`
split by `ctrl_action_dim`, and require `ctrl_action_dim` at construction:

```python
GameplayPPO("MlpPolicy", env, ctrl_action_dim=12, terminal_type="all")
```

`self.policy` is always the **control** policy, so `predict()`, `save()`, and
downstream filter wrappers see the deployable controller.

### Timeout handling

`bootstrap_on_timeout=False` (default) treats timeouts as terminal for the backup
— correct for the `g`/`l` margin semantics, which are absolute, not returns to
bootstrap. Under `terminal_type="all"` a timeout gets `min(l, g)`, which is the
horizon cutoff by construction.

---

## 7. Minimal examples

```python
# single-player reach-avoid, tensor path
from safety_sb3 import ReachAvoidPPO
model = ReachAvoidPPO("MlpPolicy", tensor_env, normalize_obs=True,
                      terminal_type="all", n_steps=48, batch_size=24576)
model.learn(2_000_000_000)

# two-player avoid (ISAACS), numpy path
from safety_sb3 import IsaacsPPO
model = IsaacsPPO("MlpPolicy", adv_env, ctrl_action_dim=2)   # no l, no terminal_type
model.learn(5_000_000)
```
