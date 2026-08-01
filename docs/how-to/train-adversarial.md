# Train an adversarial (two-player) policy

**Goal.** Train a controller that is safe against a **worst-case disturbance** by playing
a zero-sum game: a control player maximizes the safety value, a disturbance player
minimizes it. The result is a *robust* safety certificate — this is the ISAACS setting.

**Prerequisites.** An environment that defines an adversary (a disturbance action) and a
concatenated action space; see [Adapt a Gymnasium environment → two-player](adapt-environment.md#4-two-player-adversarial-action-space).

## The concatenated action

The `*2P` learners take a **single** action `Box(ctrl_dim + dstb_dim)`, split at
`ctrl_action_dim`. You must pass `ctrl_action_dim` at construction:

```python
from safety_sb3 import SafetyPPO2P, ReachAvoidPPO2P

# two-player avoid (ISAACS) — no target, no terminal_type
model = SafetyPPO2P("MlpPolicy", adv_env, ctrl_action_dim=2)

# two-player reach-avoid (Gameplay Filters)
model = ReachAvoidPPO2P("MlpPolicy", adv_env, ctrl_action_dim=12, terminal_type="all")
model.learn(5_000_000)
```

`self.policy` is always the **control** policy, so `predict()`, `save()`, and downstream
filter wrappers see the deployable controller — the disturbance actor stays internal.

## `*PPO2P` and `*SAC2P` are not the same algorithm

Both hold two actors, and it is tempting to read them as one algorithm with two
optimizers. They are not — the difference is forced by **where each family's critic takes
its action**:

|  | `*SAC2P` | `*PPO2P` |
|---|---|---|
| critic | **one** twin critic `Q(s, [a_ctrl, a_dstb])` | **two** independent `V(s)` |
| data | one replay buffer | two rollout buffers |
| scheduling | continuous updates, no phases | a phase machine (dstb pretrain, then K/M cycles) |
| what it solves | **the minimax game as ISAACS formulates it** | **an alternating best-response approximation** |

Because SAC's critic takes the action, that single `Q` *is* the game value: both actors
differentiate through the same object (ctrl ascends it, dstb descends it), nothing needs
scheduling, and replay makes the learner indifferent to whose policy generated the data.
PPO's critic is state-only, so it cannot represent "the value of this joint action" —
each player needs its own advantage, its own importance ratio, and (PPO being on-policy)
its own freshly collected data. That is what the phase machine exists for, and why SAC
needs no phases.

So they are **not interchangeable**, and results from one do not transfer to the other.
There is deliberately no cross-family `AbstractTwoPlayer` base: the two differ in their
*state* — one critic vs two, one buffer vs two, phases vs none — not merely in their
update loops.

## The self-play league (leaderboard)

To damp cycling, each learner archives past opponents and samples from them — but the two
families score the board differently:

- **`*SAC2P`** scores pairings with dedicated eval episodes and samples **one** archived
  opponent per rollout. It takes a `leaderboard_eval_env` (required to actually evaluate).
- **`*PPO2P`** has **no eval env** — it scores from training outcomes with an EMA and
  faces a whole **population** at once by assigning opponents to env slices.

Both use the shared `Leaderboard`; neither estimator is a special case of the other.

Enable and tune it via the constructor:

```python
from safety_sb3 import ReachAvoidSAC2P
model = ReachAvoidSAC2P(
    "MlpPolicy", adv_env, ctrl_action_dim=12,
    use_leaderboard=True,
    leaderboard_eval_env=eval_env,     # SAC2P only; a TensorVecEnv here is a big speedup
    leaderboard_freq=2_000_000,        # fire the league rarely
    n_eval_episodes=3,                  # few episodes per pairing
)
```

!!! warning "The league eval can dominate wall-clock"
    At scale the league eval can consume ~97% of wall-clock (≈500 FPS at 1024 envs with
    the old `leaderboard_freq=10_000` / `n_eval_episodes=10`). Three safety-neutral levers
    — raise `leaderboard_freq`, lower `n_eval_episodes`, and pass a `TensorVecEnv` eval
    env — took a 1024-env `ReachAvoidSAC2P` from ~500 to ~19,000 FPS (~30×). The league
    is a *relative* ranking, so fewer, rarer evaluations cost nothing in safety. Full
    knob table: [Hyperparameters → leaderboard](../hyperparameters.md#leaderboard-two-player-opponent-sampling).

## Per-agent controls

The two-player SAC learners expose per-network learning rates and per-actor entropy
temperatures; the PPO two-player learners expose per-player KL-adaptive learning rates.
See [Hyperparameters](../hyperparameters.md#per-agent-per-network-learning-rates-two-player-sac).

## Next steps

- [Hyperparameters](../hyperparameters.md) — the full two-player knob set and the
  applicability matrix.
- [Evaluate a policy](evaluate.md) — two-player evaluation.
- [The MAP convention](../map.md) — where `2P` sits in the naming law.
