"""The league: a policy archive shared by both two-player families.

Mirrors ``safe_adaptation_dev``'s leaderboard: a checkpoint archive of the best
control (max-player) and disturbance (min-player) actors, scored against the
*opponents* on the board, with worst-performer eviction and softmax-rationality
sampling of past disturbances into rollouts (the "strategy-space
regularization" that damps cycling).

Score matrix ``board`` has shape ``(kc+1, kd+2)``:
  * rows ``0..kc-1`` = saved ctrl checkpoints; row ``-1`` = the current ctrl.
  * cols ``0..kd-1`` = saved dstb checkpoints; col ``-2`` = the current dstb;
    col ``-1`` = the dummy (no-disturbance) opponent.
``board[i, j]`` = success metric of ctrl ``i`` vs dstb ``j`` (higher = ctrl
better / dstb weaker).  Control is a maximizer; disturbance a minimizer.

Both two-player families use :class:`Leaderboard`, but they FILL it differently
and that difference is deliberate, not an oversight:

* the **off-policy (SAC) league** scores with dedicated evaluation episodes on a
  separate eval env (:class:`LeagueEvaluator` below), writes cells with
  :meth:`Leaderboard.set_score` (an overwrite), and samples ONE archived
  opponent per rollout via :meth:`Leaderboard.sample_dstb_step`;
* the **on-policy (PPO) league** has no eval env at all. It scores from the
  *training* rollout outcomes it already produced, which are noisy, so it
  accumulates them with :meth:`Leaderboard.ema_score`, and it faces a whole
  POPULATION of opponents at once by assigning them to env slices via
  :meth:`Leaderboard.sample_dstb_slices`.

Neither is a special case of the other: dedicated-eval + overwrite + one
opponent is not the same estimator as training-outcome + EMA + population, and
on-policy learning cannot reuse another opponent's data the way replay can. So
this module offers both vocabularies rather than forcing one.
"""

from __future__ import annotations

import os

import numpy as np
import torch as th


class Leaderboard:
  def __init__(
    self,
    save_top_k_ctrl: int,
    save_top_k_dstb: int,
    softmax_rationality: float,
    model_dir: str,
    seed: int = 0,
  ) -> None:
    self.kc = int(save_top_k_ctrl)
    self.kd = int(save_top_k_dstb)
    self.rationality = float(softmax_rationality)
    self.dir = model_dir
    os.makedirs(self.dir, exist_ok=True)
    self.rng = np.random.default_rng(seed)
    self.ctrl_steps: list[int] = []  # parallel to board rows 0..kc-1
    self.dstb_steps: list[int] = []  # parallel to board cols 0..kd-1
    self.board = np.full((self.kc + 1, self.kd + 2), np.nan, dtype=float)

  # --- checkpoint io ---
  def _path(self, kind: str, step: int) -> str:
    return os.path.join(self.dir, f"{kind}_{step}.pt")

  def save_actor(self, actor: th.nn.Module, kind: str, step: int) -> None:
    th.save(actor.state_dict(), self._path(kind, step))

  def load_actor(self, actor: th.nn.Module, kind: str, step: int) -> th.nn.Module:
    actor.load_state_dict(th.load(self._path(kind, step), map_location="cpu"))
    return actor

  def _remove(self, kind: str, step: int) -> None:
    p = self._path(kind, step)
    if os.path.exists(p):
      os.remove(p)

  # --- score matrix ---
  def set_score(self, ctrl_idx: int, dstb_idx: int, metric: float) -> None:
    """Overwrite a cell — used when the score came from a dedicated eval run
    (contrast :meth:`ema_score`)."""
    self.board[ctrl_idx, dstb_idx] = metric

  def prune(self, step: int, ctrl_actor: th.nn.Module, dstb_actor: th.nn.Module) -> None:
    """Admit the current ctrl/dstb (row -1 / col -2) into the archive, evicting
    the worst performer when full.  Matches ``safe_adaptation_dev.prune_leaderboard``."""
    # --- control (maximizer): evict the lowest-average-metric checkpoint ---
    if len(self.ctrl_steps) == self.kc:
      ctrl_avg = np.nanmean(self.board, axis=1)  # over all dstb cols
      worst = int(np.argmin(ctrl_avg))
      if worst != self.kc:  # current beats some saved ctrl -> replace it
        self._remove("ctrl", self.ctrl_steps[worst])
        self.ctrl_steps[worst] = step
        self.board[worst] = self.board[-1]
        self.save_actor(ctrl_actor, "ctrl", step)
    else:
      self.ctrl_steps.append(step)
      self.save_actor(ctrl_actor, "ctrl", step)

    # --- disturbance (minimizer): evict the highest-average-metric checkpoint ---
    if len(self.dstb_steps) == self.kd:
      dstb_avg = np.nanmean(self.board[:, :-1], axis=0)  # exclude dummy col
      worst = int(np.argmax(dstb_avg))
      if worst != self.kd:  # current is more challenging than some saved dstb
        self._remove("dstb", self.dstb_steps[worst])
        self.dstb_steps[worst] = step
        self.board[:, worst] = self.board[:, -2]
        self.save_actor(dstb_actor, "dstb", step)
    else:
      self.dstb_steps.append(step)
      self.save_actor(dstb_actor, "dstb", step)

  # --- rollout sampling (softmax over how effective each dstb is) ---
  def sample_dstb_step(self) -> int | None:
    """Return a saved dstb step to roll out against, or ``None`` for the current
    dstb / dummy.  Probability ∝ exp(-rationality · avg_metric): a disturbance
    that drives the control's reach-avoid value *down* (low metric) is sampled
    more often.  Falls back to the current dstb when the board is empty."""
    n = len(self.dstb_steps)
    if n == 0 or not self.ctrl_steps:
      return None
    # choices: the n saved dstb cols, plus the dummy (-1); col -2 = "current".
    cols = list(range(n)) + [self.board.shape[1] - 1]
    with np.errstate(invalid="ignore"):
      logit = np.nanmean(self.board[: len(self.ctrl_steps)][:, cols], axis=0)
      fill = float(np.nanmean(self.board)) if np.isfinite(self.board).any() else 0.0
    logit = np.nan_to_num(logit, nan=fill)
    p = np.exp(-self.rationality * logit)
    p = p / p.sum()
    pick = int(self.rng.choice(len(cols), p=p))
    if cols[pick] == self.board.shape[1] - 1:
      return None  # dummy / current dstb (no archived checkpoint)
    return self.dstb_steps[pick]

  # --- on-policy (PPO 2P) additions -----------------------------------------

  def ema_score(self, ctrl_idx: int, dstb_idx: int, metric: float,
                beta: float = 0.1) -> None:
    """EMA score update — the on-policy board is filled from noisy per-rollout
    TRAINING outcomes rather than from dedicated eval runs, so cells are
    accumulated instead of overwritten (contrast :meth:`set_score`)."""
    old = self.board[ctrl_idx, dstb_idx]
    if np.isnan(old):
      self.board[ctrl_idx, dstb_idx] = metric
    else:
      self.board[ctrl_idx, dstb_idx] = (1.0 - beta) * old + beta * metric

  def sample_dstb_slices(self, n_slices: int) -> list[int]:
    """Opponent assignment for one on-policy rollout over env SLICES.

    Returns a list of length ``n_slices`` with entries: ``-3`` = ZERO (dummy
    opponent; also anchors nominal behavior), ``-2`` = RANDOM, ``-1`` =
    CURRENT dstb, ``>= 0`` = index into ``dstb_steps`` (archived checkpoint).
    Slices 0 and 1 are always ZERO and RANDOM; the rest are softmax-sampled
    with ``p ∝ exp(-rationality * avg_metric)`` over {archived, CURRENT}.
    """
    out = [-3, -2]
    n_free = max(n_slices - 2, 0)
    n_arch = len(self.dstb_steps)
    if n_arch == 0:
      return (out + [-1] * n_free)[:n_slices]
    cols = list(range(n_arch)) + [self.board.shape[1] - 2]
    with np.errstate(invalid="ignore"):
      logit = np.nanmean(self.board[:, cols], axis=0)
      fill = float(np.nanmean(self.board)) if np.isfinite(self.board).any() else 0.0
    logit = np.nan_to_num(logit, nan=fill)
    p = np.exp(-self.rationality * logit)
    p = p / p.sum()
    for _ in range(n_free):
      pick = int(self.rng.choice(len(cols), p=p))
      out.append(-1 if cols[pick] == self.board.shape[1] - 2 else pick)
    return out[:n_slices]


class LeagueEvaluator:
  """Scores one ``(ctrl, dstb)`` pair by rolling out dedicated eval episodes.

  This is the off-policy league's estimator, lifted out of the algorithm so that
  everything about the league lives in this module. It picks the cheapest of
  three equivalent rollout paths automatically:

  * ``step_tensor`` on a GPU-resident env — everything on device, no numpy
    ``VecEnv`` and no per-step host<->device sync. ~10-50x faster than the numpy
    path and the main league-throughput fix (profiling: the numpy vectorized path
    cost ~100s per refresh);
  * a parallel numpy ``VecEnv`` — one batch = ``num_envs`` first-episodes;
  * a single raw ``gym.Env`` loop.

  All three share the success semantics, which the caller supplies:

  :param success: ``(safe, reached) -> won``, evaluated elementwise. The avoid
      game passes ``safe`` (survival IS the win — there is no target set); the
      reach-avoid game passes ``safe & reached``. Passing the rule in is what
      lets the two modes share one evaluator without a mode flag inside it.
  :param obs_normalizer: zero-arg callable returning the LIVE training
      normalizer (or ``None``). Only the tensor path needs it: it rolls out on a
      RAW tensor env, so observations must be normalized with the same running
      stats the actors were trained under. The margins ``g``/``l`` are physical
      (unnormalized) and drive safe/reached either way.
  """

  MAX_STEPS = 400

  def __init__(self, env, device, n_eval_episodes: int, dstb_action_dim: int,
               unscale_action, success, obs_normalizer=None) -> None:
    self.env = env
    self.device = device
    self.n_eval_episodes = int(n_eval_episodes)
    self.dn = int(dstb_action_dim)
    self.unscale_action = unscale_action
    self.success = success
    self.obs_normalizer = obs_normalizer

  @th.no_grad()
  def score(self, ctrl_actor, dstb_actor) -> float:
    """Success rate of ``ctrl`` vs ``dstb`` (``None`` = dummy / no disturbance)."""
    env = self.env
    if getattr(env, "is_tensor_env", False):
      return self._score_tensor(env, ctrl_actor, dstb_actor)
    if hasattr(env, "num_envs") and hasattr(env, "step_async"):
      return self._score_vec(env, ctrl_actor, dstb_actor)

    dn = self.dn
    succ = 0
    for _ in range(self.n_eval_episodes):
      obs, _ = env.reset()
      done = reached = False
      safe = True
      while not done:
        ot = th.as_tensor(np.asarray(obs), dtype=th.float32,
                          device=self.device).reshape(1, -1)
        c = ctrl_actor(ot, deterministic=True).cpu().numpy()[0]
        d = (np.zeros(dn, np.float32) if dstb_actor is None
             else dstb_actor(ot, deterministic=True).cpu().numpy()[0])
        scaled = np.concatenate([c, d]).astype(np.float32)  # [-1, 1]
        action = self.unscale_action(scaled[None])[0]
        obs, g, term, trunc, info = env.step(action)
        done = bool(term or trunc)
        if g < 0:
          safe = False
        if float(info.get("l_x", -1.0)) >= 0:
          reached = True
      succ += int(self.success(safe, reached))
    return succ / max(self.n_eval_episodes, 1)

  def _score_vec(self, env, ctrl_actor, dstb_actor) -> float:
    """Parallel success over ``num_envs`` * ``n_eval_episodes`` first-episodes."""
    dn = self.dn
    n_env = env.num_envs
    total_succ, total = 0, 0
    for _ in range(max(1, self.n_eval_episodes)):
      obs = env.reset()
      ep_safe = np.ones(n_env, dtype=bool)
      ep_reached = np.zeros(n_env, dtype=bool)
      done_once = np.zeros(n_env, dtype=bool)
      for _ in range(self.MAX_STEPS):
        ot = th.as_tensor(np.asarray(obs), dtype=th.float32, device=self.device)
        c = ctrl_actor(ot, deterministic=True).cpu().numpy()
        d = (np.zeros((n_env, dn), np.float32) if dstb_actor is None
             else dstb_actor(ot, deterministic=True).cpu().numpy())
        env.step_async(np.concatenate([c, d], axis=1).astype(np.float32))
        obs, g, dones, infos = env.step_wait()
        active = ~done_once
        ep_safe &= ~(active & (np.asarray(g) < 0))
        lx = np.array([i.get("l_x", -1.0) for i in infos], dtype=np.float32)
        ep_reached |= active & (lx >= 0)
        done_once |= active & np.asarray(dones, dtype=bool)
        if done_once.all():
          break
      hits = self.success(ep_safe, ep_reached)
      total_succ += int(hits.sum())
      total += n_env
    return total_succ / max(total, 1)

  @th.no_grad()
  def _score_tensor(self, env, ctrl_actor, dstb_actor) -> float:
    """GPU-resident twin of :meth:`_score_vec`. Actors output in [-1, 1] and the
    env clamps, so no unscaling is needed here."""
    dn = self.dn
    dev = env.device
    n = env.num_envs
    norm = self.obs_normalizer() if self.obs_normalizer is not None else None
    total_succ, total = 0, 0
    for _ in range(max(1, self.n_eval_episodes)):
      raw = env.reset()
      if not th.is_tensor(raw):
        raw = th.as_tensor(np.asarray(raw), dtype=th.float32, device=dev)
      ep_safe = th.ones(n, dtype=th.bool, device=dev)
      ep_reached = th.zeros(n, dtype=th.bool, device=dev)
      done_once = th.zeros(n, dtype=th.bool, device=dev)
      for _ in range(self.MAX_STEPS):
        obs = norm.normalize_obs(raw) if norm is not None else raw
        c = ctrl_actor(obs, deterministic=True)
        d = (th.zeros((n, dn), device=dev) if dstb_actor is None
             else dstb_actor(obs, deterministic=True))
        raw, g, dones, _timeouts, l_x = env.step_tensor(th.cat([c, d], dim=1))
        active = ~done_once
        ep_safe &= ~(active & (g.reshape(n) < 0))
        ep_reached |= active & (l_x.reshape(n) >= 0)
        done_once |= active & dones.reshape(n).bool()
        if bool(done_once.all()):
          break
      hits = self.success(ep_safe, ep_reached)
      total_succ += int(hits.sum().item())
      total += n
    return total_succ / max(total, 1)
