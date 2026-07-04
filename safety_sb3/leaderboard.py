"""ISAACS leaderboard (policy archive) for the SB3 implementation.

Mirrors ``safe_adaptation_dev``'s leaderboard: a checkpoint archive of the best
control (max-player) and disturbance (min-player) actors, scored by a reach-avoid
metric evaluated against the *opponents* on the board, with worst-performer
eviction and softmax-rationality sampling of past disturbances into rollouts
(the off-policy "strategy-space regularization").

Score matrix ``board`` has shape ``(kc+1, kd+2)``:
  * rows ``0..kc-1`` = saved ctrl checkpoints; row ``-1`` = the current ctrl.
  * cols ``0..kd-1`` = saved dstb checkpoints; col ``-2`` = the current dstb;
    col ``-1`` = the dummy (no-disturbance) opponent.
``board[i, j]`` = reach-avoid metric of ctrl ``i`` vs dstb ``j`` (higher = ctrl
better / dstb weaker).  Control is a maximizer; disturbance a minimizer.
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

  # --- on-policy (IsaacsPPO) additions --------------------------------------

  def ema_score(self, ctrl_idx: int, dstb_idx: int, metric: float,
                beta: float = 0.1) -> None:
    """EMA score update — on-policy boards are filled from noisy per-rollout
    training outcomes rather than dedicated eval runs."""
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
