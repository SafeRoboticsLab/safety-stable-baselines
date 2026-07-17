"""Multi-car eval rendering for BicycleGoal — 8 cars, one map, one GIF.

Rolls out a trained policy from 8 different spawn points on a fixed map and
animates them together, so a single GIF shows the policy's whole competence
envelope at a glance (which starts reach, which weave, which fail). Four preset
maps (:data:`EVAL_MAPS`) — all within the training layout distribution so they
render reliably — show generalization across obstacle placement and goal
position.

Used by ``examples/bicycle5d_demo.py`` to log a panel of GIFs to wandb during
training; importable on its own for ad-hoc visualization.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .bicycle5d import CAR_L, CAR_W, BicycleGoal

#: Four fixed eval maps (2 obstacles each, within the training jitter so the
#: policy is in-distribution). Obstacle slot 0 stays "upper" (+y ~0.3), slot 1
#: "lower" (-y ~0.4) — the obs is NOT permutation-invariant across slots, so
#: eval maps must respect the slot convention the policy trained on.
EVAL_MAPS = [
  dict(name="nominal",
       obstacles=[(1.00, 0.32, 0.28), (2.00, -0.40, 0.28)], goal=(3.00, 0.00, 0.40)),
  dict(name="wide-gap",
       obstacles=[(0.90, 0.55, 0.30), (2.10, -0.60, 0.30)], goal=(3.00, 0.20, 0.40)),
  dict(name="tight",
       obstacles=[(1.15, 0.10, 0.30), (1.85, -0.15, 0.30)], goal=(3.00, 0.00, 0.40)),
  dict(name="goal-right",
       obstacles=[(1.10, 0.35, 0.26), (2.00, -0.30, 0.26)], goal=(3.10, 0.45, 0.40)),
]

_STATUS_COLOR = {"reached": "#27ae60", "collided": "#c0392b",
                 "timeout": "#7f8c8d", "active": "#2980b9"}


def value_fn_of(model):
  """A ``obs (N, d) -> V (N,)`` callable for the value heatmap, per algo family.

  PPO learns the state value ``V(s)`` directly (``predict_values``). SAC learns
  ``Q(s, a)``; its state value is ``min_i Q_i(s, π(s))`` at the policy's
  (deterministic) action -- which is exactly why the SAC value is calibrated
  off the on-policy tube (Q is fit over the action space) where the on-policy
  PPO value extrapolates.
  """
  import torch as th
  dev = model.policy.device
  if hasattr(model, "critic") and hasattr(model, "actor"):     # SAC family
    def vf(obs):
      with th.no_grad():
        o = th.as_tensor(obs, dtype=th.float32, device=dev)
        a = model.actor(o, deterministic=True)
        q = th.cat(model.critic(o, a), dim=1)
        return q.min(dim=1).values.cpu().numpy().ravel()
    return vf
  def vf(obs):                                                 # PPO family
    with th.no_grad():
      return model.policy.predict_values(
        th.as_tensor(obs, dtype=th.float32, device=dev)).cpu().numpy().ravel()
  return vf


#: 8 coverage anchors spread over the whole map (left approach, the flanks
#: above/below the obstacle band, and near the goal), so the cars together show
#: the policy's behavior from EVERY region — not just the entry edge.
_COVERAGE_PTS = [
  (0.10, -0.90), (0.10, 0.00), (0.10, 0.90),   # left approach, spread in y
  (1.20, -1.00), (1.20, 1.00),                 # below / above the obstacle band
  (2.20, -0.70), (2.20, 0.70),                 # flanks near the goal
  (2.60, 0.00),                                # right in front of the goal
]


def _coverage_spawns(map_cfg: dict, n_cars: int = 8):
  """Spawn poses covering the whole map, each facing the goal and nudged out
  of any obstacle it lands in (the 4 maps place obstacles differently, so a
  fixed grid point can fall inside one)."""
  gx, gy, _ = map_cfg["goal"]
  obst = map_cfg["obstacles"]
  out = []
  for x, y in _COVERAGE_PTS[:n_cars]:
    for _ in range(25):                        # push clear of every obstacle
      hit = None
      for ox, oy, r in obst:
        d = float(np.hypot(x - ox, y - oy))
        if d < r + 0.35:                       # obstacle radius + car footprint
          hit = (ox, oy, r, d); break
      if hit is None:
        break
      ox, oy, r, d = hit
      nx, ny = ((x - ox) / d, (y - oy) / d) if d > 1e-6 else (1.0, 0.0)
      x, y = ox + nx * (r + 0.36), oy + ny * (r + 0.36)
    out.append((float(x), float(y), float(np.arctan2(gy - y, gx - x))))
  return out


def multi_car_rollout(model, map_cfg: dict, adversary: bool = False,
                      n_cars: int = 8, max_steps: int = 300):
  """Roll ``n_cars`` policies from :func:`_spawns` on ``map_cfg``.

  Returns ``(trajs, status)``: ``trajs[i]`` is an ``(T+1, 5)`` state trajectory
  (finished cars hold their final state), ``status[i]`` in
  {reached, collided, timeout}.
  """
  envs = [BicycleGoal(adversary=adversary, randomize=False,
                      obstacles=map_cfg["obstacles"], goal=map_cfg["goal"])
          for _ in range(n_cars)]
  obs = []
  for e, (sx, sy, sp) in zip(envs, _coverage_spawns(map_cfg, n_cars)):
    e.reset(seed=0)
    e.s = np.array([sx, sy, 0.0, sp, 0.0])     # fixed spawn, from standstill
    obs.append(e._obs(e.s))
  trajs = [[e.s.copy()] for e in envs]
  status = ["active"] * n_cars

  for _ in range(max_steps):
    acts, _ = model.predict(np.asarray(obs, np.float32), deterministic=True)
    for i, e in enumerate(envs):
      if status[i] != "active":
        trajs[i].append(e.s.copy())
        continue
      a = acts[i]
      if adversary:
        a = np.concatenate([np.asarray(a).reshape(-1)[:2], np.zeros(5)])
      o, g, term, trunc, info = e.step(a)
      obs[i] = o
      trajs[i].append(e.s.copy())
      if info["collided"]:
        status[i] = "collided"
      elif info["reached"]:
        status[i] = "reached"
      elif trunc:
        status[i] = "timeout"
    if all(s != "active" for s in status):
      break
  return [np.asarray(t) for t in trajs], status


def render_frames(trajs: List[np.ndarray], status: List[str], map_cfg: dict,
                  stride: int = 3, title: Optional[str] = None,
                  fig=None, ax=None) -> List[np.ndarray]:
  """Animate the rollout: list of ``(H, W, 3)`` uint8 frames for a GIF."""
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.patches as mp
  import matplotlib.pyplot as plt
  from matplotlib.transforms import Affine2D

  own = fig is None
  if own:
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=64)
  cmap = plt.get_cmap("tab10")
  T = max(len(t) for t in trajs)
  gx, gy, gr = map_cfg["goal"]
  frames = []
  for f in range(0, T, stride):
    ax.clear()
    for ox, oy, r in map_cfg["obstacles"]:
      ax.add_patch(mp.Circle((ox, oy), r, color="#c0392b", alpha=0.85, zorder=2))
    ax.add_patch(mp.Circle((gx, gy), gr, color="#27ae60", alpha=0.40, zorder=1))
    for i, tr in enumerate(trajs):
      k = min(f, len(tr) - 1)
      col = cmap(i % 10)
      seg = tr[:k + 1]
      ax.plot(seg[:, 0], seg[:, 1], "-", color=col, lw=1.1, alpha=0.7, zorder=3)
      x, y, _, psi, _ = tr[k]
      done = status[i] if k == len(tr) - 1 else "active"
      edge = _STATUS_COLOR.get(done, "#2980b9")
      car = mp.Rectangle((-CAR_L / 2, -CAR_W / 2), CAR_L, CAR_W,
                         facecolor=col, edgecolor=edge, lw=1.6, zorder=4)
      car.set_transform(Affine2D().rotate(psi).translate(x, y) + ax.transData)
      ax.add_patch(car)
    ax.set_aspect("equal")
    ax.set_xlim(-0.9, gx + 1.0)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks([]); ax.set_yticks([])
    reached = sum(s == "reached" for s in status)
    coll = sum(s == "collided" for s in status)
    ttl = f"{map_cfg['name']}"
    if title:
      ttl = f"{title} — {ttl}"
    ax.set_title(f"{ttl}   reached {reached}/{len(trajs)}, collided {coll}",
                 fontsize=9)
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
  if own:
    plt.close(fig)
  return frames


def map_gif(model, map_cfg: dict, adversary: bool = False, title: Optional[str] = None,
            fig=None, ax=None):
  """Convenience: rollout + render one map's 8-car GIF. Returns frame list."""
  trajs, status = multi_car_rollout(model, map_cfg, adversary=adversary)
  return render_frames(trajs, status, map_cfg, title=title, fig=fig, ax=ax)


def _value_grid(model, map_cfg, nx, ny, v, delta, face_goal):
  """(X, Y, V) of the learned value swept over car position on ``map_cfg``."""
  gx, gy, gr = map_cfg["goal"]
  xs = np.linspace(-0.9, gx + 1.0, nx)
  ys = np.linspace(-1.5, 1.5, ny)
  X, Y = np.meshgrid(xs, ys)
  e = BicycleGoal(randomize=False, obstacles=map_cfg["obstacles"], goal=map_cfg["goal"])
  e.reset(seed=0)
  obs = np.empty((X.size, e.observation_space.shape[0]), np.float32)
  for i, (xx, yy) in enumerate(zip(X.ravel(), Y.ravel())):
    psi = float(np.arctan2(gy - yy, gx - xx)) if face_goal else 0.0
    e.s = np.array([xx, yy, v, psi, delta])
    obs[i] = e._obs(e.s)
  return X, Y, value_fn_of(model)(obs).reshape(ny, nx)


def _draw_value(ax, X, Y, V, map_cfg, vmax, title):
  import matplotlib.patches as mp
  gx, gy, gr = map_cfg["goal"]
  im = ax.pcolormesh(X, Y, V, cmap="RdYlGn", vmin=-vmax, vmax=vmax, shading="auto")
  try:
    ax.contour(X, Y, V, levels=[0.0], colors="k", linewidths=1.5)   # V=0 boundary
  except Exception:
    pass
  for ox, oy, r in map_cfg["obstacles"]:
    ax.add_patch(mp.Circle((ox, oy), r, fill=False, edgecolor="k", lw=1.4))
  ax.add_patch(mp.Circle((gx, gy), gr, fill=False, edgecolor="#1155cc", lw=1.8, ls="--"))
  ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
  ax.set_title(title, fontsize=9)
  return im


def compare_value_maps(models: dict, map_cfg: dict, nx: int = 90, ny: int = 66,
                       v: float = 0.0, delta: float = 0.0, face_goal: bool = True,
                       suptitle: Optional[str] = None):
  """Side-by-side value heatmaps for several models (e.g. {'PPO':..,'SAC':..})
  on ``map_cfg``, SHARED color scale so they are directly comparable. Returns a
  matplotlib Figure. V>=0 (green, inside the black contour) is the learned
  reachable-safe set."""
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  grids = {name: _value_grid(m, map_cfg, nx, ny, v, delta, face_goal)
           for name, m in models.items()}
  vmax = max(float(np.abs(V).max()) for _, _, V in grids.values()) or 1.0
  n = len(models)
  fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.4), dpi=64, squeeze=False)
  im = None
  for ax, (name, (X, Y, V)) in zip(axes[0], grids.items()):
    im = _draw_value(ax, X, Y, V, map_cfg, vmax,
                     f"{name}   V in [{V.min():+.2f}, {V.max():+.2f}]")
  fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02)
  fig.suptitle(suptitle or f"value(x,y) — {map_cfg['name']}", fontsize=10)
  return fig


def value_map(model, map_cfg: dict, nx: int = 90, ny: int = 66,
              v: float = 0.0, delta: float = 0.0, face_goal: bool = True,
              title: Optional[str] = None):
  """Heatmap of the learned value V(x, y) over the map (a matplotlib Figure).

  V is swept over car POSITION on a grid, holding speed ``v`` and steering
  ``delta`` fixed and (by default) the heading pointed at the goal — i.e. "the
  value of being here, ready to drive to the goal". For a reach-avoid critic
  ``V >= 0`` is the learned certificate (can reach the goal without hitting
  anything), so the ``V = 0`` contour is the reachable-safe set boundary;
  obstacles read as negative wells, the goal as the high-value basin. For an
  avoid critic V is the viability value (safe almost everywhere except the
  obstacle wells) — the visible contrast with reach-avoid.
  """
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  gx, gy, gr = map_cfg["goal"]
  X, Y, V = _value_grid(model, map_cfg, nx, ny, v, delta, face_goal)

  fig, ax = plt.subplots(figsize=(6, 3.4), dpi=64)
  vmax = float(np.abs(V).max()) or 1.0
  im = ax.pcolormesh(X, Y, V, cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                     shading="auto")
  # the V = 0 contour = the reachable-safe boundary
  try:
    ax.contour(X, Y, V, levels=[0.0], colors="k", linewidths=1.5)
  except Exception:
    pass
  import matplotlib.patches as mp
  for ox, oy, r in map_cfg["obstacles"]:
    ax.add_patch(mp.Circle((ox, oy), r, fill=False, edgecolor="k", lw=1.4))
  ax.add_patch(mp.Circle((gx, gy), gr, fill=False, edgecolor="#1155cc",
                         lw=1.8, ls="--"))
  ax.set_aspect("equal")
  ax.set_xticks([]); ax.set_yticks([])
  ttl = f"V(x,y)  {map_cfg['name']}"
  ax.set_title(f"{title} — {ttl}" if title else ttl, fontsize=9)
  fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
  fig.tight_layout(pad=0.2)
  return fig
