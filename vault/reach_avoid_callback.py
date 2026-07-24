"""ReachAvoidCheckpointEval -- an SB3 training callback that periodically saves a checkpoint and
evaluates the CURRENT policy's mean realized reach-avoid score (reach_avoid_value.realized_value)
plus outcome fractions (reached/safe-no-reach/failed) on a FIXED sample of constraint-set states
(same random seed every time, so successive checkpoints are compared on the same states -- low
variance, isolates genuine learning progress from sampling noise).

Produces a learning-curve-ready history (JSON, written incrementally so nothing is lost if
training is interrupted) of {step, mean_score, reached, safe_no_reach, failed} records.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .mujoco_plant import MujocoPlant
from .reach_avoid_eval import _sample_constraint_set_state, rollout_outcome
from .safety_filter import SACFallback


class ReachAvoidCheckpointEval(BaseCallback):
    def __init__(self, ckpt_dir: str, eval_every: int = 10_000, n_eval: int = 25,
                 horizon: int = 300, mu: float = 0.8, eval_seed: int = 123, verbose: int = 1):
        super().__init__(verbose)
        self.ckpt_dir = Path(ckpt_dir)
        self.eval_every = eval_every
        self.n_eval = n_eval
        self.horizon = horizon
        self.mu = mu
        self.eval_seed = eval_seed
        self.history: list[dict] = []

    def _init_callback(self) -> None:
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_every == 0:
            self._checkpoint_and_eval()
        return True

    def _checkpoint_and_eval(self) -> None:
        self.model.save(str(self.ckpt_dir / f"step_{self.num_timesteps}"))

        rng = np.random.default_rng(self.eval_seed)   # same seed every call -> same eval states
        plant = MujocoPlant(wheel="cylinder", mu=self.mu, dt=0.0005, substeps=20,
                            contact_geometry=True)
        fb = SACFallback(self.model)
        outcomes = {"reached": 0, "safe_no_reach": 0, "failed": 0}
        scores = []
        while len(scores) < self.n_eval:
            x0 = _sample_constraint_set_state(rng, plant)
            if x0 is None:
                continue
            outcome, _, score = rollout_outcome(plant, fb, x0, self.mu, self.horizon)
            outcomes[outcome] += 1
            scores.append(score)

        record = {"step": int(self.num_timesteps), "mean_score": float(np.mean(scores)), **outcomes}
        self.history.append(record)
        if self.verbose:
            print(f"[checkpoint-eval @ {self.num_timesteps}] mean_score={record['mean_score']:+.3f} "
                  f"reached={outcomes['reached']}/{self.n_eval} "
                  f"safe_no_reach={outcomes['safe_no_reach']}/{self.n_eval} "
                  f"failed={outcomes['failed']}/{self.n_eval}")
        (self.ckpt_dir / "history.json").write_text(json.dumps(self.history, indent=2))
