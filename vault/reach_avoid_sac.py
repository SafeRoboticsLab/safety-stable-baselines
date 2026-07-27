"""ReachAvoidSafetySAC -- extends safety_sb3.SafetySAC's discounted safety Bellman backup with a
target-set ("reach") term, without touching safety_sb3/ itself (that package stays exactly as-is
for the original analytic avoid-only pipeline; this is a vault-local subclass).

safety_sb3.SafetySAC.train() implements the discounted AVOID-only Bellman backup (Fisac et al.):
    v_to_go        = min(g(s'), Q(s', a'))
    target_q_values = (1-gamma)*g(s') + gamma*v_to_go        [at non-terminal transitions]
                    = g(s')                                   [at terminal transitions]
where g(s') is the env reward (the avoid margin at the post-transition state) and Q(s',a') is the
(twin-min, entropy-adjusted) target critic evaluated at the next action -- itself an estimate of
the value achievable by committing to a' at s'.

The standard discrete-time REACH-AVOID Bellman recursion (e.g. Fisac et al.'s HJ reach-avoid
formulation) replaces "the best achievable continuation value" with "the best of: already at the
target now, OR the achievable continuation value":
    V(x) = min( g(x), max( l(x), V(x_next) ) )
i.e. reaching the target set (l(x) >= 0) lets you "cash in" that value instead of having to keep
searching for a better continuation -- while g(x) still caps the value from above regardless (you
can't get credit for being near a stop if you're currently violating a constraint).

This subclass makes the minimal corresponding change to the SAME blended/discounted structure
already used by SafetySAC (preserving its convergence properties), swapping `next_q_values` for
`max(ls, next_q_values)`:
    v_to_go = min(gs, max(ls, next_q_values))
where ls = target_margin.target_margin(s') is recomputed ON THE FLY from the replay buffer's
already-stored next_observations (no buffer schema change needed -- target_margin is a pure,
cheap function of state). At a terminal transition (not_done=0), the target becomes
min(gs, ls) instead of gs alone, so an episode that ends by successfully reaching the target
records a positive value rather than just its raw (uncredited) avoid margin.

Jaime's original 2026-07-24 review marker, preserved verbatim:

NOTE (flagged for review, not independently verified against the literature): the exact blend
weights `(1-gamma)`/`gamma` are carried over unchanged from the avoid-only backup, generalizing
its structure with the minimal `max(ls, ...)` substitution. This is the most direct, minimal-risk
generalization preserving the original's contraction properties, but the precise discounted
reach-avoid Bellman used in any specific paper/derivation should be checked against this if exact
parity matters.

Resolution (2026-07-27): RSS 2021 Eq. 15 and Theorem 1 were checked
directly. Under the paper's negative-good to this repository's positive-good
sign flip, the immediate term is `min(ls, gs)`, which the executable already
used. `vault/tests/test_drabe_operator.py` on the preserved exploratory branch
records the tabular contraction and under-approximation witness; the defect
here was documentation, not the backup implementation.

Three additional, independently-togglable experiment factors (2026-07-24 training-acceleration
study, see reach_avoid_eval.py / train_reach_avoid.py for how these compose):

  avoid_value_model : if given a FROZEN, already-trained SafetySAC (e.g. the avoid-only fallback),
      gs is computed as that model's OWN value V_avoid(s') = Q_avoid(s', pi_avoid(s')) instead of
      the raw env-reward margin -- "reach-and-always-avoid": the avoid term becomes the (already
      learned) infinite-horizon avoid value rather than an instantaneous constraint margin. Per
      Jaime: equivalent to standard reach-avoid when the target set is controlled-invariant, and
      may accelerate learning since the avoid half of the problem is already solved and handed to
      the critic as a fixed, informative shaping signal instead of something it has to rediscover
      via bootstrapping. avoid_value_model is used strictly as a frozen oracle (no_grad, eval mode)
      -- it is never updated during this training run, regardless of whether ITS OWN weights also
      happen to be the initialization for the model being trained (see critic_warmup_steps).

  saturate_target : if True, pass l(s') through a smooth floor at -1 (identity for l>=0, tanh(l)
      for l<0 -- C1-continuous at 0 since d/dl[tanh(l)]|_0 = 1 = d/dl[l]|_0) before use in the
      Bellman recursion. Meant to test whether a target margin that's allowed to swing arbitrarily
      negative (e.g. -16, -33 for extreme states) lets the "reach" term dominate training too
      strongly relative to "avoid" -- capping its magnitude (while keeping a nonzero gradient,
      unlike a hard ReLU-style clip) may let the two objectives balance better.

  critic_warmup_steps : when warm-starting from an existing checkpoint's weights (both actor AND
      critic), freeze the actor's optimizer step for this many gradient steps so the critic can
      "policy-evaluate" the (unchanged) inherited actor under the NEW value target first --
      avoids the actor being yanked by noisy/adapting critic estimates before the critic has
      adjusted to the new regime (relevant especially when avoid_value_model changes what gs even
      means). The critic and entropy-coefficient updates proceed normally throughout; only the
      actor's own gradient step is gated. Counts from 0 at the start of THIS training run (not
      carried over via save/load).
"""
from __future__ import annotations

import numpy as np
import torch as th
import torch.nn.functional as F
from safety_sb3 import SafetySAC
from stable_baselines3.common.utils import polyak_update

from . import config as C


def _target_margin_torch(next_obs: th.Tensor) -> th.Tensor:
    """target_margin.target_margin, computed in torch directly on the (batch, 5) next_observations
    tensor already sitting in the replay buffer -- avoids a CPU/GPU round trip through numpy on
    every gradient step. Must stay numerically identical to target_margin.target_margin's formula
    (including theta -- see that module's docstring for why omitting it was wrong)."""
    v, theta, theta_dot, psi_dot = next_obs[:, 0], next_obs[:, 1], next_obs[:, 2], next_obs[:, 3]
    m_v = 1.0 - v.abs() / C.TARGET_V_STOP
    m_theta = 1.0 - theta.abs() / C.TARGET_THETA_STOP
    m_thd = 1.0 - theta_dot.abs() / C.TARGET_THETA_DOT_STOP
    m_psid = 1.0 - psi_dot.abs() / C.TARGET_PSI_DOT_STOP
    return th.minimum(th.minimum(m_v, m_theta), th.minimum(m_thd, m_psid)).reshape(-1, 1)


def _saturate_floor(l: th.Tensor) -> th.Tensor:
    """Smooth floor at -1: identity for l>=0, tanh(l) for l<0. See class docstring."""
    return th.where(l >= 0, l, th.tanh(l))


class ReachAvoidSafetySAC(SafetySAC):
    """SafetySAC with the target-set term folded into the Bellman backup, plus the three
    optional experiment factors documented in the module docstring."""

    def __init__(self, *args, avoid_value_model: SafetySAC | None = None,
                 saturate_target: bool = False, critic_warmup_steps: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.avoid_value_model = avoid_value_model
        if self.avoid_value_model is not None:
            self.avoid_value_model.policy.set_training_mode(False)
        self.saturate_target = saturate_target
        self.critic_warmup_steps = critic_warmup_steps
        self._warmup_grad_steps_done = 0

    def _avoid_value_torch(self, next_obs: th.Tensor) -> th.Tensor:
        """V_avoid(s') = Q_avoid(s', pi_avoid(s')) from the frozen avoid_value_model -- the
        "reach-and-always-avoid" substitute for the raw env-reward avoid margin."""
        with th.no_grad():
            action = self.avoid_value_model.actor(next_obs, deterministic=True)
            q = th.cat(self.avoid_value_model.critic(next_obs, action), dim=1)
            q_min, _ = th.min(q, dim=1, keepdim=True)
        return q_min

    def train(self, gradient_steps: int, batch_size: int) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            update_actor = self._warmup_grad_steps_done >= self.critic_warmup_steps
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            if self.use_sde:
                self.actor.reset_noise()

            if update_actor:
                actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
                log_prob = log_prob.reshape(-1, 1)

                ent_coef_loss = None
                if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                    ent_coef = th.exp(self.log_ent_coef.detach())
                    ent_coef_loss = -(self.log_ent_coef *
                                      (log_prob + self.target_entropy).detach()).mean()
                    ent_coef_losses.append(ent_coef_loss.item())
                else:
                    ent_coef = self.ent_coef_tensor
                ent_coefs.append(ent_coef.item())

                if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                    self.ent_coef_optimizer.zero_grad()
                    ent_coef_loss.backward()
                    self.ent_coef_optimizer.step()
            else:
                # critic warmup: keep the actor and entropy coefficient exactly as inherited.
                ent_coef = (th.exp(self.log_ent_coef.detach())
                           if self.log_ent_coef is not None else self.ent_coef_tensor)

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions), dim=1
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # NOTE (2026-07-27 audit): the entropy-adjusted continuation
                # below is not the unregularized discounted reach-avoid
                # operator from the theorem. A sufficiently large entropy
                # bonus can make the learned critic optimistic. Training
                # behavior is intentionally unchanged on this model-consumer
                # branch pending a separately authorized correction.
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)

                if self.avoid_value_model is not None:
                    gs = self._avoid_value_torch(replay_data.next_observations)
                else:
                    gs = replay_data.rewards                              # g(s'), the avoid margin
                ls = _target_margin_torch(replay_data.next_observations)   # l(s'), the reach margin
                if self.saturate_target:
                    ls = _saturate_floor(ls)
                not_done = 1.0 - replay_data.dones
                # v_to_go = min(g(s'), max(l(s'), V(s''))) -- the reach-avoid extension; see
                # module docstring for the avoid-only backup this generalizes.
                v_to_go = th.minimum(gs, th.maximum(ls, next_q_values))
                target_q_values = (
                    1.0 - self.gamma * not_done
                ) * th.minimum(gs, ls) + self.gamma * not_done * v_to_go
                # at terminal transitions (not_done=0) this reduces to min(gs, ls): a successful
                # reach-target termination (gs>=0, ls>=0) now records a positive value, instead of
                # the avoid-only backup's plain gs which discarded the reach credit entirely.

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, target_q_values) for current_q in current_q_values
            )
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if update_actor:
                q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
                min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
                actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
                actor_losses.append(actor_loss.item())

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

            self._warmup_grad_steps_done += 1

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs) if ent_coefs else float("nan"))
        self.logger.record("train/actor_loss", np.mean(actor_losses) if actor_losses else float("nan"))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
