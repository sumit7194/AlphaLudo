"""V15RichTrainer — PPO training step for V15 policy.

Mirrors `ActorCriticTrainerV10`'s loss formulae from
`td_ludo/td_ludo/training/trainer_v10.py` but adapted for V15:
  - 225-way source-cell policy (vs 4-way rank-indexed)
  - 8-frame history state shape (B, 8, 15, 15, 3) (vs single-frame (B, 21, 15, 15))
  - No moves_remaining / progress aux heads (V15 dropped them per design)
  - No search aux loss

What's preserved:
  - Monte-Carlo discounted return (γ=0.999, +0.40 score-event reward, ±1 terminal)
  - EMA running mean/std return normalization
  - PPO clipped surrogate with ratio clamp safety
  - Win-prob BCE (sigmoid head → BCE target = 1.0 if won else 0.0)
  - Entropy bonus
  - Pre-update advantage computation once per buffer flush
  - Optional KL anchor to V15 SL (V13.5 trainer didn't use KL; we keep it as
    a safety net to prevent policy drift early in RL)
"""
from __future__ import annotations

import collections
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


GAMMA = 0.999
SCORE_REWARD = 0.40  # +reward per token scored
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
RETURN_EMA_ALPHA = 0.01


class V15RichTrainer:
    """PPO trainer for V15 GraphTransformer.

    Usage:
        trainer = V15RichTrainer(model, device, ...)
        # For each completed game:
        trainer.train_on_game(trajectory, winner, model_player)
        # When buffer is full, PPO update fires automatically and returns metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-5,
        ppo_clip: float = 0.2,
        ppo_epochs: int = 3,
        ppo_buffer_games: int = 64,
        ppo_minibatch_size: int = 256,
        entropy_coeff: float = 0.03,
        win_bce_coeff: float = 0.5,
        kl_anchor_coeff: float = 0.0,
        kl_anchor_model: Optional[nn.Module] = None,
        use_gae: bool = False,
        gae_lambda: float = 0.95,
        terminal_only: bool = False,
        value_target: str = "terminal",
    ):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY,
        )
        self.ppo_clip = ppo_clip
        self.ppo_epochs = ppo_epochs
        self.ppo_buffer_games = ppo_buffer_games
        self.ppo_minibatch_size = ppo_minibatch_size
        self.entropy_coeff = entropy_coeff
        self.win_bce_coeff = win_bce_coeff
        self.kl_anchor_coeff = kl_anchor_coeff
        self.kl_anchor_model = kl_anchor_model
        # GAE (ported from v13.6 trainer_v10; Exp 58). use_gae=False → legacy MC.
        self.use_gae = bool(use_gae)
        self.gae_lambda = float(gae_lambda)
        # terminal_only: zero every per-step reward, train on ±1 win/loss only
        # (the "best pipeline" from v13.6: terminal-only + GAE).
        self.terminal_only = bool(terminal_only)
        # value_target: "terminal" = BCE to binary win/loss (high variance + bias,
        # the original). "lambda" = MSE to the GAE bootstrapped λ-return
        # (value+advantage) — lower variance, the squeeze lever (Exp 62).
        self.value_target = str(value_target)

        # PPO buffering
        self._ppo_buffer: List[dict] = []
        self._ppo_games_buffered = 0
        self._traj_lengths: List[int] = []   # per-game own-decision count (GAE)

        # EMA return normalization
        self._return_running_mean = 0.0
        self._return_running_std = 1.0
        self._return_alpha = RETURN_EMA_ALPHA

        # Counters
        self.total_games = 0
        self.total_updates = 0

        # Rolling diagnostics (1000-deep, like trainer_v10)
        self.recent_policy_entropy = collections.deque(maxlen=1000)
        self.recent_value_loss = collections.deque(maxlen=1000)
        self.recent_policy_loss = collections.deque(maxlen=1000)
        self.recent_advantages = collections.deque(maxlen=1000)
        self.recent_clip_fractions = collections.deque(maxlen=1000)
        self.recent_approx_kl = collections.deque(maxlen=1000)

    # ── Trajectory ingestion ────────────────────────────────────────────────
    def train_on_game(self, trajectory: List[dict], winner: int, model_player: int) -> Optional[dict]:
        """Process one completed game.

        trajectory: list of dicts with keys:
            'v15_x':       np.ndarray (8, 15, 15, 3) float32
            'v15_mask':    np.ndarray (225,) float32
            'action':      int (chosen cell index, 0..224)
            'old_log_prob': float
            'temperature': float (typically 1.0)
            'step_reward': float (sparse +SCORE_REWARD per score event during this step)

        Returns: metrics dict iff this game triggered a PPO update; else None.
        """
        if not trajectory:
            return None

        # Monte-Carlo discounted return (backwards roll).
        loss_penalty = -1.0  # 2-player → loser gets -1
        z = 1.0 if model_player == winner else (0.0 if winner < 0 else loss_penalty)
        won_target = 1.0 if model_player == winner else 0.0

        if self.terminal_only:
            rewards = [0.0] * len(trajectory)
        else:
            rewards = [step["step_reward"] for step in trajectory]
        rewards[-1] = rewards[-1] + z  # add terminal reward to last step

        # Backwards discount
        R = 0.0
        returns = [0.0] * len(trajectory)
        for i in range(len(trajectory) - 1, -1, -1):
            R = rewards[i] + GAMMA * R
            returns[i] = R

        # Buffer
        for i, (step, ret) in enumerate(zip(trajectory, returns)):
            self._ppo_buffer.append({
                "v15_x": step["v15_x"],
                "v15_mask": step["v15_mask"],
                "action": step["action"],
                "old_log_prob": step["old_log_prob"],
                "temperature": step.get("temperature", 1.0),
                "return": ret,
                "reward": rewards[i],   # per-step r_t for GAE
                "won_target": won_target,
                # v16 only: sparse tactical targets (capture-available,
                # in-danger) recorded at rollout time. Absent for v15 runs,
                # in which case every v16 hook below is a no-op.
                "aux_target": step.get("aux_target"),
            })
        self._ppo_games_buffered += 1
        self._traj_lengths.append(len(trajectory))
        self.total_games += 1

        if self._ppo_games_buffered >= self.ppo_buffer_games:
            return self._ppo_update()
        return None

    def _compute_gae(self, rewards, values, gamma=GAMMA):
        """Generalized Advantage Estimation, per trajectory (ported from
        v13.6 trainer_v10._compute_gae, Exp 58).

        rewards / values: (N,) tensors aligned to the flattened PPO buffer;
        self._traj_lengths splits them back into per-game trajectories.
            δ_t = r_t + γ·V(s_{t+1}) − V(s_t)   (V at the trajectory end = 0)
            A_t = δ_t + γλ·A_{t+1}              (backward recursion)
        At λ=1 this telescopes to (raw discounted return from t) − V_t, i.e.
        exactly the Monte-Carlo advantage (the correctness gate). The value
        head is the BCE-trained win-prob baseline; GAE only reads it. Computed
        on CPU/numpy (segments ~50-150 steps); result moved back to device.
        """
        r = rewards.detach().cpu().numpy()
        v = values.detach().cpu().numpy()
        adv = np.zeros_like(r)
        lam = self.gae_lambda
        idx = 0
        for L in self._traj_lengths:
            L = int(L)
            a = 0.0
            for t in range(L - 1, -1, -1):
                v_next = float(v[idx + t + 1]) if (t + 1) < L else 0.0
                delta = float(r[idx + t]) + gamma * v_next - float(v[idx + t])
                a = delta + gamma * lam * a
                adv[idx + t] = a
            idx += L
        return torch.from_numpy(adv).to(values.device, dtype=torch.float32)

    # ── PPO update ──────────────────────────────────────────────────────────
    def _ppo_update(self) -> dict:
        buf = self._ppo_buffer
        device = self.device

        all_states = np.stack([b["v15_x"] for b in buf], axis=0)         # (N,8,15,15,3)
        all_masks = np.stack([b["v15_mask"] for b in buf], axis=0)       # (N,225)
        all_actions = np.array([b["action"] for b in buf], dtype=np.int64)
        all_old_lp = np.array([b["old_log_prob"] for b in buf], dtype=np.float32)
        all_temps = np.array([b["temperature"] for b in buf], dtype=np.float32)
        all_returns_raw = np.array([b["return"] for b in buf], dtype=np.float32)
        all_rewards_raw = np.array([b.get("reward", 0.0) for b in buf], dtype=np.float32)
        all_won_targets = np.array([b["won_target"] for b in buf], dtype=np.float32)
        # v16 only — None for v15 runs, and every aux hook below no-ops.
        _aux = [b.get("aux_target") for b in buf]
        all_aux_t = (torch.from_numpy(np.asarray(_aux, dtype=np.float32)).to(device)
                     if _aux and _aux[0] is not None else None)

        # Update EMA running stats
        batch_mean = float(all_returns_raw.mean())
        batch_std = float(all_returns_raw.std() + 1e-8)
        a = self._return_alpha
        self._return_running_mean = (1 - a) * self._return_running_mean + a * batch_mean
        self._return_running_std = (1 - a) * self._return_running_std + a * batch_std
        all_returns = (all_returns_raw - self._return_running_mean) / (self._return_running_std + 1e-8)

        # Move to device tensors
        all_states_t = torch.from_numpy(all_states).to(device, dtype=torch.float32)
        all_masks_t = torch.from_numpy(all_masks).to(device, dtype=torch.float32)
        all_actions_t = torch.from_numpy(all_actions).to(device)
        all_old_lp_t = torch.from_numpy(all_old_lp).to(device)
        all_temps_t = torch.from_numpy(all_temps).to(device)
        all_returns_t = torch.from_numpy(all_returns).to(device)
        all_rewards_raw_t = torch.from_numpy(all_rewards_raw).to(device)
        all_won_t = torch.from_numpy(all_won_targets).to(device)

        # Pre-compute advantages once per update — CHUNKED to avoid OOM on
        # large PPO buffers (V15 GT attention has O(B·N²·H) activations).
        chunk = self.ppo_minibatch_size
        win_probs = []
        with torch.no_grad():
            for s0 in range(0, len(all_returns_raw), chunk):
                _, wp = self.model(
                    all_states_t[s0:s0 + chunk], all_masks_t[s0:s0 + chunk])
                win_probs.append(wp)
            win_prob0 = torch.cat(win_probs, dim=0)
            all_values = 2.0 * win_prob0 - 1.0
            if self.use_gae and sum(self._traj_lengths) == len(buf):
                # GAE on raw rewards + bootstrap values, per trajectory.
                # Scale-consistent (raw return − value at λ=1), unlike the MC
                # path's normalized-return − value. Falls back to MC if the
                # trajectory lengths don't sum to the buffer (safety).
                all_advantages = self._compute_gae(all_rewards_raw_t, all_values)
            else:
                all_advantages = all_returns_t - all_values
            # λ-return value target = value + (raw, un-normalized) advantage.
            # Computed BEFORE advantage normalization. Clamp to the reward range.
            all_value_targets = (all_values + all_advantages).clamp(-1.0, 1.0)
            all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

        N = len(buf)
        metrics_acc = {
            "policy_loss": 0.0, "win_bce_loss": 0.0,
            "entropy": 0.0, "advantage": 0.0,
            "clip_fraction": 0.0, "approx_kl": 0.0,
            "kl_anchor": 0.0,
        }
        n_minibatches = 0

        for epoch in range(self.ppo_epochs):
            order = np.random.permutation(N)
            for s in range(0, N, self.ppo_minibatch_size):
                idx = order[s:s + self.ppo_minibatch_size]
                if len(idx) < 2:
                    continue
                mb_idx = torch.from_numpy(idx).to(device)
                mb_states = all_states_t[mb_idx]
                mb_masks = all_masks_t[mb_idx]
                mb_actions = all_actions_t[mb_idx]
                mb_old_lp = all_old_lp_t[mb_idx]
                mb_temps = all_temps_t[mb_idx]
                mb_won = all_won_t[mb_idx]
                mb_adv = all_advantages[mb_idx]
                mb_vtarg = all_value_targets[mb_idx]
                mb_aux = all_aux_t[mb_idx] if all_aux_t is not None else None

                # v16: select which weight property receives this backward.
                # No-op for v15.
                self._mode_main()
                policy, win_prob = self.model(mb_states, mb_masks)
                # Re-derive behavior policy with the temperature used at rollout time.
                behavior_logits = torch.log(policy + 1e-8) / mb_temps.unsqueeze(1)
                # Re-mask (illegal cells were already -inf via mask in forward,
                # so policy already has 0 there; the temperature-scaling can
                # bring those rows close to log-zero rather than -inf so
                # we re-apply mask via softmax over the legal slice.)
                behavior_logits = behavior_logits.masked_fill(mb_masks < 0.5, -1e9)
                behavior_policy = F.softmax(behavior_logits, dim=1)
                new_lp = torch.log(behavior_policy.gather(
                    1, mb_actions.unsqueeze(1)).squeeze(1) + 1e-8)

                raw_ratio = torch.exp(new_lp - mb_old_lp)
                ratio = torch.clamp(raw_ratio, 0.0, 10.0)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio,
                                    1.0 - self.ppo_clip,
                                    1.0 + self.ppo_clip) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                if self.value_target == "lambda":
                    # MSE regress the value (2·win_prob−1 ∈ [−1,1]) to the
                    # bootstrapped λ-return — lower-variance than binary BCE.
                    win_bce_loss = F.mse_loss(2.0 * win_prob - 1.0, mb_vtarg)
                else:
                    win_bce_loss = F.binary_cross_entropy(
                        win_prob.clamp(1e-6, 1 - 1e-6), mb_won)

                # Entropy over masked policy
                log_p_all = torch.log(policy + 1e-8)
                entropy = -(policy * log_p_all).sum(dim=1).mean()

                loss = (
                    policy_loss
                    + self.win_bce_coeff * win_bce_loss
                    - self.entropy_coeff * entropy
                )

                kl_anchor_val = 0.0
                if self.kl_anchor_model is not None and self.kl_anchor_coeff > 0:
                    with torch.no_grad():
                        t_pol, _ = self.kl_anchor_model(mb_states, mb_masks)
                    kl = F.kl_div(log_p_all, t_pol, reduction="batchmean",
                                  log_target=False)
                    loss = loss + self.kl_anchor_coeff * kl
                    kl_anchor_val = float(kl.item())

                # NaN/Inf safety
                if not torch.isfinite(loss):
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                # v16: SECOND backward on the auxiliary objective, into the
                # second weight property. Runs BEFORE clipping/step so both
                # objectives share one clip and one optimizer step — the RL
                # update itself is untouched. Returns 0.0 for v15.
                aux_val = self._aux_backward(mb_states, mb_masks, mb_aux)
                nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

                with torch.no_grad():
                    clip_frac = ((raw_ratio - 1.0).abs() > self.ppo_clip).float().mean().item()
                    approx_kl = (mb_old_lp - new_lp).mean().item()

                metrics_acc["policy_loss"] += float(policy_loss.item())
                metrics_acc["win_bce_loss"] += float(win_bce_loss.item())
                metrics_acc["entropy"] += float(entropy.item())
                metrics_acc["advantage"] += float(mb_adv.mean().item())
                metrics_acc["clip_fraction"] += float(clip_frac)
                metrics_acc["approx_kl"] += float(approx_kl)
                metrics_acc["kl_anchor"] += kl_anchor_val
                metrics_acc["aux_loss"] = metrics_acc.get("aux_loss", 0.0) + aux_val
                n_minibatches += 1

        if n_minibatches > 0:
            for k in metrics_acc:
                metrics_acc[k] /= n_minibatches
        metrics_acc["n_steps"] = N
        metrics_acc["n_minibatches"] = n_minibatches

        # Rolling diagnostics
        self.recent_policy_loss.append(metrics_acc["policy_loss"])
        self.recent_value_loss.append(metrics_acc["win_bce_loss"])
        self.recent_policy_entropy.append(metrics_acc["entropy"])
        self.recent_advantages.append(metrics_acc["advantage"])
        self.recent_clip_fractions.append(metrics_acc["clip_fraction"])
        self.recent_approx_kl.append(metrics_acc["approx_kl"])

        # Reset buffer
        self._ppo_buffer = []
        self._ppo_games_buffered = 0
        self._traj_lengths = []
        self.total_updates += 1
        return metrics_acc

    # ── v16 hooks — no-ops here, overridden by V16RichTrainer ───────────────
    # These exist so v15 and v16 share ONE implementation of the PPO/GAE math.
    # The first two-signal attempt reimplemented the RL loop and diverged from
    # this pipeline in ~10 ways; a subclass with three hooks cannot drift.
    def _mode_main(self) -> None:
        """Select the property that receives the RL backward."""
        return None

    def _aux_backward(self, mb_states, mb_masks, mb_aux) -> float:
        """Second backward for the auxiliary objective. Returns its loss."""
        return 0.0

    # ── Convenience accessors for dashboard / logging ───────────────────────
    def get_diagnostic_means(self) -> dict:
        def m(d):
            return float(np.mean(list(d))) if d else 0.0
        return {
            "policy_entropy": m(self.recent_policy_entropy),
            "avg_value_loss": m(self.recent_value_loss),
            "avg_policy_loss": m(self.recent_policy_loss),
            "avg_advantage": m(self.recent_advantages),
            "clip_fraction": m(self.recent_clip_fractions),
            "approx_kl": m(self.recent_approx_kl),
        }
