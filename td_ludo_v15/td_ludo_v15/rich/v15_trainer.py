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

        # PPO buffering
        self._ppo_buffer: List[dict] = []
        self._ppo_games_buffered = 0

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

        rewards = [step["step_reward"] for step in trajectory]
        rewards[-1] = rewards[-1] + z  # add terminal reward to last step

        # Backwards discount
        R = 0.0
        returns = [0.0] * len(trajectory)
        for i in range(len(trajectory) - 1, -1, -1):
            R = rewards[i] + GAMMA * R
            returns[i] = R

        # Buffer
        for step, ret in zip(trajectory, returns):
            self._ppo_buffer.append({
                "v15_x": step["v15_x"],
                "v15_mask": step["v15_mask"],
                "action": step["action"],
                "old_log_prob": step["old_log_prob"],
                "temperature": step.get("temperature", 1.0),
                "return": ret,
                "won_target": won_target,
            })
        self._ppo_games_buffered += 1
        self.total_games += 1

        if self._ppo_games_buffered >= self.ppo_buffer_games:
            return self._ppo_update()
        return None

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
        all_won_targets = np.array([b["won_target"] for b in buf], dtype=np.float32)

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
            all_advantages = all_returns_t - all_values
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
        self.total_updates += 1
        return metrics_acc

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
