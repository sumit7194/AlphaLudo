"""
AlphaLudo V10 Trainer — ActorCriticTrainer with 3-head model.

V10 model returns (policy, win_prob, moves_remaining). Adaptations:
  - Use win_prob as the value function: value = 2 * win_prob - 1 ∈ [-1, 1].
    Value loss is SmoothL1 against normalized returns (V6.3's exact pattern).
    After RL, win_prob drifts from calibrated P(win) to a standard value
    estimate — accepted trade-off (Exp 9 in journal showed γ=1 BCE
    training is too noisy for Ludo's terminal-reward variance).
  - moves_remaining head trained with auxiliary SmoothL1 loss against
    actual remaining own-turns (computed at game end from trajectory).
    Weight 0.003 matches the SL-phase balance; prevents head drift.

Everything else (PPO clipping, return normalization, gradient clipping,
entropy bonus) is unchanged from the base trainer.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

from td_ludo.training.trainer import ActorCriticTrainer


# Terminal-reward scaling. Default 1.0 (full ±1 win/loss). Set
# LUDO_TERMINAL_COEFF=0.0 in the env for the pure-shaping experiment
# (terminal signal removed entirely; only step rewards drive learning).
# Any value in [0, 1] is valid for partial scaling.
try:
    _TERMINAL_COEFF = float(os.environ.get('LUDO_TERMINAL_COEFF', '1.0'))
except ValueError:
    _TERMINAL_COEFF = 1.0
if _TERMINAL_COEFF != 1.0:
    print(f"[trainer_v10] LUDO_TERMINAL_COEFF={_TERMINAL_COEFF} "
          f"(terminal reward scaled; 0.0 = pure shaping)")


class ActorCriticTrainerV10(ActorCriticTrainer):
    """V10.2 trainer: PPO with BCE-trained win_prob head (calibration preserved).

    Changes vs original V10 trainer:
    - Drop SmoothL1 value loss entirely. The `2*win_prob-1` rescale was
      pulling win_prob to match normalized returns — which inverted the SL
      calibration because shaped returns are anticorrelated with P(win) in
      end-game states.
    - Add BCE loss on win_prob with binary outcome target (same as SL).
      This keeps win_prob as a true probability of winning throughout RL.
    - Use win_prob as value baseline with `.detach()` so gradient only flows
      through BCE — policy loss sees a stable baseline without interfering
      with calibration.
    - Reward shaping is sparse (score events only, +0.40 each) via
      players/v10.py's compute_sparse_reward. Keeps per-game return in a
      narrow range so the baseline remains useful.
    """

    def __init__(self, model, device, learning_rate=1e-5,
                 moves_aux_coeff=0.003, win_bce_coeff=0.5,
                 alpha_search=0.0, progress_coeff=0.0,
                 consequence_coeff=0.0,
                 use_gae=False, gae_lambda=0.95, **kwargs):
        super().__init__(model, device, learning_rate=learning_rate, **kwargs)
        self.moves_aux_coeff = moves_aux_coeff
        self.win_bce_coeff = win_bce_coeff
        # Exp 24: search-during-training auxiliary loss weight.
        # 0.0 disables; recommended start is 0.5 when search is enabled.
        self.alpha_search = alpha_search
        # Running stats for the auxiliary loss (averaged across PPO updates).
        self.recent_search_loss = []
        self.recent_search_kl = []
        self.recent_search_coverage = []

        # V13.5 progress aux head loss weight (the head predicts S(pos) per
        # canonical rank). Triggered only when the model exposes a
        # `has_progress_head` class attribute (V135Symmetric /
        # V135ProductionAdapter set this True). 0 disables.
        self.progress_coeff = float(progress_coeff)
        self.has_progress_head = bool(
            getattr(model, 'has_progress_head', False)
            or getattr(getattr(model, 'inner', None), 'has_progress_head', False)
        )
        self.recent_progress_loss = []

        # V13.6 consequence (world-model) aux head loss weight. The model
        # exposes capture + cells-at-risk heads (token-indexed). 0 disables.
        # Targets come from the player as 'capture_target'/'risk_target'/
        # 'consequence_valid' (per-token). See WORLD_MODEL_ATTACK_ANGLE.md.
        self.consequence_coeff = float(consequence_coeff)
        self.recent_consequence_loss = []

        # GAE (Exp 55): when True, the policy-gradient advantage is computed
        # via Generalized Advantage Estimation (value-bootstrapped, per
        # trajectory) instead of the high-variance Monte-Carlo
        # (return − value). The value head is STILL trained on BCE-of-outcome
        # (untouched) — GAE only uses its predictions as the bootstrap
        # baseline. λ=1 reduces exactly to the MC advantage (correctness gate).
        # Requires the buffer to carry per-step raw reward + per-trajectory
        # lengths (filled in train_on_game).
        self.use_gae = bool(use_gae)
        self.gae_lambda = float(gae_lambda)
        self._traj_lengths = []   # own-decision count per buffered game

    def train_on_game(self, trajectories, winner, model_player, aux_trajectory=None):
        """Buffer trajectory steps with own_moves_remaining + binary won target.

        aux_trajectory: optional list of {'state': tensor, 'cp': int} opp-turn
        states encoded canonically. Buffered for value-head training only
        (off-policy actions, no PPO grad). See trainer.py for the same flag on
        the base trainer.
        """
        if winner == -1:
            return {}

        from src.config import NUM_ACTIVE_PLAYERS

        # Buffer aux states with their value targets (current_player POV).
        if aux_trajectory:
            loss_penalty_aux = -1.0 / max(1, (NUM_ACTIVE_PLAYERS - 1))
            for step in aux_trajectory:
                cp_aux = step['cp']
                z_aux = 1.0 if cp_aux == winner else loss_penalty_aux
                won_target_aux = 1.0 if cp_aux == winner else 0.0
                self._aux_buffer.append({
                    'state': step['state'],
                    'z': z_aux,
                    'won_target': won_target_aux,
                })

        trajectory = trajectories.get(model_player, [])
        if not trajectory:
            return {}

        # Outcome from model's perspective. The terminal reward `z` is
        # scaled by LUDO_TERMINAL_COEFF (default 1.0). Set to 0.0 for a
        # pure-shaping experiment where ONLY step_reward drives learning.
        loss_penalty = -1.0 / max(1, (NUM_ACTIVE_PLAYERS - 1))
        z_raw = 1.0 if model_player == winner else loss_penalty
        z = z_raw * _TERMINAL_COEFF
        # `won_target` for BCE on the win-probability head stays unscaled
        # — the head still learns to PREDICT actual win/loss outcome
        # (useful as a calibrated estimator for evals + diagnostics) even
        # when the policy gradient ignores the terminal signal.
        won_target = 1.0 if model_player == winner else 0.0

        # Discounted returns (same as before) + per-step raw reward r_t
        # (the latter feeds GAE in _ppo_update; harmless when GAE is off).
        gamma = 0.999
        discounted_returns = []
        raw_rewards = []
        R = 0.0
        for i in reversed(range(len(trajectory))):
            step = trajectory[i]
            shaped_reward = step.get('step_reward', 0.0)
            if i == len(trajectory) - 1:
                r_t = shaped_reward + z
            else:
                r_t = shaped_reward
            R = r_t + gamma * R
            discounted_returns.insert(0, R)
            raw_rewards.insert(0, r_t)

        # Own moves remaining for each step
        total_own_moves = len(trajectory)

        # Buffer each step — ADDED: won_target (0/1), pi_search (optional),
        # progress_target / progress_valid (V13.5 progress aux head, optional).
        for i, step in enumerate(trajectory):
            own_moves_remaining = float(total_own_moves - (i + 1))
            self._ppo_buffer.append({
                'state': step['state'],
                'action': step['action'],
                'legal_mask': step['legal_mask'],
                'old_log_prob': step['old_log_prob'],
                'temperature': step.get('temperature', 1.0),
                'z': discounted_returns[i],
                'reward': raw_rewards[i],   # GAE per-step r_t
                'moves_remaining_target': own_moves_remaining,
                'won_target': won_target,
                'pi_search': step.get('pi_search'),  # np.ndarray (4,) or None
                # V13.5 progress aux: per-rank S(pos) target + valid mask.
                # player_v11.py fills these when progress_target_enabled=True;
                # otherwise zeros and they're effectively no-ops in loss.
                'progress_target': step.get('progress_target'),
                'progress_valid': step.get('progress_valid'),
                # V13.6 consequence targets (per-token capture / risk / valid).
                'capture_target': step.get('capture_target'),
                'risk_target': step.get('risk_target'),
                'consequence_valid': step.get('consequence_valid'),
            })
        # Record this game's own-decision count so _ppo_update can split the
        # flattened buffer back into trajectories for the GAE backward pass.
        self._traj_lengths.append(len(trajectory))
        self._ppo_games_buffered += 1

        if self._ppo_games_buffered >= self.ppo_buffer_games:
            return self._ppo_update()
        return {}

    def _compute_gae(self, rewards, values, gamma=0.999):
        """Generalized Advantage Estimation, per trajectory (Exp 55).

        rewards / values: (N,) tensors aligned to the flattened PPO buffer;
        self._traj_lengths splits them back into per-game trajectories.
            δ_t = r_t + γ·V(s_{t+1}) − V(s_t)   (V at the trajectory end = 0)
            A_t = δ_t + γλ·A_{t+1}              (backward recursion)
        At λ=1 this telescopes to A_t = (raw discounted return from t) − V_t,
        i.e. exactly the Monte-Carlo advantage (the correctness gate). The
        value head is the BCE-trained win-prob baseline; GAE only reads it.
        Computed on CPU/numpy (segments are short, ~50-150 steps) to avoid
        per-step GPU syncs; result moved back to device.
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

    def _ppo_update(self):
        """PPO update with win_prob as value head + moves aux loss."""
        if not self._ppo_buffer:
            return {}

        self.model.train()
        n_steps = len(self._ppo_buffer)

        # Stack buffered data (identical structure to base, plus moves_target)
        all_states = torch.from_numpy(
            np.stack([s['state'] for s in self._ppo_buffer])
        ).to(self.device, dtype=torch.float32)

        all_actions = torch.tensor(
            [s['action'] for s in self._ppo_buffer],
            dtype=torch.long, device=self.device
        )

        all_masks = torch.from_numpy(
            np.stack([s['legal_mask'] for s in self._ppo_buffer])
        ).to(self.device, dtype=torch.float32)

        all_old_lp = torch.tensor(
            [s['old_log_prob'] for s in self._ppo_buffer],
            dtype=torch.float32, device=self.device
        )

        all_temperatures = torch.tensor(
            [s.get('temperature', 1.0) for s in self._ppo_buffer],
            dtype=torch.float32, device=self.device
        )

        all_returns_raw = torch.tensor(
            [s['z'] for s in self._ppo_buffer],
            dtype=torch.float32, device=self.device
        )

        # GAE: per-step raw reward r_t (terminal z folded into the last step
        # of each trajectory). Only consumed when self.use_gae.
        all_rewards_raw = torch.tensor(
            [s.get('reward', 0.0) for s in self._ppo_buffer],
            dtype=torch.float32, device=self.device
        )

        # V10: own_moves_remaining targets for auxiliary loss
        all_moves_targets = torch.tensor(
            [s['moves_remaining_target'] for s in self._ppo_buffer],
            dtype=torch.float32, device=self.device
        )

        # V10.2: binary won targets for BCE loss on win_prob (calibration)
        all_won_targets = torch.tensor(
            [s['won_target'] for s in self._ppo_buffer],
            dtype=torch.float32, device=self.device
        )

        # Exp 24: stack pi_search targets and a mask of which steps have one.
        # Steps without a search target store None; those rows are zero-masked
        # so they contribute nothing to the auxiliary loss.
        if self.alpha_search > 0.0:
            pi_search_arrs = []
            search_mask_list = []
            for s in self._ppo_buffer:
                ps = s.get('pi_search')
                if ps is None:
                    pi_search_arrs.append(np.zeros(4, dtype=np.float32))
                    search_mask_list.append(0.0)
                else:
                    pi_search_arrs.append(ps.astype(np.float32))
                    search_mask_list.append(1.0)
            all_pi_search = torch.from_numpy(np.stack(pi_search_arrs)).to(
                self.device, dtype=torch.float32,
            )
            all_search_mask = torch.tensor(
                search_mask_list, dtype=torch.float32, device=self.device,
            )
        else:
            all_pi_search = None
            all_search_mask = None

        # V13.5 progress aux head: per-rank S(pos) targets + valid mask.
        # Only stacked when both the model has the head AND progress_coeff > 0.
        if self.has_progress_head and self.progress_coeff > 0.0:
            prog_target_arrs = []
            prog_valid_arrs = []
            for s in self._ppo_buffer:
                pt = s.get('progress_target')
                pv = s.get('progress_valid')
                if pt is None:
                    prog_target_arrs.append(np.zeros(4, dtype=np.float32))
                    prog_valid_arrs.append(np.zeros(4, dtype=np.float32))
                else:
                    prog_target_arrs.append(pt.astype(np.float32))
                    prog_valid_arrs.append(pv.astype(np.float32))
            all_progress_targets = torch.from_numpy(np.stack(prog_target_arrs)).to(
                self.device, dtype=torch.float32,
            )
            all_progress_valid = torch.from_numpy(np.stack(prog_valid_arrs)).to(
                self.device, dtype=torch.float32,
            )
        else:
            all_progress_targets = None
            all_progress_valid = None

        # V13.6 consequence aux: per-token capture + cells-at-risk + valid.
        if self.consequence_coeff > 0.0:
            cap_arrs, risk_arrs, cval_arrs = [], [], []
            for s in self._ppo_buffer:
                ct = s.get('capture_target')
                rt = s.get('risk_target')
                cv = s.get('consequence_valid')
                if ct is None:
                    cap_arrs.append(np.zeros(4, dtype=np.float32))
                    risk_arrs.append(np.zeros(4, dtype=np.float32))
                    cval_arrs.append(np.zeros(4, dtype=np.float32))
                else:
                    cap_arrs.append(ct.astype(np.float32))
                    risk_arrs.append(rt.astype(np.float32))
                    cval_arrs.append(cv.astype(np.float32))
            all_capture_targets = torch.from_numpy(np.stack(cap_arrs)).to(
                self.device, dtype=torch.float32)
            all_risk_targets = torch.from_numpy(np.stack(risk_arrs)).to(
                self.device, dtype=torch.float32)
            all_consequence_valid = torch.from_numpy(np.stack(cval_arrs)).to(
                self.device, dtype=torch.float32)
        else:
            all_capture_targets = None
            all_risk_targets = None
            all_consequence_valid = None

        # Return normalization (identical to base)
        with torch.no_grad():
            batch_mean = all_returns_raw.mean().item()
            batch_std = all_returns_raw.std().item()
            if not self._return_stats_initialized:
                self._return_running_mean = batch_mean
                self._return_running_std = max(batch_std, 1e-6)
                self._return_stats_initialized = True
            else:
                self._return_running_mean = 0.99 * self._return_running_mean + 0.01 * batch_mean
                self._return_running_std = 0.99 * self._return_running_std + 0.01 * max(batch_std, 1e-6)
            all_returns = (all_returns_raw - self._return_running_mean) / (self._return_running_std + 1e-8)

        # Stats accumulators. Kept as GPU tensors (2026-06-11 throughput
        # fix): the previous per-minibatch `.item()` accumulation forced
        # 6-8 GPU→CPU syncs per minibatch. They're now accumulated on
        # device and synced ONCE at the end of the update. Search-loss
        # accumulators stay as floats — that path only runs when search
        # is enabled (it isn't in production runs).
        _dev = self.device
        total_policy_loss = torch.zeros((), device=_dev)
        total_value_loss = torch.zeros((), device=_dev)
        total_moves_loss = torch.zeros((), device=_dev)
        total_entropy = torch.zeros((), device=_dev)
        total_advantage = torch.zeros((), device=_dev)
        total_clip_frac = torch.zeros((), device=_dev)
        total_approx_kl = torch.zeros((), device=_dev)
        total_progress_loss = torch.zeros((), device=_dev)
        n_progress_mb = 0
        total_consequence_loss = torch.zeros((), device=_dev)
        n_consequence_mb = 0
        total_search_loss = 0.0
        total_search_kl = 0.0  # KL(pi_search || pi_model) on covered states
        total_search_coverage = 0.0
        n_minibatches = 0

        # Precompute advantages using V10's win_prob → value transformation
        with torch.no_grad():
            # V10 returns (policy, win_prob, moves_remaining); V13.5 returns
            # (policy, win_prob, moves_remaining, progress). Index by [1] is
            # win_prob in either case.
            fwd_result = self.model(all_states, all_masks)
            win_prob = fwd_result[1].view(-1)  # robust to batch_size=1 (.squeeze bug)
            all_values = 2.0 * win_prob - 1.0  # [0,1] → [-1,1]
            if self.use_gae and sum(self._traj_lengths) == n_steps:
                # GAE on the raw rewards + bootstrap values, per trajectory.
                # Scale-consistent (raw return − value at λ=1), unlike the MC
                # path's normalized-return − value. Falls back to MC if the
                # trajectory lengths don't sum to the buffer (safety).
                all_advantages = self._compute_gae(all_rewards_raw, all_values)
            else:
                # Monte-Carlo advantage (legacy path; unchanged).
                all_advantages = all_returns - all_values
            adv_mean = all_advantages.mean()
            adv_std = all_advantages.std()
            all_advantages = (all_advantages - adv_mean) / (adv_std + 1e-8)

        for epoch in range(self.ppo_epochs):
            indices = np.random.permutation(n_steps)

            for start in range(0, n_steps, self.ppo_minibatch_size):
                end = min(start + self.ppo_minibatch_size, n_steps)
                mb_idx = indices[start:end]

                mb_states = all_states[mb_idx]
                mb_actions = all_actions[mb_idx]
                mb_masks = all_masks[mb_idx]
                mb_old_lp = all_old_lp[mb_idx]
                mb_temperatures = all_temperatures[mb_idx]
                mb_returns = all_returns[mb_idx]
                mb_advantages = all_advantages[mb_idx]
                mb_moves_targets = all_moves_targets[mb_idx]
                mb_won_targets = all_won_targets[mb_idx]
                if all_pi_search is not None:
                    mb_pi_search = all_pi_search[mb_idx]
                    mb_search_mask = all_search_mask[mb_idx]
                else:
                    mb_pi_search = None
                    mb_search_mask = None

                if all_progress_targets is not None:
                    mb_progress_targets = all_progress_targets[mb_idx]
                    mb_progress_valid = all_progress_valid[mb_idx]
                else:
                    mb_progress_targets = None
                    mb_progress_valid = None

                if all_capture_targets is not None:
                    mb_capture_targets = all_capture_targets[mb_idx]
                    mb_risk_targets = all_risk_targets[mb_idx]
                    mb_consequence_valid = all_consequence_valid[mb_idx]
                else:
                    mb_capture_targets = None
                    mb_risk_targets = None
                    mb_consequence_valid = None

                # V10/V13.5/V13.6: forward returns (policy, win_prob, moves)
                # for V12.x; (..., progress) for V13.5; (..., progress,
                # capture, risk) for V13.6. Unpack robustly.
                fwd = self.model(mb_states, mb_masks)
                policy, win_prob, moves_pred = fwd[0], fwd[1], fwd[2]
                progress_pred = fwd[3] if len(fwd) > 3 else None
                capture_pred = fwd[4] if len(fwd) > 4 else None
                risk_pred = fwd[5] if len(fwd) > 5 else None
                # Model already squeezes last dim. Use .view(-1) to guarantee
                # 1D shape even when batch size is 1 (.squeeze(-1) on [1]
                # collapses to 0-dim scalar, which breaks F.binary_cross_entropy
                # when target is still shape [1]).
                win_prob = win_prob.view(-1)
                moves_pred = moves_pred.view(-1)
                # V10.2: NO `value = 2*win_prob - 1` here — win_prob is for
                # BCE calibration only. Advantage was already precomputed from
                # detached win_prob outside the minibatch loop.

                advantage = mb_advantages

                # Behavior policy reconstruction (identical to base/V6.3)
                behavior_temps = mb_temperatures.clamp_min(1e-6).unsqueeze(1)
                behavior_logits = torch.log(policy + 1e-8) / behavior_temps
                behavior_policy = F.softmax(behavior_logits, dim=1)
                new_lp = torch.log(
                    behavior_policy.gather(1, mb_actions.unsqueeze(1)).squeeze(1) + 1e-8
                )

                # PPO ratio + clipping (identical to base)
                raw_ratio = torch.exp(new_lp - mb_old_lp)
                ratio = torch.clamp(raw_ratio, 0.0, 10.0)

                surr1 = ratio * advantage
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                ) * advantage
                policy_loss = -torch.min(surr1, surr2).mean()

                # V10.2: BCE loss on win_prob (replaces V10's broken SmoothL1
                # value loss). Same objective as SL: predict P(player wins).
                # Prevents shaped-return signal from inverting the head.
                win_bce_loss = F.binary_cross_entropy(
                    win_prob.clamp(1e-6, 1 - 1e-6), mb_won_targets
                )

                entropy = -(policy * torch.log(policy + 1e-8)).sum(dim=1).mean()

                # V10: auxiliary moves-remaining loss with SmoothL1.
                moves_loss = F.smooth_l1_loss(moves_pred, mb_moves_targets)

                # Exp 24: search-during-training auxiliary loss.
                # Cross-entropy from pi_search (smoothed one-hot at search-
                # argmax) to the model's policy. Computed only on covered
                # rows; averaged over those rows so alpha_search has the
                # nominal "per-covered-state" meaning regardless of fraction.
                if mb_pi_search is not None and mb_search_mask.sum() > 0:
                    log_policy = torch.log(policy + 1e-8)
                    per_row_ce = -(mb_pi_search * log_policy).sum(dim=1)
                    n_covered = mb_search_mask.sum().clamp_min(1.0)
                    search_loss = (per_row_ce * mb_search_mask).sum() / n_covered
                    coverage_frac = (n_covered / float(end - start)).item()

                    # KL diagnostic (search || model) on covered rows only.
                    with torch.no_grad():
                        kl_per = (
                            mb_pi_search * (
                                torch.log(mb_pi_search + 1e-8) - log_policy
                            )
                        ).sum(dim=1)
                        kl_avg = (kl_per * mb_search_mask).sum() / n_covered
                else:
                    search_loss = torch.zeros((), device=self.device)
                    coverage_frac = 0.0
                    kl_avg = torch.zeros((), device=self.device)

                # V13.5 progress aux loss: per-rank S(pos) prediction.
                # Uses BCE since both predicted and target are in [0, 1] with
                # sigmoid output. Masked by progress_valid (1 where the rank
                # corresponds to a real own-token position, 0 for unused
                # ranks beyond the number of unique positions).
                # NOTE (2026-06-11): the old `mb_progress_valid.sum() > 0`
                # guard forced a GPU→CPU sync per minibatch. With an
                # all-zero valid mask the masked sum is 0 and clamp_min
                # keeps the denominator at 1 → loss is exactly 0 anyway,
                # so the guard was redundant. Output-identical.
                if progress_pred is not None and mb_progress_targets is not None:
                    pred = progress_pred.clamp(1e-6, 1 - 1e-6)
                    tgt = mb_progress_targets.clamp(0.0, 1.0)
                    # Element-wise BCE then average over valid slots only
                    bce_per = -(
                        tgt * torch.log(pred) + (1.0 - tgt) * torch.log(1.0 - pred)
                    )
                    n_valid = mb_progress_valid.sum().clamp_min(1.0)
                    progress_loss = (bce_per * mb_progress_valid).sum() / n_valid
                else:
                    progress_loss = torch.zeros((), device=self.device)

                # V13.6 consequence aux loss: per-token capture prob (BCE) +
                # cells-at-risk (BCE, target ∈ [0,1]), masked by
                # consequence_valid (own tokens on the main track). Both heads
                # sigmoid; we use BCE for both since targets are in [0, 1].
                if (capture_pred is not None and risk_pred is not None
                        and mb_capture_targets is not None):
                    cap_p = capture_pred.clamp(1e-6, 1 - 1e-6)
                    risk_p = risk_pred.clamp(1e-6, 1 - 1e-6)
                    cap_t = mb_capture_targets.clamp(0.0, 1.0)
                    risk_t = mb_risk_targets.clamp(0.0, 1.0)
                    cap_bce = -(cap_t * torch.log(cap_p)
                                + (1.0 - cap_t) * torch.log(1.0 - cap_p))
                    risk_bce = -(risk_t * torch.log(risk_p)
                                 + (1.0 - risk_t) * torch.log(1.0 - risk_p))
                    n_cvalid = mb_consequence_valid.sum().clamp_min(1.0)
                    consequence_loss = (
                        ((cap_bce + risk_bce) * mb_consequence_valid).sum() / n_cvalid
                    )
                else:
                    consequence_loss = torch.zeros((), device=self.device)

                # AUX value loss (BCE on win_prob from opp-turn states).
                # Off-policy actions, so value-only — no policy grad path.
                # Encoded canonically post encoder-fix → value head learns
                # P(current_player wins) from broader state distribution.
                aux_value_loss = torch.tensor(0.0, device=self.device)
                if self._aux_buffer:
                    n_aux = len(self._aux_buffer)
                    aux_mb = min(n_aux, mb_states.shape[0])
                    # Sample on CPU via numpy (2026-06-11): the old
                    # device-side torch.randint + per-element `i.item()`
                    # cost one GPU→CPU sync per aux row per minibatch.
                    # Statistically identical sampling.
                    aux_idx = np.random.randint(0, n_aux, size=aux_mb)
                    aux_states = torch.from_numpy(
                        np.stack([self._aux_buffer[int(i)]['state'] for i in aux_idx])
                    ).to(self.device, dtype=torch.float32)
                    aux_won_targets = torch.tensor(
                        [self._aux_buffer[int(i)]['won_target'] for i in aux_idx],
                        dtype=torch.float32, device=self.device
                    )
                    # Permissive mask — value head is mask-independent.
                    aux_masks = torch.ones((aux_mb, 4), dtype=torch.float32, device=self.device)
                    aux_fwd = self.model(aux_states, aux_masks)
                    aux_win_prob = aux_fwd[1]
                    # Use view(-1) instead of squeeze(-1) to handle batch=1 case.
                    # squeeze(-1) on shape (1,) yields scalar (), causing BCE
                    # to crash with 'target [1] vs input []' mismatch.
                    aux_win_prob = aux_win_prob.view(-1)
                    aux_value_loss = F.binary_cross_entropy(
                        aux_win_prob.clamp(1e-6, 1 - 1e-6), aux_won_targets
                    )

                loss = (policy_loss
                        + self.win_bce_coeff * win_bce_loss
                        + self.moves_aux_coeff * moves_loss
                        + self.alpha_search * search_loss
                        + self.aux_value_loss_coeff * aux_value_loss
                        + self.progress_coeff * progress_loss
                        + self.consequence_coeff * consequence_loss
                        - self.entropy_coeff * entropy)

                # Safety net: skip NaN/Inf batches (belt-and-braces for MPS).
                # Single isfinite check = one sync instead of two (2026-06-11).
                if not torch.isfinite(loss):
                    self.optimizer.zero_grad()
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.total_updates += 1

                # On-device accumulation (2026-06-11) — no per-minibatch
                # .item() syncs; one .cpu() after the epoch loop.
                total_policy_loss += policy_loss.detach()
                total_value_loss += win_bce_loss.detach()  # tracks BCE loss
                total_moves_loss += moves_loss.detach()
                total_entropy += entropy.detach()
                total_advantage += advantage.mean().detach()
                if all_pi_search is not None:
                    total_search_loss += float(search_loss.item())
                    total_search_kl += float(kl_avg.item())
                    total_search_coverage += coverage_frac
                if progress_pred is not None and self.progress_coeff > 0.0:
                    # Was: per-minibatch float append (one sync each).
                    # Now: accumulate on device, append the per-update mean
                    # once after the loop. Dashboard granularity only.
                    total_progress_loss += progress_loss.detach()
                    n_progress_mb += 1

                if capture_pred is not None and self.consequence_coeff > 0.0:
                    total_consequence_loss += consequence_loss.detach()
                    n_consequence_mb += 1

                with torch.no_grad():
                    clipped = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()
                    approx_kl_t = (mb_old_lp - new_lp).mean()
                total_clip_frac += clipped
                total_approx_kl += approx_kl_t.abs()
                n_minibatches += 1

        # Average stats — single GPU→CPU sync for the whole update
        # (2026-06-11; was one .item() per metric per minibatch).
        if n_minibatches > 0:
            _stats = (torch.stack([
                total_policy_loss, total_value_loss, total_moves_loss,
                total_entropy, total_advantage, total_clip_frac,
                total_approx_kl,
            ]) / n_minibatches).cpu()
            (avg_pl, avg_vl, avg_ml, avg_ent,
             avg_adv, avg_clip, avg_kl) = (float(x) for x in _stats)
        else:
            avg_pl = avg_vl = avg_ml = avg_ent = 0
            avg_adv = avg_clip = avg_kl = 0
        if n_progress_mb > 0:
            self.recent_progress_loss.append(
                float(total_progress_loss.cpu()) / n_progress_mb
            )
            if len(self.recent_progress_loss) > 1000:
                self.recent_progress_loss.pop(0)
        if n_consequence_mb > 0:
            self.recent_consequence_loss.append(
                float(total_consequence_loss.cpu()) / n_consequence_mb
            )
            if len(self.recent_consequence_loss) > 1000:
                self.recent_consequence_loss.pop(0)

        self.recent_policy_loss.append(avg_pl)
        self.recent_value_loss.append(avg_vl)
        self.recent_policy_entropy.append(avg_ent)
        self.recent_advantages.append(avg_adv)
        self.recent_clip_fractions.append(avg_clip)
        self.recent_approx_kl.append(avg_kl)

        if all_pi_search is not None and n_minibatches > 0:
            avg_search_loss = total_search_loss / n_minibatches
            avg_search_kl = total_search_kl / n_minibatches
            avg_search_cov = total_search_coverage / n_minibatches
            self.recent_search_loss.append(avg_search_loss)
            self.recent_search_kl.append(avg_search_kl)
            self.recent_search_coverage.append(avg_search_cov)
        else:
            avg_search_loss = 0.0
            avg_search_kl = 0.0
            avg_search_cov = 0.0

        n_aux_used = len(self._aux_buffer)
        self._ppo_buffer = []
        self._aux_buffer.clear()
        self._traj_lengths = []
        self._ppo_games_buffered = 0

        return {
            'policy_loss': avg_pl,
            'win_bce_loss': avg_vl,  # V10.2: replaces old value_loss (SmoothL1)
            'value_loss': avg_vl,    # kept as alias for dashboard backward-compat
            'moves_loss': avg_ml,
            'entropy': avg_ent,
            'advantage': avg_adv,
            'clip_fraction': avg_clip,
            'approx_kl': avg_kl,
            'search_loss': avg_search_loss,
            'search_kl': avg_search_kl,
            'search_coverage': avg_search_cov,
            'n_steps': n_steps,
            'n_minibatches': n_minibatches,
            'aux_states_used': n_aux_used,
        }
