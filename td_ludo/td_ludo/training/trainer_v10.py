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

import numpy as np
import torch
import torch.nn.functional as F

from td_ludo.training.trainer import ActorCriticTrainer


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
                 alpha_search=0.0, **kwargs):
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

    def train_on_game(self, trajectories, winner, model_player):
        """Buffer trajectory steps with own_moves_remaining + binary won target."""
        if winner == -1:
            return {}

        from src.config import NUM_ACTIVE_PLAYERS
        trajectory = trajectories.get(model_player, [])
        if not trajectory:
            return {}

        # Outcome from model's perspective
        loss_penalty = -1.0 / max(1, (NUM_ACTIVE_PLAYERS - 1))
        z = 1.0 if model_player == winner else loss_penalty
        won_target = 1.0 if model_player == winner else 0.0  # for BCE

        # Discounted returns (same as before)
        gamma = 0.999
        discounted_returns = []
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

        # Own moves remaining for each step
        total_own_moves = len(trajectory)

        # Buffer each step — ADDED: won_target (0/1), pi_search (optional)
        for i, step in enumerate(trajectory):
            own_moves_remaining = float(total_own_moves - (i + 1))
            self._ppo_buffer.append({
                'state': step['state'],
                'action': step['action'],
                'legal_mask': step['legal_mask'],
                'old_log_prob': step['old_log_prob'],
                'temperature': step.get('temperature', 1.0),
                'z': discounted_returns[i],
                'moves_remaining_target': own_moves_remaining,
                'won_target': won_target,
                'pi_search': step.get('pi_search'),  # np.ndarray (4,) or None
            })
        self._ppo_games_buffered += 1

        if self._ppo_games_buffered >= self.ppo_buffer_games:
            return self._ppo_update()
        return {}

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

        # Stats accumulators
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_moves_loss = 0.0
        total_entropy = 0.0
        total_advantage = 0.0
        total_clip_frac = 0.0
        total_approx_kl = 0.0
        total_search_loss = 0.0
        total_search_kl = 0.0  # KL(pi_search || pi_model) on covered states
        total_search_coverage = 0.0
        n_minibatches = 0

        # Precompute advantages using V10's win_prob → value transformation
        with torch.no_grad():
            # V10 model returns (policy, win_prob, moves_remaining)
            fwd_result = self.model(all_states, all_masks)
            win_prob = fwd_result[1].view(-1)  # robust to batch_size=1 (.squeeze bug)
            all_values = 2.0 * win_prob - 1.0  # [0,1] → [-1,1]
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

                # V10: forward returns (policy, win_prob, moves_remaining)
                policy, win_prob, moves_pred = self.model(mb_states, mb_masks)
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

                loss = (policy_loss
                        + self.win_bce_coeff * win_bce_loss
                        + self.moves_aux_coeff * moves_loss
                        + self.alpha_search * search_loss
                        - self.entropy_coeff * entropy)

                # Safety net: skip NaN/Inf batches (belt-and-braces for MPS)
                if torch.isnan(loss) or torch.isinf(loss):
                    self.optimizer.zero_grad()
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.total_updates += 1

                total_policy_loss += policy_loss.item()
                total_value_loss += win_bce_loss.item()  # now tracks BCE loss
                total_moves_loss += moves_loss.item()
                total_entropy += entropy.item()
                total_advantage += advantage.mean().item()
                if all_pi_search is not None:
                    total_search_loss += float(search_loss.item())
                    total_search_kl += float(kl_avg.item())
                    total_search_coverage += coverage_frac

                with torch.no_grad():
                    clipped = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                    approx_kl = (mb_old_lp - new_lp).mean().item()
                total_clip_frac += clipped
                total_approx_kl += abs(approx_kl)
                n_minibatches += 1

        # Average stats
        avg_pl = total_policy_loss / n_minibatches if n_minibatches > 0 else 0
        avg_vl = total_value_loss / n_minibatches if n_minibatches > 0 else 0
        avg_ml = total_moves_loss / n_minibatches if n_minibatches > 0 else 0
        avg_ent = total_entropy / n_minibatches if n_minibatches > 0 else 0
        avg_adv = total_advantage / n_minibatches if n_minibatches > 0 else 0
        avg_clip = total_clip_frac / n_minibatches if n_minibatches > 0 else 0
        avg_kl = total_approx_kl / n_minibatches if n_minibatches > 0 else 0

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

        self._ppo_buffer = []
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
        }
