"""
AlphaLudo V6.3 Trainer — ActorCriticTrainer + auxiliary capture prediction loss.

Extends the base trainer with:
- Buffers `aux_capture_target` from V6.3 game player trajectories
- Adds BCE aux loss (0.1× weight) to the PPO combined loss
- Logs aux_loss to training stats

Everything else (PPO clipping, return normalization, gradient clipping,
value loss, entropy bonus) is unchanged from the base trainer.
"""

import numpy as np
import torch
import torch.nn.functional as F

from td_ludo.training.trainer import ActorCriticTrainer


class ActorCriticTrainerV63(ActorCriticTrainer):
    """V6.3 trainer: base PPO + auxiliary capture prediction."""

    def __init__(self, model, device, learning_rate=1e-5,
                 aux_loss_coeff=0.0, **kwargs):
        super().__init__(model, device, learning_rate=learning_rate, **kwargs)
        self.aux_loss_coeff = aux_loss_coeff  # 0.0 = aux disabled

    def train_on_game(self, trajectories, winner, model_player):
        """Override: also buffer aux_capture_target from trajectory steps."""
        if winner == -1:
            return {}

        from src.config import NUM_ACTIVE_PLAYERS
        trajectory = trajectories.get(model_player, [])
        if not trajectory:
            return {}

        # Outcome from model's perspective
        loss_penalty = -1.0 / max(1, (NUM_ACTIVE_PLAYERS - 1))
        z = 1.0 if model_player == winner else loss_penalty

        # Discounted returns (identical to base trainer)
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

        # Buffer each step — ADDED: aux_capture_target
        for i, step in enumerate(trajectory):
            self._ppo_buffer.append({
                'state': step['state'],
                'action': step['action'],
                'legal_mask': step['legal_mask'],
                'old_log_prob': step['old_log_prob'],
                'temperature': step.get('temperature', 1.0),
                'z': discounted_returns[i],
                'aux_capture_target': step.get('aux_capture_target', 0.0),
            })
        self._ppo_games_buffered += 1

        if self._ppo_games_buffered >= self.ppo_buffer_games:
            return self._ppo_update()
        return {}

    def _ppo_update(self):
        """Override: unpack 3-value model output and add aux capture loss."""
        if not self._ppo_buffer:
            return {}

        self.model.train()
        n_steps = len(self._ppo_buffer)

        # Stack all buffered data (identical to base)
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

        # V6.3: auxiliary capture targets
        all_aux_targets = torch.tensor(
            [s.get('aux_capture_target', 0.0) for s in self._ppo_buffer],
            dtype=torch.float32, device=self.device
        )

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
        total_aux_loss = 0.0
        total_entropy = 0.0
        total_advantage = 0.0
        total_clip_frac = 0.0
        total_approx_kl = 0.0
        n_minibatches = 0

        # Precompute advantages (identical to base)
        with torch.no_grad():
            # V6.3 model returns 3 values — ignore aux here
            fwd_result = self.model(all_states, all_masks)
            all_values = fwd_result[1].squeeze(-1)
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
                mb_aux_targets = all_aux_targets[mb_idx]

                # V6.3: forward returns (policy, value, aux) — aux ignored
                policy, value, _aux = self.model(mb_states, mb_masks)
                value = value.squeeze(-1)

                advantage = mb_advantages

                # Behavior policy reconstruction (identical to base)
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

                value_loss = F.smooth_l1_loss(value, mb_returns)

                entropy = -(policy * torch.log(policy + 1e-8)).sum(dim=1).mean()

                # Aux disabled — pure PPO with new channels only
                aux_loss = torch.tensor(0.0)

                # Combined loss (policy + value - entropy)
                loss = (policy_loss
                        + self.value_loss_coeff * value_loss
                        - self.entropy_coeff * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.total_updates += 1

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_aux_loss += aux_loss.item()
                total_entropy += entropy.item()
                total_advantage += advantage.mean().item()

                with torch.no_grad():
                    clipped = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                    approx_kl = (mb_old_lp - new_lp).mean().item()
                total_clip_frac += clipped
                total_approx_kl += abs(approx_kl)
                n_minibatches += 1

        # Record average stats
        avg_pl = total_policy_loss / n_minibatches if n_minibatches > 0 else 0
        avg_vl = total_value_loss / n_minibatches if n_minibatches > 0 else 0
        avg_al = total_aux_loss / n_minibatches if n_minibatches > 0 else 0
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

        # Clear buffer
        self._ppo_buffer = []
        self._ppo_games_buffered = 0

        return {
            'policy_loss': avg_pl,
            'value_loss': avg_vl,
            'aux_loss': avg_al,
            'entropy': avg_ent,
            'advantage': avg_adv,
            'clip_fraction': avg_clip,
            'approx_kl': avg_kl,
            'n_steps': n_steps,
            'n_minibatches': n_minibatches,
        }
