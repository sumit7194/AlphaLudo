"""
TD-Ludo V8 Trainer — PPO for V6 CNN + Temporal Transformer

Same PPO algorithm as V7 trainer, but handles V8's grid-based inputs:
- Trajectories contain (grids, prev_actions, seq_mask) per step
  instead of V7's (token_positions, continuous, actions_seq, seq_mask).
- model.forward() takes (grids, prev_actions, seq_mask, legal_mask).

All other logic (PPO clipping, advantage normalization, gamma-discounted returns,
ghost saving, checkpoint management) is identical to V7.
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import shutil
from collections import deque

from src.config import (
    LEARNING_RATE, WEIGHT_DECAY, MAX_GRAD_NORM,
    CHECKPOINT_DIR,
    MAIN_CKPT_PATH, BEST_CKPT_PATH, METRICS_PATH,
    GHOSTS_DIR, MAX_GHOSTS, GHOST_SAVE_INTERVAL,
    STATS_PATH,
    ENTROPY_COEFF, VALUE_LOSS_COEFF,
    CLIP_EPSILON, PPO_EPOCHS, PPO_BUFFER_GAMES, PPO_MINIBATCH_SIZE,
    NUM_ACTIVE_PLAYERS,
)


class V8Trainer:
    """
    PPO trainer for AlphaLudoV8 (V6 CNN + Temporal Transformer).

    Same interface as V7Trainer but handles grid-based sequence inputs.
    """

    def __init__(self, model, device, learning_rate=LEARNING_RATE):
        self.model = model
        self.device = device
        self.model.to(device)

        # Only optimize trainable parameters (CNN may be frozen)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(
            trainable_params,
            lr=learning_rate,
            weight_decay=WEIGHT_DECAY,
        )

        self.max_grad_norm = MAX_GRAD_NORM
        self.entropy_coeff = ENTROPY_COEFF
        self.value_loss_coeff = VALUE_LOSS_COEFF

        # PPO parameters
        self.clip_epsilon = CLIP_EPSILON
        self.ppo_epochs = PPO_EPOCHS
        self.ppo_buffer_games = PPO_BUFFER_GAMES
        self.ppo_minibatch_size = PPO_MINIBATCH_SIZE

        # PPO buffer
        self._ppo_buffer = []
        self._ppo_games_buffered = 0

        # Tracking
        self.total_updates = 0
        self.total_games = 0
        self.best_win_rate = 0.0
        self.last_ghost_game = 0
        self.last_eval_wr = None

        # Running stats
        self.recent_policy_entropy = deque(maxlen=1000)
        self.recent_value_loss = deque(maxlen=1000)
        self.recent_policy_loss = deque(maxlen=1000)
        self.recent_advantages = deque(maxlen=1000)
        self.recent_clip_fractions = deque(maxlen=1000)
        self.recent_approx_kl = deque(maxlen=1000)

        self.metrics_history = []

    def rebuild_optimizer(self, cnn_lr_scale=0.1):
        """Rebuild optimizer with differential LR after unfreezing CNN."""
        base_lr = self.optimizer.param_groups[0]['lr']
        cnn_params = [p for p in self.model.cnn.parameters() if p.requires_grad]
        cnn_ids = {id(p) for p in cnn_params}
        other_params = [p for p in self.model.parameters()
                        if p.requires_grad and id(p) not in cnn_ids]

        self.optimizer = optim.Adam([
            {'params': cnn_params, 'lr': base_lr * cnn_lr_scale},
            {'params': other_params, 'lr': base_lr},
        ], weight_decay=WEIGHT_DECAY)

        cnn_count = sum(p.numel() for p in cnn_params)
        other_count = sum(p.numel() for p in other_params)
        print(f"[V8Trainer] Rebuilt optimizer: CNN {cnn_count:,} params @ {base_lr * cnn_lr_scale:.1e}, "
              f"Transformer+heads {other_count:,} params @ {base_lr:.1e}")

    def train_on_game(self, trajectories, winner, model_player):
        """Buffer a completed game's trajectory for PPO update."""
        if winner == -1:
            return {}

        trajectory = trajectories.get(model_player, [])
        if not trajectory:
            return {}

        loss_penalty = -1.0 / max(1, (NUM_ACTIVE_PLAYERS - 1))
        z = 1.0 if model_player == winner else loss_penalty

        # Compute discounted returns
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

        for i, step in enumerate(trajectory):
            self._ppo_buffer.append({
                'grids': step['grids'],              # (K, 17, 15, 15)
                'prev_actions': step['prev_actions'], # (K,)
                'seq_mask': step['seq_mask'],         # (K,)
                'action': step['action'],
                'legal_mask': step['legal_mask'],
                'old_log_prob': step['old_log_prob'],
                'temperature': step.get('temperature', 1.0),
                'z': discounted_returns[i],
            })
        self._ppo_games_buffered += 1

        if self._ppo_games_buffered >= self.ppo_buffer_games:
            return self._ppo_update()
        return {}

    def _ppo_update(self):
        """Run K epochs of mini-batch PPO training."""
        if not self._ppo_buffer:
            return {}

        self.model.train()
        n_steps = len(self._ppo_buffer)

        # Stack all grid data
        all_grids = torch.from_numpy(
            np.stack([s['grids'] for s in self._ppo_buffer])
        ).to(self.device, dtype=torch.float32)  # (N, K, 17, 15, 15)

        all_acts = torch.from_numpy(
            np.stack([s['prev_actions'] for s in self._ppo_buffer])
        ).to(self.device)  # (N, K) int64

        all_seq_mask = torch.from_numpy(
            np.stack([s['seq_mask'] for s in self._ppo_buffer])
        ).to(self.device)  # (N, K) bool

        all_actions = torch.tensor(
            [s['action'] for s in self._ppo_buffer],
            dtype=torch.long, device=self.device,
        )

        all_masks = torch.from_numpy(
            np.stack([s['legal_mask'] for s in self._ppo_buffer])
        ).to(self.device, dtype=torch.float32)

        all_old_lp = torch.tensor(
            [s['old_log_prob'] for s in self._ppo_buffer],
            dtype=torch.float32, device=self.device,
        )

        all_temperatures = torch.tensor(
            [s.get('temperature', 1.0) for s in self._ppo_buffer],
            dtype=torch.float32, device=self.device,
        )

        all_returns = torch.tensor(
            [s['z'] for s in self._ppo_buffer],
            dtype=torch.float32, device=self.device,
        )

        # Pre-compute CNN features ONCE when CNN is frozen (saves ~16× compute)
        use_cached = self.model.cnn_frozen
        if use_cached:
            all_cached_cnn = self.model.compute_cnn_features(all_grids)  # (N, K, 128)

        # Track stats
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_advantage = 0.0
        total_clip_frac = 0.0
        total_approx_kl = 0.0
        n_minibatches = 0

        # Precompute advantages
        with torch.no_grad():
            if use_cached:
                _, all_values = self.model.forward_cached(all_cached_cnn, all_acts, all_seq_mask, all_masks)
            else:
                _, all_values = self.model(all_grids, all_acts, all_seq_mask, all_masks)
            all_values = all_values.squeeze(-1)
            all_advantages = all_returns - all_values
            adv_mean = all_advantages.mean()
            adv_std = all_advantages.std()
            all_advantages = (all_advantages - adv_mean) / (adv_std + 1e-8)

        for epoch in range(self.ppo_epochs):
            indices = np.random.permutation(n_steps)

            for start in range(0, n_steps, self.ppo_minibatch_size):
                end = min(start + self.ppo_minibatch_size, n_steps)
                mb_idx = indices[start:end]

                mb_acts = all_acts[mb_idx]
                mb_seq_mask = all_seq_mask[mb_idx]
                mb_actions = all_actions[mb_idx]
                mb_masks = all_masks[mb_idx]
                mb_old_lp = all_old_lp[mb_idx]
                mb_temperatures = all_temperatures[mb_idx]
                mb_returns = all_returns[mb_idx]
                mb_advantages = all_advantages[mb_idx]

                # Forward pass (use cached CNN features when available)
                if use_cached:
                    mb_cached = all_cached_cnn[mb_idx]
                    policy, value = self.model.forward_cached(
                        mb_cached, mb_acts, mb_seq_mask, mb_masks
                    )
                else:
                    mb_grids = all_grids[mb_idx]
                    policy, value = self.model(
                        mb_grids, mb_acts, mb_seq_mask, mb_masks
                    )
                value = value.squeeze(-1)

                advantage = mb_advantages

                # Reconstruct behavior policy
                behavior_temps = mb_temperatures.clamp_min(1e-6).unsqueeze(1)
                behavior_logits = torch.log(policy + 1e-8) / behavior_temps
                behavior_policy = F.softmax(behavior_logits, dim=1)
                new_lp = torch.log(
                    behavior_policy.gather(1, mb_actions.unsqueeze(1)).squeeze(1) + 1e-8
                )

                # PPO ratio
                raw_ratio = torch.exp(new_lp - mb_old_lp)
                ratio = torch.clamp(raw_ratio, 0.0, 10.0)

                # Clipped surrogate
                surr1 = ratio * advantage
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                ) * advantage
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (Huber)
                value_loss = F.smooth_l1_loss(value, mb_returns)

                # Entropy
                entropy = -(policy * torch.log(policy + 1e-8)).sum(dim=1).mean()

                # Combined loss
                loss = (
                    policy_loss
                    + self.value_loss_coeff * value_loss
                    - self.entropy_coeff * entropy
                )

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.max_grad_norm,
                )
                self.optimizer.step()
                self.total_updates += 1

                # Stats
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_advantage += advantage.mean().item()

                with torch.no_grad():
                    clipped = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                    approx_kl = (mb_old_lp - new_lp).mean().item()
                total_clip_frac += clipped
                total_approx_kl += abs(approx_kl)
                n_minibatches += 1

        # Record stats
        if n_minibatches > 0:
            avg_pl = total_policy_loss / n_minibatches
            avg_vl = total_value_loss / n_minibatches
            avg_ent = total_entropy / n_minibatches
            avg_adv = total_advantage / n_minibatches
            avg_clip = total_clip_frac / n_minibatches
            avg_kl = total_approx_kl / n_minibatches
            self.recent_policy_loss.append(avg_pl)
            self.recent_value_loss.append(avg_vl)
            self.recent_policy_entropy.append(avg_ent)
            self.recent_advantages.append(avg_adv)
            self.recent_clip_fractions.append(avg_clip)
            self.recent_approx_kl.append(avg_kl)

        games_trained = self._ppo_games_buffered
        self._ppo_buffer.clear()
        self._ppo_games_buffered = 0

        return {
            'policy_loss': avg_pl if n_minibatches > 0 else 0,
            'value_loss': avg_vl if n_minibatches > 0 else 0,
            'entropy': avg_ent if n_minibatches > 0 else 0,
            'total_steps': n_steps,
            'ppo_games': games_trained,
            'ppo_minibatches': n_minibatches,
        }

    def flush_buffer(self):
        if self._ppo_buffer:
            print(f"[V8Trainer] Flushing PPO buffer ({self._ppo_games_buffered} games, {len(self._ppo_buffer)} steps)")
            self._ppo_update()

    # --- Stats accessors (same as V6/V7) ---
    def get_avg_entropy(self):
        return float(np.mean(self.recent_policy_entropy)) if self.recent_policy_entropy else 0.0

    def get_avg_value_loss(self):
        return float(np.mean(self.recent_value_loss)) if self.recent_value_loss else 0.0

    def get_avg_policy_loss(self):
        return float(np.mean(self.recent_policy_loss)) if self.recent_policy_loss else 0.0

    def get_avg_advantage(self):
        return float(np.mean(self.recent_advantages)) if self.recent_advantages else 0.0

    def get_avg_clip_fraction(self):
        return float(np.mean(self.recent_clip_fractions)) if self.recent_clip_fractions else 0.0

    def get_avg_approx_kl(self):
        return float(np.mean(self.recent_approx_kl)) if self.recent_approx_kl else 0.0

    # --- Ghost management (same as V7) ---
    def maybe_save_ghost(self, elo_tracker=None):
        games_since_ghost = self.total_games - self.last_ghost_game
        if games_since_ghost >= GHOST_SAVE_INTERVAL:
            ghost_path = os.path.join(GHOSTS_DIR, f"ghost_{self.total_games:06d}.pt")
            os.makedirs(GHOSTS_DIR, exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'total_games': self.total_games,
            }, ghost_path)
            self.last_ghost_game = self.total_games
            self._cleanup_ghosts(elo_tracker)
            return True
        return False

    def _cleanup_ghosts(self, elo_tracker=None):
        if not os.path.exists(GHOSTS_DIR):
            return
        ghosts = sorted([f for f in os.listdir(GHOSTS_DIR) if f.endswith('.pt')])
        while len(ghosts) > MAX_GHOSTS:
            victim = ghosts[0]
            if elo_tracker is not None:
                ghost_elos = []
                for g in ghosts:
                    name = g.replace('.pt', '')
                    elo = elo_tracker.get_rating(name)
                    ghost_elos.append((elo, g))
                ghost_elos.sort()
                victim = ghost_elos[0][1]
            os.remove(os.path.join(GHOSTS_DIR, victim))
            ghosts.remove(victim)

    # --- Live stats (same as V7) ---
    def write_live_stats(self, win_rate, gpm, temperature=1.0,
                         eval_wr=None, elo_tracker=None, game_db=None):
        stats = {
            'total_games': self.total_games,
            'total_updates': self.total_updates,
            'win_rate_100': round(win_rate * 100, 1),
            'policy_entropy': round(self.get_avg_entropy(), 4),
            'avg_value_loss': round(self.get_avg_value_loss(), 6),
            'avg_policy_loss': round(self.get_avg_policy_loss(), 6),
            'avg_advantage': round(self.get_avg_advantage(), 4),
            'clip_fraction': round(self.get_avg_clip_fraction(), 4),
            'approx_kl': round(self.get_avg_approx_kl(), 6),
            'temperature': round(temperature, 3),
            'games_per_minute': round(gpm, 1),
            'best_eval_win_rate': round(self.best_win_rate * 100, 1),
            'ghost_count': len([f for f in os.listdir(GHOSTS_DIR) if f.endswith('.pt')]) if os.path.exists(GHOSTS_DIR) else 0,
            'is_stagnated': getattr(self, 'is_stagnated', False),
            'play_alarm': getattr(self, 'play_alarm', False),
            'timestamp': time.time(),
        }
        if eval_wr is not None:
            self.last_eval_wr = eval_wr
            stats['eval_win_rate'] = round(eval_wr * 100, 1)
        elif getattr(self, 'last_eval_wr', None) is not None:
            stats['eval_win_rate'] = round(self.last_eval_wr * 100, 1)

        if elo_tracker is not None:
            stats['main_elo'] = round(elo_tracker.get_rating('Model'), 1)
            stats['elo_rankings'] = [
                {'name': n, 'elo': round(e, 1)}
                for n, e in elo_tracker.get_rankings(top_n=15)
            ]

        if game_db is not None:
            stats['opponent_stats'] = game_db.get_opponent_stats(model_name='Model')
            stats['db_total'] = game_db.get_total_games()

        try:
            with open(STATS_PATH, 'w') as f:
                json.dump(stats, f, indent=2)
        except Exception:
            pass

    def log_metrics(self, win_rate, games_played, eval_win_rate=None):
        entry = {
            "games": games_played,
            "updates": self.total_updates,
            "win_rate": round(win_rate, 4),
            "policy_entropy": round(self.get_avg_entropy(), 4),
            "avg_value_loss": round(self.get_avg_value_loss(), 6),
            "avg_policy_loss": round(self.get_avg_policy_loss(), 6),
            "timestamp": time.time(),
        }
        if eval_win_rate is not None:
            entry["eval_win_rate"] = round(eval_win_rate, 4)
        self.metrics_history.append(entry)
        try:
            with open(METRICS_PATH, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
        except Exception as e:
            print(f"[V8Trainer] Failed to save metrics: {e}")

    # --- Checkpoint management (same as V7) ---
    def save_checkpoint(self, path=None, is_best=False):
        path = path or MAIN_CKPT_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)

        raw_state_dict = self.model.state_dict()
        clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in raw_state_dict.items()}

        save_dict = {
            'model_state_dict': clean_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_updates': self.total_updates,
            'total_games': self.total_games,
            'best_win_rate': self.best_win_rate,
            'last_ghost_game': self.last_ghost_game,
            'metrics_history': self.metrics_history,
            'last_eval_wr': getattr(self, 'last_eval_wr', None),
        }

        tmp_path = path + '.tmp'
        torch.save(save_dict, tmp_path)
        os.replace(tmp_path, path)

        if is_best:
            torch.save({
                'model_state_dict': clean_state_dict,
                'total_games': self.total_games,
                'best_win_rate': self.best_win_rate,
            }, BEST_CKPT_PATH)

    def load_checkpoint(self, path=None):
        path = path or MAIN_CKPT_PATH
        if not os.path.exists(path):
            return False
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

            is_compiled = hasattr(self.model, '_orig_mod')
            if is_compiled:
                state_dict = {'_orig_mod.' + k: v for k, v in state_dict.items()}

            self.model.load_state_dict(state_dict)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                if 'optimizer_state_dict' in checkpoint:
                    try:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    except Exception as e:
                        print(f"[V8Trainer] Warning: Could not load optimizer state: {e}")
                self.total_updates = checkpoint.get('total_updates', 0)
                self.total_games = checkpoint.get('total_games', 0)
                self.best_win_rate = checkpoint.get('best_win_rate', 0.0)
                self.last_ghost_game = checkpoint.get('last_ghost_game', 0)
                self.metrics_history = checkpoint.get('metrics_history', [])
                self.last_eval_wr = checkpoint.get('last_eval_wr', None)
                print(f"[V8Trainer] Loaded checkpoint: {self.total_games} games, {self.total_updates} updates")
            else:
                self.total_updates = 0
                self.total_games = 0
                self.best_win_rate = 0.0
                self.last_ghost_game = 0
                self.metrics_history = []
                print(f"[V8Trainer] Loaded raw model weights (fresh start)")
            return True
        except Exception as e:
            print(f"[V8Trainer] Failed to load checkpoint: {e}")
            return False
