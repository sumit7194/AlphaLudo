"""
TD-Ludo Trainer — PPO (Proximal Policy Optimization)

Training paradigm:
- Policy head (Actor): Learns π(a|s) — which move to make
- Value head (Critic): Learns V(s) — win probability (used as variance-reducing baseline)
- Rewards: Pure sparse (+1 win, -1 loss)
- Targets: Game outcome z (Monte Carlo return) — no bootstrapping
- Algorithm: PPO with clipped surrogate objective

PPO collects trajectories from N games, then does K epochs of mini-batch training:
  1. Buffer game trajectories until PPO_BUFFER_GAMES reached
  2. Stack all steps, shuffle, split into mini-batches
  3. For each mini-batch:
     ratio = π_new(a|s) / π_old(a|s)
     policy_loss = -min(ratio * A, clip(ratio, 1±ε) * A)
     value_loss = MSE(V(s), z)
     entropy = -Σ π·log(π)
     total = policy_loss + 0.5*value_loss - 0.05*entropy
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


class ActorCriticTrainer:
    """
    Actor-Critic trainer using REINFORCE with baseline.
    
    Training happens at the END of each game:
    1. Collect trajectory: [(state, action, legal_mask), ...] per player
    2. Assign outcome z = +1 (winner) or -1 (loser)
    3. For each step: compute advantage = z - V(s), update π and V
    """
    
    def __init__(self, model, device, learning_rate=LEARNING_RATE):
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=WEIGHT_DECAY
        )
        
        self.max_grad_norm = MAX_GRAD_NORM
        self.entropy_coeff = ENTROPY_COEFF
        self.value_loss_coeff = VALUE_LOSS_COEFF
        
        # PPO parameters
        self.clip_epsilon = CLIP_EPSILON
        self.ppo_epochs = PPO_EPOCHS
        self.ppo_buffer_games = PPO_BUFFER_GAMES
        self.ppo_minibatch_size = PPO_MINIBATCH_SIZE
        
        # PPO buffer (accumulates steps across games, trains in batches)
        self._ppo_buffer = []
        self._ppo_games_buffered = 0

        # Running return normalization (fixes value head positive bias)
        # Without this, returns are always positive (+1 to +7) due to dense rewards,
        # causing the value head to predict ~+3 for all states and providing
        # poor advantage estimates. Normalizing centers returns around 0.
        self._return_running_mean = 0.0
        self._return_running_std = 1.0
        self._return_stats_initialized = False
        
        # Tracking
        self.total_updates = 0
        self.total_games = 0
        self.best_win_rate = 0.0
        self.last_ghost_game = 0
        self.last_eval_wr = None
        
        # Running stats for dashboard
        self.recent_policy_entropy = deque(maxlen=1000)
        self.recent_value_loss = deque(maxlen=1000)
        self.recent_policy_loss = deque(maxlen=1000)
        self.recent_advantages = deque(maxlen=1000)
        self.recent_clip_fractions = deque(maxlen=1000)
        self.recent_approx_kl = deque(maxlen=1000)
        
        # Metrics history (appended to file)
        self.metrics_history = []

    def train_on_game(self, trajectories, winner, model_player):
        """
        Buffer a completed game's trajectory for the next PPO update.
        
        Training only happens when the buffer reaches PPO_BUFFER_GAMES.
        
        Args:
            trajectories: dict mapping player_id → list of step dicts
            winner: int, the player_id who won (or -1 for draw)
            model_player: int, which player was controlled by the model
            
        Returns:
            dict with training stats (only when PPO update fires)
        """
        if winner == -1:
            return {}  # Skip draws
        
        # Only use the model player's trajectory
        trajectory = trajectories.get(model_player, [])
        if not trajectory:
            return {}
        
        # Outcome from model's perspective (sparse terminal signal)
        loss_penalty = -1.0 / max(1, (NUM_ACTIVE_PLAYERS - 1))
        z = 1.0 if model_player == winner else loss_penalty
        
        # Compute discounted returns (Monte Carlo with dense rewards)
        # Gamma must be extreme (0.999) because Ludo games span 150+ moves.
        # This prevents dense captures (+0.20) from eclipsing the terminal Win/Loss signal.
        gamma = 0.999
        discounted_returns = []
        R = 0.0
        
        for i in reversed(range(len(trajectory))):
            step = trajectory[i]
            shaped_reward = step.get('step_reward', 0.0)
            
            # Terminal signal only applies to the very last step
            if i == len(trajectory) - 1:
                r_t = shaped_reward + z
            else:
                r_t = shaped_reward
                
            R = r_t + gamma * R
            discounted_returns.insert(0, R)
        
        # Buffer each step with its corresponding return
        for i, step in enumerate(trajectory):
            self._ppo_buffer.append({
                'state': step['state'],
                'action': step['action'],
                'legal_mask': step['legal_mask'],
                'old_log_prob': step['old_log_prob'],
                'temperature': step.get('temperature', 1.0),
                'z': discounted_returns[i],  # True discounted return for this specific step
            })
        self._ppo_games_buffered += 1
        
        # If buffer is full, run PPO update
        if self._ppo_games_buffered >= self.ppo_buffer_games:
            return self._ppo_update()
        
        return {}
    
    def _ppo_update(self):
        """
        Run K epochs of mini-batch PPO training over the buffered data.
        
        This is where the actual learning happens:
        1. Stack all buffered steps into tensors
        2. For each epoch, shuffle and split into mini-batches
        3. For each mini-batch: compute ratio, clip, update
        """
        if not self._ppo_buffer:
            return {}
        
        self.model.train()
        
        n_steps = len(self._ppo_buffer)
        
        # Stack all buffered data into tensors
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

        # Normalize returns to zero-mean, unit-std using running statistics.
        # Raw returns are always positive (+1 to +7 from dense rewards), which
        # causes the value head to predict ~+3 for all states. Normalizing brings
        # returns into a bounded range centered at 0, making the value head's
        # job tractable and preventing the slow policy degradation observed
        # from 170K→820K games.
        with torch.no_grad():
            batch_mean = all_returns_raw.mean().item()
            batch_std = all_returns_raw.std().item()
            if not self._return_stats_initialized:
                self._return_running_mean = batch_mean
                self._return_running_std = max(batch_std, 1e-6)
                self._return_stats_initialized = True
            else:
                # Exponential moving average (α=0.01 for stability)
                self._return_running_mean = 0.99 * self._return_running_mean + 0.01 * batch_mean
                self._return_running_std = 0.99 * self._return_running_std + 0.01 * max(batch_std, 1e-6)

            all_returns = (all_returns_raw - self._return_running_mean) / (self._return_running_std + 1e-8)

        # Track stats across all mini-batches
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_advantage = 0.0
        total_clip_frac = 0.0
        total_approx_kl = 0.0
        n_minibatches = 0

        # Precompute and normalize advantages over the *entire* buffer once per update
        with torch.no_grad():
            all_values = self.model(all_states, all_masks)[1].squeeze(-1)
            all_advantages = all_returns - all_values
            adv_mean = all_advantages.mean()
            adv_std = all_advantages.std()
            all_advantages = (all_advantages - adv_mean) / (adv_std + 1e-8)
            
        for epoch in range(self.ppo_epochs):
            # Shuffle indices for this epoch
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
                
                # Forward pass
                policy, value = self.model(mb_states, mb_masks)
                value = value.squeeze(-1)
                
                # Advantage: Pre-computed and Normalized over batch
                advantage = mb_advantages
                
                # Reconstruct the exact behavior policy used for sampling.
                behavior_temps = mb_temperatures.clamp_min(1e-6).unsqueeze(1)
                behavior_logits = torch.log(policy + 1e-8) / behavior_temps
                behavior_policy = F.softmax(behavior_logits, dim=1)
                new_lp = torch.log(
                    behavior_policy.gather(1, mb_actions.unsqueeze(1)).squeeze(1) + 1e-8
                )
                
                # PPO ratio: π_new(a|s) / π_old(a|s)
                raw_ratio = torch.exp(new_lp - mb_old_lp)
                # Safeguard against ratio explosions caused by temperature-induced low-prob exploration
                ratio = torch.clamp(raw_ratio, 0.0, 10.0) 
                
                # Clipped surrogate objective
                surr1 = ratio * advantage
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                ) * advantage
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss: Huber loss (smooth L1) protects against exploding Critic gradients
                value_loss = F.smooth_l1_loss(value, mb_returns)
                
                # Entropy bonus: -Σ π·log(π)
                entropy = -(policy * torch.log(policy + 1e-8)).sum(dim=1).mean()
                
                # Combined loss
                loss = (policy_loss 
                        + self.value_loss_coeff * value_loss 
                        - self.entropy_coeff * entropy)
                
                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.total_updates += 1
                
                # Accumulate stats
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_advantage += advantage.mean().item()
                
                # PPO diagnostics: clip fraction and approx KL
                with torch.no_grad():
                    clipped = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                    approx_kl = (mb_old_lp - new_lp).mean().item()
                total_clip_frac += clipped
                total_approx_kl += abs(approx_kl)
                n_minibatches += 1
        
        # Record average stats
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
        
        # Clear the buffer
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
        """Train on any remaining buffered data (called before shutdown)."""
        if self._ppo_buffer:
            print(f"[Trainer] Flushing PPO buffer ({self._ppo_games_buffered} games, {len(self._ppo_buffer)} steps)")
            self._ppo_update()
    
    def get_avg_entropy(self):
        """Average policy entropy over recent updates."""
        if not self.recent_policy_entropy:
            return 0.0
        return float(np.mean(self.recent_policy_entropy))
    
    def get_avg_value_loss(self):
        """Average value loss over recent updates."""
        if not self.recent_value_loss:
            return 0.0
        return float(np.mean(self.recent_value_loss))
    
    def get_avg_policy_loss(self):
        """Average policy loss over recent updates."""
        if not self.recent_policy_loss:
            return 0.0
        return float(np.mean(self.recent_policy_loss))
    
    def get_avg_advantage(self):
        """Average advantage over recent updates."""
        if not self.recent_advantages:
            return 0.0
        return float(np.mean(self.recent_advantages))
    
    def get_avg_clip_fraction(self):
        """Average PPO clip fraction (should be 0.1-0.3 when healthy)."""
        if not self.recent_clip_fractions:
            return 0.0
        return float(np.mean(self.recent_clip_fractions))
    
    def get_avg_approx_kl(self):
        """Average approximate KL divergence between old and new policies."""
        if not self.recent_approx_kl:
            return 0.0
        return float(np.mean(self.recent_approx_kl))
    
    def predict_value(self, state_tensor):
        """Predict V(s) for a single state tensor (for logging/debugging)."""
        self.model.eval()
        with torch.no_grad():
            t = torch.from_numpy(state_tensor).unsqueeze(0).to(self.device, dtype=torch.float32)
            _, v = self.model(t)
            return v.item()
    
    def predict_value_batch(self, state_tensors):
        """Predict V(s) for a batch of state tensors."""
        self.model.eval()
        with torch.no_grad():
            t = torch.from_numpy(state_tensors).to(self.device, dtype=torch.float32)
            _, v = self.model(t)
            return v.squeeze(-1).cpu().numpy()
    
    def maybe_save_ghost(self, elo_tracker=None):
        """Save a ghost snapshot if enough games have passed."""
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
        """Prune ghosts beyond MAX_GHOSTS. Uses Elo-based pruning if available."""
        if not os.path.exists(GHOSTS_DIR):
            return
        ghosts = sorted([f for f in os.listdir(GHOSTS_DIR) if f.endswith('.pt')])
        while len(ghosts) > MAX_GHOSTS:
            # Simple: remove oldest
            victim = ghosts[0]
            if elo_tracker is not None:
                # Find lowest Elo ghost
                ghost_elos = []
                for g in ghosts:
                    name = g.replace('.pt', '')
                    elo = elo_tracker.get_rating(name)
                    ghost_elos.append((elo, g))
                ghost_elos.sort()
                victim = ghost_elos[0][1]
            
            os.remove(os.path.join(GHOSTS_DIR, victim))
            ghosts.remove(victim)
    
    def write_live_stats(self, win_rate, gpm, temperature=1.0,
                         eval_wr=None, elo_tracker=None, game_db=None):
        """Write live stats JSON for dashboard consumption."""
        
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
        
        # Enrich with Elo data
        if elo_tracker is not None:
            stats['main_elo'] = round(elo_tracker.get_rating('Model'), 1)
            stats['elo_rankings'] = [
                {'name': n, 'elo': round(e, 1)} 
                for n, e in elo_tracker.get_rankings(top_n=15)
            ]
        
        # Enrich with opponent stats from DB
        if game_db is not None:
            stats['opponent_stats'] = game_db.get_opponent_stats(model_name='Model')
            stats['db_total'] = game_db.get_total_games()
            
            # --- Last 500 Games Opponent Stats ---
            try:
                import sqlite3
                with sqlite3.connect(game_db.db_path) as conn:
                    c = conn.cursor()
                    c.execute('''
                        SELECT
                            CASE WHEN model_player_idx = 0 THEN p2 ELSE p0 END as opponent,
                            COUNT(*) as total_games,
                            SUM(CASE WHEN winner = model_player_idx THEN 1 ELSE 0 END) as wins
                        FROM (SELECT * FROM games ORDER BY id DESC LIMIT 500)
                        GROUP BY opponent
                    ''')
                    recent_stats = {}
                    for row in c.fetchall():
                        opp_name = row[0]
                        games = row[1]
                        wins = row[2]
                        if opp_name not in ('Inactive', 'Model') and games > 0:
                            recent_stats[opp_name] = {
                                'wins': wins,
                                'games': games,
                                'win_rate': round((wins / games) * 100, 1)
                            }
                    stats['recent_opponent_stats'] = recent_stats
            except Exception:
                pass
        
        try:
            with open(STATS_PATH, 'w') as f:
                json.dump(stats, f, indent=2)
        except Exception:
            pass
    
    def log_metrics(self, win_rate, games_played, eval_win_rate=None):
        """Append training metrics to history file."""
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
            print(f"[Trainer] Failed to save metrics: {e}")
    
    def save_checkpoint(self, path=None, is_best=False):
        """Save full training state (model + optimizer)."""
        path = path or MAIN_CKPT_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Always save "clean" weights (strip torch.compile prefix if present)
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
            'return_running_mean': self._return_running_mean,
            'return_running_std': self._return_running_std,
        }
        
        # Save to temp file first, then rename (atomic)
        tmp_path = path + '.tmp'
        torch.save(save_dict, tmp_path)
        os.replace(tmp_path, path)
        
        if is_best:
            raw_state_dict = self.model.state_dict()
            clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in raw_state_dict.items()}
            torch.save({
                'model_state_dict': clean_state_dict,
                'total_games': self.total_games,
                'best_win_rate': self.best_win_rate,
            }, BEST_CKPT_PATH)
    
    def load_checkpoint(self, path=None):
        """Load full training state. Returns True if successful."""
        path = path or MAIN_CKPT_PATH
        if not os.path.exists(path):
            return False
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            
            # Handle both formats: wrapped dict or raw state_dict
            # Extract the raw state dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # 1. Strip '_orig_mod.' from checkpoint if it exists (e.g. from an interrupted compiled run)
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            
            # 2. If current model is compiled, add '_orig_mod.' back to keys so load_state_dict matches
            is_compiled = hasattr(self.model, '_orig_mod')
            if is_compiled:
                state_dict = {'_orig_mod.' + k: v for k, v in state_dict.items()}
            
            # Load mapped states
            self.model.load_state_dict(state_dict)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Full checkpoint
                if 'optimizer_state_dict' in checkpoint:
                    try:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    except Exception as e:
                        print(f"[Trainer] Warning: Could not load optimizer state: {e}")
                self.total_updates = checkpoint.get('total_updates', 0)
                self.total_games = checkpoint.get('total_games', 0)
                self.best_win_rate = checkpoint.get('best_win_rate', 0.0)
                self.last_ghost_game = checkpoint.get('last_ghost_game', 0)
                self.metrics_history = checkpoint.get('metrics_history', [])
                self.last_eval_wr = checkpoint.get('last_eval_wr', None)
                # Restore return normalization stats (if available)
                if 'return_running_mean' in checkpoint:
                    self._return_running_mean = checkpoint['return_running_mean']
                    self._return_running_std = checkpoint['return_running_std']
                    self._return_stats_initialized = True
                print(f"[Trainer] Loaded full checkpoint: {self.total_games} games, {self.total_updates} updates")
            else:
                # Raw weights
                self.total_updates = 0
                self.total_games = 0
                self.best_win_rate = 0.0
                self.last_ghost_game = 0
                self.metrics_history = []
                print(f"[Trainer] Loaded raw model weights (fresh optimizer, counters reset to 0)")
            return True
        except Exception as e:
            print(f"[Trainer] Failed to load checkpoint: {e}")
            return False
