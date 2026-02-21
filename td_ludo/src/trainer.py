"""
TD-Ludo Trainer — Online TD(0) Value Learning + Prioritized Experience Replay

Features:
- Online TD(0) semi-gradient updates after each move
- Prioritized experience buffer with TD-error-weighted sampling (PER)
- Numpy-based buffer persistence for fast save/load
- Ghost checkpoint saving at regular intervals
- Metrics logging for dashboard consumption
- Graceful checkpoint save/load with optimizer state
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

from src.model import AlphaLudoV3
from src.config import (
    LEARNING_RATE, WEIGHT_DECAY, MAX_GRAD_NORM,
    GRAD_ACCUM_STEPS, TD_GAMMA, CHECKPOINT_DIR,
    MAIN_CKPT_PATH, BEST_CKPT_PATH, METRICS_PATH,
    GHOSTS_DIR, MAX_GHOSTS, GHOST_SAVE_INTERVAL,
    USE_EXPERIENCE_BUFFER, BUFFER_SIZE, BUFFER_PATH,
    REPLAY_BATCH_SIZE, REPLAY_STEPS, STATS_PATH,
    PER_ALPHA, PER_BETA_START, PER_BETA_END, PER_BETA_ANNEAL_GAMES,
)


# Tensor shape for state: (11, 15, 15)
STATE_SHAPE = (11, 15, 15)
STATE_SIZE = 11 * 15 * 15  # 2475 floats


class PrioritizedExperienceBuffer:
    """
    Prioritized Experience Replay buffer with numpy-based persistence.
    
    - FIFO eviction (ring buffer) — simple, no overhead
    - TD-error-weighted sampling — focuses learning on "surprising" transitions
    - Importance sampling correction — prevents gradient bias from non-uniform sampling
    - Numpy-based save/load — fast I/O, ~80% smaller than torch.save of individual tensors
    
    Each transition stores: (state, next_state, reward, done, priority)
    Priority = |TD error|^α, where α controls how much prioritization matters.
    """
    
    def __init__(self, max_size=BUFFER_SIZE, alpha=PER_ALPHA):
        self.max_size = max_size
        self.alpha = alpha
        
        # Pre-allocate numpy arrays for the ring buffer
        self.states = np.zeros((max_size, *STATE_SHAPE), dtype=np.float32)
        self.next_states = np.zeros((max_size, *STATE_SHAPE), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.float32)
        self.priorities = np.ones(max_size, dtype=np.float32)  # Start with max priority
        
        self.position = 0  # Next write position
        self.size = 0       # Current number of valid entries
        self.max_priority = 1.0  # Track max priority for new entries
    
    def add(self, state, next_state, reward, done):
        """Add a single transition."""
        self.states[self.position] = state
        self.next_states[self.position] = next_state
        self.rewards[self.position] = reward
        self.dones[self.position] = done
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, states, next_states, rewards, dones):
        """Add a batch of transitions efficiently."""
        batch_size = len(states)
        if batch_size == 0: return

        # Handle wrap-around
        if self.position + batch_size <= self.max_size:
            idx = slice(self.position, self.position + batch_size)
            self.states[idx] = states
            self.next_states[idx] = next_states
            self.rewards[idx] = rewards
            self.dones[idx] = dones
            self.priorities[idx] = self.max_priority
        else:
            # Split into two parts
            first_part = self.max_size - self.position
            idx1 = slice(self.position, self.max_size)
            self.states[idx1] = states[:first_part]
            self.next_states[idx1] = next_states[:first_part]
            self.rewards[idx1] = rewards[:first_part]
            self.dones[idx1] = dones[:first_part]
            self.priorities[idx1] = self.max_priority

            second_part = batch_size - first_part
            idx2 = slice(0, second_part)
            self.states[idx2] = states[first_part:]
            self.next_states[idx2] = next_states[first_part:]
            self.rewards[idx2] = rewards[first_part:]
            self.dones[idx2] = dones[first_part:]
            self.priorities[idx2] = self.max_priority

        self.position = (self.position + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)
    
    def sample(self, batch_size, beta=0.4):
        """Sample with prioritized indices."""
        batch_size = min(batch_size, self.size)
        
        # O(N) but optimized with numpy
        priorities = self.priorities[:self.size]
        probs = np.power(priorities, self.alpha)
        prob_sum = probs.sum()
        if prob_sum == 0:
            probs = np.ones(self.size, dtype=np.float32) / self.size
        else:
            probs /= prob_sum
        
        # CRITICAL: use replace=True to get O(N) sampling instead of O(N log N) or worse.
        # For PER, replace=True is standard and fast.
        indices = np.random.choice(self.size, size=batch_size, replace=True, p=probs)
        
        # Importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        # Batch gather
        states = torch.from_numpy(self.states[indices])
        next_states = torch.from_numpy(self.next_states[indices])
        rewards = torch.from_numpy(self.rewards[indices])
        dones = torch.from_numpy(self.dones[indices])
        is_weights = torch.from_numpy(weights.astype(np.float32))
        
        return states, next_states, rewards, dones, indices, is_weights
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on new TD errors from replay."""
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().cpu().numpy()
        
        new_priorities = np.abs(td_errors) + 1e-6  # Small ε to avoid zero priority
        self.priorities[indices] = new_priorities
        self.max_priority = max(self.max_priority, new_priorities.max())
    
    def clear(self):
        """Reset buffer to empty (keeps allocated memory)."""
        self.position = 0
        self.size = 0
        self.max_priority = 1.0
        print("[Buffer] Cleared")
    
    def __len__(self):
        return self.size
    
    def save(self, path=BUFFER_PATH):
        """Save buffer to disk as compressed numpy archive (~80% smaller than torch.save)."""
        try:
            t0 = time.time()
            np.savez_compressed(
                path,
                states=self.states[:self.size],
                next_states=self.next_states[:self.size],
                rewards=self.rewards[:self.size],
                dones=self.dones[:self.size],
                priorities=self.priorities[:self.size],
                position=np.array([self.position]),
                max_priority=np.array([self.max_priority]),
            )
            dt = time.time() - t0
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"[Buffer] Saved {self.size} transitions ({size_mb:.1f} MB, {dt:.1f}s)")
        except Exception as e:
            print(f"[Buffer] Failed to save: {e}")
    
    def load(self, path=BUFFER_PATH):
        """Load buffer from disk. Also tries legacy .pt format for migration."""
        # Try numpy format first
        if os.path.exists(path):
            try:
                t0 = time.time()
                data = np.load(path)
                n = len(data['states'])
                self.states[:n] = data['states']
                self.next_states[:n] = data['next_states']
                self.rewards[:n] = data['rewards']
                self.dones[:n] = data['dones']
                self.priorities[:n] = data['priorities']
                self.position = int(data['position'][0])
                self.max_priority = float(data['max_priority'][0])
                self.size = n
                dt = time.time() - t0
                print(f"[Buffer] Loaded {n} transitions ({dt:.1f}s)")
                return True
            except Exception as e:
                print(f"[Buffer] Failed to load numpy buffer: {e}")
                return False
        
        # Try legacy .pt format (migration path)
        legacy_path = path.replace('.npz', '.pt')
        if os.path.exists(legacy_path):
            try:
                print(f"[Buffer] Migrating legacy buffer from {legacy_path}...")
                t0 = time.time()
                data = torch.load(legacy_path, weights_only=False)
                for i, (s, ns, r, d) in enumerate(data):
                    if i >= self.max_size:
                        break
                    self.states[i] = s.numpy() if isinstance(s, torch.Tensor) else s
                    self.next_states[i] = ns.numpy() if isinstance(ns, torch.Tensor) else ns
                    self.rewards[i] = float(r.item() if isinstance(r, torch.Tensor) else r)
                    self.dones[i] = float(d.item() if isinstance(d, torch.Tensor) else d)
                    self.priorities[i] = 1.0  # Default priority for migrated data
                self.size = min(len(data), self.max_size)
                self.position = self.size % self.max_size
                dt = time.time() - t0
                print(f"[Buffer] Migrated {self.size} transitions from legacy format ({dt:.1f}s)")
                # Save in new format immediately
                self.save(path)
                return True
            except Exception as e:
                print(f"[Buffer] Failed to migrate legacy buffer: {e}")
                return False
        
        return False


class TDTrainer:
    """
    Online TD(0) trainer for value-based Ludo learning.
    
    The model predicts V(s) ∈ [-1, +1] for each board state.
    Training uses semi-gradient TD(0):
        δ = R + γ·V(s') - V(s)
        loss = δ²
        ∇loss only through V(s), not V(s')  (semi-gradient)
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
        
        self.gamma = TD_GAMMA
        self.grad_accum_steps = GRAD_ACCUM_STEPS
        self.max_grad_norm = MAX_GRAD_NORM
        
        # Experience buffer (Prioritized)
        self.experience_buffer = PrioritizedExperienceBuffer() if USE_EXPERIENCE_BUFFER else None
        
        # Tracking
        self.total_updates = 0
        self.total_games = 0
        self.best_win_rate = 0.0
        self.accum_loss = 0.0
        self.accum_count = 0
        self.last_ghost_game = 0
        
        # Metrics history (appended to file)
        self.metrics_history = []
    
    def predict_value(self, state_tensor):
        """Predict V(s) for a single state tensor."""
        self.model.eval()
        with torch.no_grad():
            x = state_tensor.unsqueeze(0).to(self.device, dtype=torch.float32)
            _, value, _ = self.model(x)
            return value.item()
    
    def predict_value_batch(self, state_tensors):
        """Predict V(s) for a batch of state tensors."""
        self.model.eval()
        with torch.no_grad():
            if isinstance(state_tensors, list):
                import numpy as np
                batch = torch.from_numpy(np.stack(state_tensors)).to(self.device, dtype=torch.float32)
            elif isinstance(state_tensors, torch.Tensor):
                batch = state_tensors.to(self.device, dtype=torch.float32)
            else:
                batch = torch.from_numpy(state_tensors).to(self.device, dtype=torch.float32)
            _, values, _ = self.model(batch)
            return values.squeeze(-1).tolist()
    
    def td_update(self, state_tensor, next_state_tensor, reward, done):
        """
        Single TD(0) semi-gradient update.
        
        Returns: float — absolute TD error |δ|
        """
        self.model.train()
        
        s = state_tensor.unsqueeze(0).to(self.device, dtype=torch.float32)
        
        # V(s) — gradient flows through this
        _, v_s, _ = self.model(s)
        v_s = v_s.squeeze()
        
        # V(s') — detached, no gradient (semi-gradient TD)
        with torch.no_grad():
            if done:
                v_next = 0.0
            else:
                s_next = next_state_tensor.unsqueeze(0).to(self.device, dtype=torch.float32)
                _, v_s_next, _ = self.model(s_next)
                v_next = v_s_next.squeeze().item()
        
        # TD target
        target = reward + (1.0 - float(done)) * self.gamma * v_next
        target_tensor = torch.tensor(target, device=self.device, dtype=torch.float32)
        
        # Loss = (target - V(s))²
        td_error = target_tensor - v_s
        loss = td_error ** 2
        
        # Scale for gradient accumulation
        scaled_loss = loss / self.grad_accum_steps
        scaled_loss.backward()
        
        self.accum_loss += loss.item()
        self.accum_count += 1
        
        # Step optimizer every N accumulations
        if self.accum_count >= self.grad_accum_steps:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.total_updates += 1
            self.accum_count = 0
        
        # Store in experience buffer for replay
        if self.experience_buffer is not None:
            self.experience_buffer.add(state_tensor, next_state_tensor, reward, done)
        
        return abs(td_error.item())
    
    def td_update_batch(self, states, next_states, rewards, dones):
        """Batched TD(0) update."""
        self.model.train()
        
        # Support both numpy and torch input to reduce redundant transfers
        s_np = states if isinstance(states, np.ndarray) else states.detach().cpu().numpy()
        ns_np = next_states if isinstance(next_states, np.ndarray) else next_states.detach().cpu().numpy()
        r_np = rewards if isinstance(rewards, np.ndarray) else rewards.detach().cpu().numpy()
        d_np = dones if isinstance(dones, np.ndarray) else dones.detach().cpu().numpy()

        s_gpu = torch.from_numpy(s_np).to(self.device, dtype=torch.float32)
        ns_gpu = torch.from_numpy(ns_np).to(self.device, dtype=torch.float32)
        r_gpu = torch.from_numpy(r_np).to(self.device, dtype=torch.float32)
        d_gpu = torch.from_numpy(d_np).to(self.device, dtype=torch.float32)
        
        # V(s)
        _, v_s, _ = self.model(s_gpu)
        v_s = v_s.squeeze(-1)
        
        # V(s')
        with torch.no_grad():
            _, v_next_all, _ = self.model(ns_gpu)
            v_next_all = v_next_all.squeeze(-1)
            # Mask terminal states
            v_next_all = v_next_all * (1.0 - d_gpu)
            
        # Target (clamped to tanh range to prevent gradient saturation)
        target = torch.clamp(r_gpu + self.gamma * v_next_all, -1.0, 1.0)
        
        # Loss
        loss = F.mse_loss(v_s, target)
        
        # Scale
        scaled_loss = loss / self.grad_accum_steps
        scaled_loss.backward()
        
        self.accum_loss += loss.item()
        self.accum_count += 1
        
        if self.accum_count >= self.grad_accum_steps:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.total_updates += 1
            self.accum_count = 0
            
        # Add to buffer
        if self.experience_buffer is not None:
            self.experience_buffer.add_batch(s_np, ns_np, r_np, d_np)

        return loss.item()
    
    def replay_experience(self):
        """
        Train on prioritized replay from the buffer.
        Uses importance sampling weights to correct gradient bias.
        Updates priorities with fresh TD errors after each batch.
        """
        if self.experience_buffer is None or len(self.experience_buffer) < REPLAY_BATCH_SIZE:
            return 0.0
        
        self.model.train()
        total_loss = 0.0
        
        # Anneal β from PER_BETA_START → PER_BETA_END over training
        beta_progress = min(1.0, self.total_games / max(1, PER_BETA_ANNEAL_GAMES))
        beta = PER_BETA_START + (PER_BETA_END - PER_BETA_START) * beta_progress
        
        for _ in range(REPLAY_STEPS):
            # Sample with priorities
            states, next_states, rewards, dones, indices, is_weights = \
                self.experience_buffer.sample(REPLAY_BATCH_SIZE, beta=beta)
            
            states = states.to(self.device, dtype=torch.float32)
            next_states = next_states.to(self.device, dtype=torch.float32)
            rewards = rewards.to(self.device)
            dones = dones.to(self.device)
            is_weights = is_weights.to(self.device)
            
            # Batch V(s)
            _, v_s, _ = self.model(states)
            v_s = v_s.squeeze(-1)
            
            # Batch V(s') — detached
            with torch.no_grad():
                _, v_next, _ = self.model(next_states)
                v_next = v_next.squeeze(-1)
            
            # TD targets and errors (clamped to tanh range)
            targets = torch.clamp(rewards + (1.0 - dones) * self.gamma * v_next, -1.0, 1.0)
            td_errors = targets - v_s
            
            # IS-weighted MSE loss: weight each transition's loss by its IS weight
            loss = (is_weights * td_errors.pow(2)).mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Update priorities with fresh TD errors
            self.experience_buffer.update_priorities(indices, td_errors)
            
            total_loss += loss.item()
            self.total_updates += 1
        
        return total_loss / REPLAY_STEPS
    
    def flush_gradients(self):
        """Force a gradient step even if accumulation isn't full."""
        if self.accum_count > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.total_updates += 1
            self.accum_count = 0
    
    def maybe_save_ghost(self, elo_tracker=None):
        """Save a ghost snapshot if enough games have passed."""
        games_since_ghost = self.total_games - self.last_ghost_game
        if games_since_ghost >= GHOST_SAVE_INTERVAL:
            ghost_path = os.path.join(GHOSTS_DIR, f"ghost_{self.total_games:06d}.pt")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'total_games': self.total_games,
                'total_updates': self.total_updates,
            }, ghost_path)
            self.last_ghost_game = self.total_games
            print(f"[Ghost] Saved ghost at game {self.total_games}: {ghost_path}")
            
            # Cleanup old ghosts (Elo-based if tracker available, else FIFO)
            self._cleanup_ghosts(elo_tracker)
    
    def _cleanup_ghosts(self, elo_tracker=None):
        """Prune ghosts beyond MAX_GHOSTS. Uses Elo-based pruning if available."""
        try:
            ghosts = sorted([
                f for f in os.listdir(GHOSTS_DIR) if f.startswith('ghost_') and f.endswith('.pt')
            ])
            while len(ghosts) > MAX_GHOSTS:
                if elo_tracker is not None:
                    # Elo-based: remove weakest ghost
                    weakest = elo_tracker.get_weakest_ghost(GHOSTS_DIR)
                    if weakest:
                        path, name, elo = weakest
                        os.remove(path)
                        ghosts = [g for g in ghosts if g != os.path.basename(path)]
                        print(f"[Ghost] Pruned weakest: {name} (Elo {elo:.0f})")
                    else:
                        oldest = ghosts.pop(0)
                        os.remove(os.path.join(GHOSTS_DIR, oldest))
                else:
                    # Fallback: FIFO
                    oldest = ghosts.pop(0)
                    os.remove(os.path.join(GHOSTS_DIR, oldest))
        except Exception:
            pass
    
    def write_live_stats(self, win_rate, td_error, epsilon, gpm, 
                         eval_wr=None, elo_tracker=None, game_db=None):
        """Write live stats JSON for dashboard consumption."""
        
        # Calculate Value Calibration Metrics
        v_mean, v_pre_win, v_pre_loss = 0.0, 0.0, 0.0
        if self.experience_buffer and self.experience_buffer.size > 0:
            import numpy as np
            buf = self.experience_buffer
            size = buf.size
            idx = np.random.choice(size, min(1000, size), replace=False)
            
            states = torch.from_numpy(buf.states[idx]).to(self.device).float()
            self.model.eval()
            with torch.no_grad():
                _, vals, _ = self.model(states)
                vals_np = vals.squeeze(-1).cpu().numpy()
            v_mean = float(vals_np.mean())
            
            d = buf.dones[:size]
            r = buf.rewards[:size]
            win_idx = np.where((d > 0.5) & (np.isclose(r, 1.0, atol=0.01)))[0]
            loss_idx = np.where((d > 0.5) & (np.isclose(r, -1.0, atol=0.01)))[0]
            
            if len(win_idx) > 10:
                ws = torch.from_numpy(buf.states[win_idx[-100:]]).to(self.device).float()
                with torch.no_grad():
                    _, wv, _ = self.model(ws)
                v_pre_win = float(wv.mean().item())
            if len(loss_idx) > 10:
                ls = torch.from_numpy(buf.states[loss_idx[-100:]]).to(self.device).float()
                with torch.no_grad():
                    _, lv, _ = self.model(ls)
                v_pre_loss = float(lv.mean().item())

        stats = {
            'total_games': self.total_games,
            'total_updates': self.total_updates,
            'win_rate_100': round(win_rate * 100, 1),
            'avg_td_error': round(td_error, 6),
            'epsilon': round(epsilon, 4),
            'games_per_minute': round(gpm, 1),
            'best_eval_win_rate': round(self.best_win_rate * 100, 1),
            'buffer_size': len(self.experience_buffer) if self.experience_buffer else 0,
            'ghost_count': len([f for f in os.listdir(GHOSTS_DIR) if f.endswith('.pt')]) if os.path.exists(GHOSTS_DIR) else 0,
            'v_mean': round(v_mean, 4),
            'v_pre_win': round(v_pre_win, 4),
            'v_pre_loss': round(v_pre_loss, 4),
            'timestamp': time.time(),
        }
        if eval_wr is not None:
            stats['eval_win_rate'] = round(eval_wr * 100, 1)
        
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
                conn = sqlite3.connect(game_db.db_path)
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
                conn.close()
            except Exception:
                pass
        
        try:
            with open(STATS_PATH, 'w') as f:
                json.dump(stats, f, indent=2)
        except Exception:
            pass
    
    def log_metrics(self, win_rate, avg_td_error, epsilon, games_played, eval_win_rate=None):
        """Append training metrics to history file."""
        entry = {
            "games": games_played,
            "updates": self.total_updates,
            "win_rate": round(win_rate, 4),
            "avg_td_error": round(avg_td_error, 6),
            "epsilon": round(epsilon, 4),
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
        """Save full training state (model + optimizer + buffer)."""
        path = path or MAIN_CKPT_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_updates': self.total_updates,
            'total_games': self.total_games,
            'best_win_rate': self.best_win_rate,
            'last_ghost_game': self.last_ghost_game,
            'metrics_history': self.metrics_history,
        }
        
        # Save to temp file first, then rename (atomic)
        tmp_path = path + '.tmp'
        torch.save(save_dict, tmp_path)
        os.replace(tmp_path, path)
        
        if is_best:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'total_games': self.total_games,
                'best_win_rate': self.best_win_rate,
            }, BEST_CKPT_PATH)
        
        # Save experience buffer separately (can be large)
        if self.experience_buffer is not None and len(self.experience_buffer) > 0:
            self.experience_buffer.save()
    
    def load_checkpoint(self, path=None):
        """Load full training state. Returns True if successful."""
        path = path or MAIN_CKPT_PATH
        if not os.path.exists(path):
            return False
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.total_updates = checkpoint.get('total_updates', 0)
            self.total_games = checkpoint.get('total_games', 0)
            self.best_win_rate = checkpoint.get('best_win_rate', 0.0)
            self.last_ghost_game = checkpoint.get('last_ghost_game', 0)
            self.metrics_history = checkpoint.get('metrics_history', [])
            print(f"[Trainer] Loaded checkpoint: {self.total_games} games, {self.total_updates} updates")
            
            # Load experience buffer
            if self.experience_buffer is not None:
                self.experience_buffer.load()
            
            return True
        except Exception as e:
            print(f"[Trainer] Failed to load checkpoint: {e}")
            return False
    
    def load_kickstart(self, kickstart_path):
        """Load weights from kickstart model (model weights only, no optimizer)."""
        if not os.path.exists(kickstart_path):
            print(f"[Trainer] Kickstart not found: {kickstart_path}")
            return False
        try:
            checkpoint = torch.load(kickstart_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[Trainer] Loaded kickstart weights from {kickstart_path}")
            return True
        except Exception as e:
            print(f"[Trainer] Failed to load kickstart: {e}")
            return False
