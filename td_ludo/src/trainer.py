"""
TD-Ludo Trainer — Online TD(0) Value Learning + Experience Replay

Features:
- Online TD(0) semi-gradient updates after each move
- Optional experience buffer for replay training (stabilizes learning)
- Ghost checkpoint saving at regular intervals
- Metrics logging for dashboard consumption
- Graceful checkpoint save/load with optimizer state
"""

import os
import time
import random
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
)


class ExperienceBuffer:
    """
    Simple ring buffer for storing (state, next_state, reward, done) transitions.
    
    Each experience is used once for the online TD update, then optionally
    stored here for replay. This gives us the best of both worlds:
    - Online learning for immediate adaptation
    - Replay for stability and sample efficiency
    """
    
    def __init__(self, max_size=BUFFER_SIZE):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state_tensor, next_state_tensor, reward, done):
        """Add a transition to the buffer."""
        self.buffer.append((
            state_tensor.cpu(),
            next_state_tensor.cpu(),
            reward,
            done,
        ))
    
    def sample(self, batch_size):
        """Sample a random batch of transitions."""
        batch_size = min(batch_size, len(self.buffer))
        indices = random.sample(range(len(self.buffer)), batch_size)
        
        states, next_states, rewards, dones = [], [], [], []
        for idx in indices:
            s, ns, r, d = self.buffer[idx]
            states.append(s)
            next_states.append(ns)
            rewards.append(r)
            dones.append(d)
        
        return (
            torch.stack(states),
            torch.stack(next_states),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )
    
    def __len__(self):
        return len(self.buffer)
    
    def save(self, path=BUFFER_PATH):
        """Save buffer to disk."""
        try:
            # Only save last N entries to keep file size reasonable
            save_data = list(self.buffer)[-self.max_size:]
            torch.save(save_data, path)
        except Exception as e:
            print(f"[Buffer] Failed to save: {e}")
    
    def load(self, path=BUFFER_PATH):
        """Load buffer from disk."""
        if not os.path.exists(path):
            return False
        try:
            data = torch.load(path, weights_only=False)
            self.buffer = deque(data, maxlen=self.max_size)
            print(f"[Buffer] Loaded {len(self.buffer)} transitions")
            return True
        except Exception as e:
            print(f"[Buffer] Failed to load: {e}")
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
        
        # Experience buffer
        self.experience_buffer = ExperienceBuffer() if USE_EXPERIENCE_BUFFER else None
        
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
                batch = torch.stack(state_tensors).to(self.device, dtype=torch.float32)
            else:
                batch = state_tensors.to(self.device, dtype=torch.float32)
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
        """
        Batched TD(0) update.
        """
        self.model.train()
        
        # Ensure tensors are on device
        states = states.to(self.device, dtype=torch.float32)
        next_states = next_states.to(self.device, dtype=torch.float32)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        
        # V(s)
        _, v_s, _ = self.model(states)
        v_s = v_s.squeeze(-1)
        
        # V(s')
        with torch.no_grad():
            _, v_next, _ = self.model(next_states)
            v_next = v_next.squeeze(-1)
            # Mask terminal states
            v_next = v_next * (1.0 - dones)
            
        # Target
        target = rewards + self.gamma * v_next
        
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
            # We can't add batch to deque directly if add() expects single.
            # But we can iterate.
            # Optimized: buffer.extend? No, buffer stores tuples.
            s_cpu = states.detach().cpu()
            ns_cpu = next_states.detach().cpu()
            r_cpu = rewards.detach().cpu()
            d_cpu = dones.detach().cpu()
            for i in range(len(states)):
                self.experience_buffer.add(s_cpu[i], ns_cpu[i], r_cpu[i], d_cpu[i])

        return loss.item()
    
    def replay_train(self):
        """
        Train on a batch of replayed experiences from the buffer.
        This happens periodically and gives each experience multiple uses.
        """
        if self.experience_buffer is None or len(self.experience_buffer) < REPLAY_BATCH_SIZE:
            return 0.0
        
        self.model.train()
        total_loss = 0.0
        
        for _ in range(REPLAY_STEPS):
            states, next_states, rewards, dones = self.experience_buffer.sample(REPLAY_BATCH_SIZE)
            states = states.to(self.device, dtype=torch.float32)
            next_states = next_states.to(self.device, dtype=torch.float32)
            rewards = rewards.to(self.device)
            dones = dones.to(self.device)
            
            # Batch V(s)
            _, v_s, _ = self.model(states)
            v_s = v_s.squeeze(-1)
            
            # Batch V(s') — detached
            with torch.no_grad():
                _, v_next, _ = self.model(next_states)
                v_next = v_next.squeeze(-1)
            
            # TD targets
            targets = rewards + (1.0 - dones) * self.gamma * v_next
            
            # MSE loss
            loss = F.mse_loss(v_s, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
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
            'timestamp': time.time(),
        }
        if eval_wr is not None:
            stats['eval_win_rate'] = round(eval_wr * 100, 1)
        
        # Enrich with Elo data
        if elo_tracker is not None:
            stats['main_elo'] = round(elo_tracker.get_rating('Main'), 1)
            stats['elo_rankings'] = [
                {'name': n, 'elo': round(e, 1)} 
                for n, e in elo_tracker.get_rankings(top_n=15)
            ]
        
        # Enrich with opponent stats from DB
        if game_db is not None:
            stats['opponent_stats'] = game_db.get_opponent_stats()
            stats['db_total'] = game_db.get_total_games()
        
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
