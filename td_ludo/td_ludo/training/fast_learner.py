"""
Fast Learner for V9 Multi-Process Training

Consumes trajectories from actor processes, reconstructs context windows,
runs PPO updates on MPS/CUDA, and periodically saves weights for actors.
"""

import os
import time
import json
import queue
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

V9_IN_CHANNELS = 14
V9_EMBED_DIM = 80


class FastLearner:
    """
    PPO learner that runs on MPS/CUDA.
    Receives compact game data, reconstructs full context windows, trains.
    """

    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config

        lr = config.get('learning_rate', 1e-5)
        wd = config.get('weight_decay', 1e-4)

        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.entropy_coeff = config.get('entropy_coeff', 0.005)
        self.value_loss_coeff = config.get('value_loss_coeff', 0.5)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.ppo_epochs = config.get('ppo_epochs', 3)
        self.ppo_buffer_steps = config.get('ppo_buffer_steps', 4096)
        self.ppo_minibatch_size = config.get('ppo_minibatch_size', 256)
        self.context_length = config.get('context_length', 16)

        # Core ML Neural Engine for CNN feature pre-computation
        self._coreml_model = None
        self._coreml_batch_size = 256
        self._coreml_update_counter = 0
        if config.get('use_coreml', False):
            self._init_coreml()

        # PPO buffer (list of ready-to-train step dicts)
        self._ppo_buffer = []
        self._games_buffered = 0

        # Stats (note: total_updates may be set by checkpoint resume)
        self.total_updates = 0
        self.total_games = 0
        self.best_win_rate = 0.0
        self.last_ghost_game = 0
        self.last_eval_wr = None

        self.recent_policy_entropy = deque(maxlen=1000)
        self.recent_value_loss = deque(maxlen=1000)
        self.recent_policy_loss = deque(maxlen=1000)
        self.recent_advantages = deque(maxlen=1000)
        self.recent_clip_fractions = deque(maxlen=1000)
        self.recent_approx_kl = deque(maxlen=1000)

        self.metrics_history = []

    def _init_coreml(self):
        """Try to initialize Core ML model for Neural Engine CNN acceleration."""
        try:
            import coremltools as ct
            self._ct = ct
            self._convert_coreml()
        except ImportError:
            print("[Learner] coremltools not available — using GPU for CNN")
        except Exception as e:
            print(f"[Learner] Core ML init failed: {e} — using GPU for CNN")

    def _convert_coreml(self):
        """Convert current CNN backbone to Core ML for Neural Engine."""
        import coremltools as ct

        import copy

        # Extract backbone as standalone module (deep copy to CPU)
        class _Backbone(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.stem = copy.deepcopy(model.stem)
                self.res_blocks = copy.deepcopy(model.res_blocks)
                self.cnn_proj = copy.deepcopy(model.cnn_proj)
            def forward(self, x):
                out = self.stem(x)
                for block in self.res_blocks:
                    residual = out
                    out = block(out)
                    out = F.relu(out + residual)
                out = F.adaptive_avg_pool2d(out, 1).flatten(1)
                return self.cnn_proj(out)

        backbone = _Backbone(self.model).cpu().eval()
        bs = self._coreml_batch_size
        example = torch.randn(bs, V9_IN_CHANNELS, 15, 15)

        with torch.no_grad():
            traced = torch.jit.trace(backbone, example)

        ml_model = ct.convert(
            traced,
            inputs=[ct.TensorType(name='grid', shape=(bs, V9_IN_CHANNELS, 15, 15))],
            outputs=[ct.TensorType(name='features')],
            compute_precision=ct.precision.FLOAT32,
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS15,
        )
        self._coreml_model = ml_model
        print(f"[Learner] Core ML Neural Engine enabled (batch={bs})")

    def _refresh_coreml(self):
        """Re-convert Core ML model after weight update (every 10 PPO updates)."""
        self._coreml_update_counter += 1
        if self._coreml_model is not None and self._coreml_update_counter % 10 == 0:
            try:
                self._convert_coreml()
            except Exception as e:
                print(f"[Learner] Core ML refresh failed: {e}")

    def _compute_cnn_ane(self, flat_grids_np):
        """
        Compute CNN features using Neural Engine via Core ML.
        flat_grids_np: (total, 14, 15, 15) numpy float32
        Returns: (total, embed_dim) numpy float32
        """
        bs = self._coreml_batch_size
        total = flat_grids_np.shape[0]
        all_features = []

        for i in range(0, total, bs):
            chunk = flat_grids_np[i:i + bs]
            actual_bs = chunk.shape[0]

            if actual_bs < bs:
                # Pad to batch size
                padded = np.zeros((bs, V9_IN_CHANNELS, 15, 15), dtype=np.float32)
                padded[:actual_bs] = chunk
                result = self._coreml_model.predict({'grid': padded})
                all_features.append(result['features'][:actual_bs])
            else:
                result = self._coreml_model.predict({'grid': chunk})
                all_features.append(result['features'])

        return np.concatenate(all_features, axis=0)

    def add_game_data(self, game_data):
        """
        Receive compact game data from actor, reconstruct context windows,
        add to PPO buffer. Returns PPO update stats if buffer is full.
        """
        K = self.context_length
        grids = game_data['player_grids']     # (T, 14, 15, 15)
        prev_actions = game_data['prev_actions']  # (T,)
        step_actions = game_data['step_actions']  # (T,)
        legal_masks = game_data['legal_masks']    # (T, 4)
        old_log_probs = game_data['old_log_probs']  # (T,)
        temperatures = game_data['temperatures']     # (T,)
        returns = game_data['returns']               # (T,)
        T = len(grids)

        for j in range(T):
            # Reconstruct K-length context window
            start = max(0, j - K + 1)
            n_valid = j - start + 1
            n_pad = K - n_valid

            ctx_grids = np.zeros((K, V9_IN_CHANNELS, 15, 15), dtype=np.float32)
            ctx_prev_acts = np.full(K, 4, dtype=np.int64)
            ctx_mask = np.ones(K, dtype=bool)

            ctx_grids[n_pad:] = grids[start:j + 1]
            ctx_prev_acts[n_pad:] = prev_actions[start:j + 1]
            ctx_mask[n_pad:] = False

            self._ppo_buffer.append({
                'grids': ctx_grids,
                'prev_actions': ctx_prev_acts,
                'seq_mask': ctx_mask,
                'action': int(step_actions[j]),
                'legal_mask': legal_masks[j],
                'old_log_prob': float(old_log_probs[j]),
                'temperature': float(temperatures[j]),
                'z': float(returns[j]),
            })

        self._games_buffered += 1
        self.total_games += 1

        if len(self._ppo_buffer) >= self.ppo_buffer_steps:
            return self._ppo_update()
        return None

    def _ppo_update(self):
        """Run PPO update on accumulated buffer. Returns stats dict."""
        if not self._ppo_buffer:
            return {}

        self.model.train()
        n_steps = len(self._ppo_buffer)

        # Stack all data to device
        all_grids = torch.from_numpy(
            np.stack([s['grids'] for s in self._ppo_buffer])
        ).to(self.device, dtype=torch.float32)

        all_acts = torch.from_numpy(
            np.stack([s['prev_actions'] for s in self._ppo_buffer])
        ).to(self.device)

        all_seq_mask = torch.from_numpy(
            np.stack([s['seq_mask'] for s in self._ppo_buffer])
        ).to(self.device)

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

        # Precompute advantages
        with torch.no_grad():
            _, all_values = self.model(all_grids, all_acts, all_seq_mask, all_masks)
            all_values = all_values.squeeze(-1)
            all_advantages = all_returns - all_values
            adv_mean = all_advantages.mean()
            adv_std = all_advantages.std()
            all_advantages = (all_advantages - adv_mean) / (adv_std + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_advantage = 0.0
        total_clip_frac = 0.0
        total_approx_kl = 0.0
        n_minibatches = 0

        for epoch in range(self.ppo_epochs):
            indices = np.random.permutation(n_steps)

            for start in range(0, n_steps, self.ppo_minibatch_size):
                end = min(start + self.ppo_minibatch_size, n_steps)
                mb_idx = indices[start:end]

                mb_grids = all_grids[mb_idx]
                mb_acts = all_acts[mb_idx]
                mb_seq_mask = all_seq_mask[mb_idx]
                mb_actions = all_actions[mb_idx]
                mb_masks = all_masks[mb_idx]
                mb_old_lp = all_old_lp[mb_idx]
                mb_temperatures = all_temperatures[mb_idx]
                mb_returns = all_returns[mb_idx]
                mb_advantages = all_advantages[mb_idx]

                policy, value = self.model(
                    mb_grids, mb_acts, mb_seq_mask, mb_masks
                )
                value = value.squeeze(-1)

                behavior_temps = mb_temperatures.clamp_min(1e-6).unsqueeze(1)
                behavior_logits = torch.log(policy + 1e-8) / behavior_temps
                behavior_policy = F.softmax(behavior_logits, dim=1)
                new_lp = torch.log(
                    behavior_policy.gather(1, mb_actions.unsqueeze(1)).squeeze(1) + 1e-8
                )

                raw_ratio = torch.exp(new_lp - mb_old_lp)
                ratio = torch.clamp(raw_ratio, 0.0, 10.0)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.smooth_l1_loss(value, mb_returns)

                entropy = -(policy * torch.log(policy + 1e-8)).sum(dim=1).mean()

                loss = (
                    policy_loss
                    + self.value_loss_coeff * value_loss
                    - self.entropy_coeff * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.total_updates += 1

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_advantage += mb_advantages.mean().item()

                with torch.no_grad():
                    clipped = (
                        (ratio - 1.0).abs() > self.clip_epsilon
                    ).float().mean().item()
                    approx_kl = (mb_old_lp - new_lp).mean().item()
                total_clip_frac += clipped
                total_approx_kl += abs(approx_kl)
                n_minibatches += 1

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

        self._ppo_buffer.clear()
        self._games_buffered = 0

        self.model.eval()

        return {
            'policy_loss': avg_pl if n_minibatches > 0 else 0,
            'value_loss': avg_vl if n_minibatches > 0 else 0,
            'entropy': avg_ent if n_minibatches > 0 else 0,
            'total_steps': n_steps,
            'ppo_minibatches': n_minibatches,
        }

    def flush_buffer(self):
        if self._ppo_buffer:
            print(f"[Learner] Flushing PPO buffer "
                  f"({self._games_buffered} games, {len(self._ppo_buffer)} steps)")
            self._ppo_update()

    # =========================================================================
    # Stats helpers
    # =========================================================================
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

    # =========================================================================
    # Weight export (for actors)
    # =========================================================================
    def export_weights(self, path):
        """Save clean state dict for actors to load."""
        raw = self.model.state_dict()
        clean = {k.replace('_orig_mod.', ''): v.cpu() for k, v in raw.items()}
        tmp = path + '.tmp'
        torch.save(clean, tmp)
        os.replace(tmp, path)

    # =========================================================================
    # Checkpoints (full training state)
    # =========================================================================
    def save_checkpoint(self, path, is_best=False, best_path=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        raw = self.model.state_dict()
        clean = {k.replace('_orig_mod.', ''): v for k, v in raw.items()}

        save_dict = {
            'model_state_dict': clean,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_updates': self.total_updates,
            'total_games': self.total_games,
            'best_win_rate': self.best_win_rate,
            'last_ghost_game': self.last_ghost_game,
            'metrics_history': self.metrics_history,
            'last_eval_wr': self.last_eval_wr,
        }

        tmp = path + '.tmp'
        torch.save(save_dict, tmp)
        os.replace(tmp, path)

        if is_best and best_path:
            torch.save({
                'model_state_dict': clean,
                'total_games': self.total_games,
                'best_win_rate': self.best_win_rate,
            }, best_path)

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            return False
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

            # Handle context_length mismatch (e.g. checkpoint K=16, model K=8)
            model_K = self.config.get('context_length', 16)
            if 'temporal_pos_embed.weight' in state_dict:
                ckpt_K = state_dict['temporal_pos_embed.weight'].shape[0]
                if ckpt_K != model_K:
                    print(f"[Learner] Adapting pos_embed from K={ckpt_K} to K={model_K}")
                    state_dict['temporal_pos_embed.weight'] = state_dict['temporal_pos_embed.weight'][:model_K]
            if 'causal_mask' in state_dict:
                ckpt_K = state_dict['causal_mask'].shape[0]
                if ckpt_K != model_K:
                    state_dict['causal_mask'] = state_dict['causal_mask'][:model_K, :model_K]

            self.model.load_state_dict(state_dict)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                if 'optimizer_state_dict' in checkpoint:
                    try:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        # Move optimizer state tensors to correct device
                        for state in self.optimizer.state.values():
                            for k, v in state.items():
                                if isinstance(v, torch.Tensor):
                                    state[k] = v.to(self.device)
                    except Exception as e:
                        print(f"[Learner] Warning: Could not load optimizer state: {e}")
                self.total_updates = checkpoint.get('total_updates', 0)
                self.total_games = checkpoint.get('total_games', 0)
                self.best_win_rate = checkpoint.get('best_win_rate', 0.0)
                self.last_ghost_game = checkpoint.get('last_ghost_game', 0)
                self.metrics_history = checkpoint.get('metrics_history', [])
                self.last_eval_wr = checkpoint.get('last_eval_wr', None)
                print(f"[Learner] Loaded checkpoint: {self.total_games} games, "
                      f"{self.total_updates} updates")
            return True
        except Exception as e:
            print(f"[Learner] Failed to load checkpoint: {e}")
            return False

    # =========================================================================
    # Ghost management
    # =========================================================================
    def maybe_save_ghost(self, ghosts_dir, max_ghosts=20):
        ghost_interval = self.config.get('ghost_save_interval', 2000)
        games_since = self.total_games - self.last_ghost_game
        if games_since < ghost_interval:
            return False

        os.makedirs(ghosts_dir, exist_ok=True)
        ghost_path = os.path.join(ghosts_dir, f"ghost_{self.total_games:06d}.pt")
        raw = self.model.state_dict()
        clean = {k.replace('_orig_mod.', ''): v for k, v in raw.items()}
        torch.save({
            'model_state_dict': clean,
            'total_games': self.total_games,
        }, ghost_path)
        self.last_ghost_game = self.total_games

        # Cleanup old ghosts
        ghosts = sorted([
            f for f in os.listdir(ghosts_dir) if f.endswith('.pt')
        ])
        while len(ghosts) > max_ghosts:
            os.remove(os.path.join(ghosts_dir, ghosts[0]))
            ghosts.pop(0)

        return True

    # =========================================================================
    # Stats / metrics files (for dashboard)
    # =========================================================================
    def write_live_stats(self, stats_path, win_rate, gpm, temperature=1.0,
                         elo_tracker=None, game_db=None, ghosts_dir=None):
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
            'ghost_count': len([
                f for f in os.listdir(ghosts_dir) if f.endswith('.pt')
            ]) if ghosts_dir and os.path.exists(ghosts_dir) else 0,
            'is_stagnated': False,
            'timestamp': time.time(),
        }
        # Alpha gate value (transformer contribution)
        import math
        if hasattr(self.model, 'transformer_alpha'):
            raw = self.model.transformer_alpha.item()
            stats['alpha_gate'] = round(math.tanh(raw), 4)
            stats['alpha_gate_raw'] = round(raw, 4)
        if self.last_eval_wr is not None:
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
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
        except Exception:
            pass

    def log_metrics(self, metrics_path, win_rate, eval_win_rate=None):
        entry = {
            'games': self.total_games,
            'updates': self.total_updates,
            'win_rate': round(win_rate, 4),
            'policy_entropy': round(self.get_avg_entropy(), 4),
            'avg_value_loss': round(self.get_avg_value_loss(), 6),
            'avg_policy_loss': round(self.get_avg_policy_loss(), 6),
            'timestamp': time.time(),
        }
        if eval_win_rate is not None:
            entry['eval_win_rate'] = round(eval_win_rate, 4)
        if hasattr(self.model, 'transformer_alpha'):
            import math
            entry['alpha_gate'] = round(math.tanh(self.model.transformer_alpha.item()), 4)
        self.metrics_history.append(entry)
        try:
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
        except Exception:
            pass


def learner_worker(trajectory_queue, stats_queue,
                   weight_path, weight_version, total_games_counter,
                   stop_event,
                   device_str, config, checkpoint_dir, ghosts_dir,
                   resume_path=None, sl_weights_path=None,
                   weight_update_queue=None):
    """
    Learner process entry point.
    Runs on MPS/CUDA, consumes trajectories, does PPO updates.
    """
    import torch
    from src.model_v9 import AlphaLudoV9

    device = torch.device(device_str)
    context_length = config.get('context_length', 16)

    model = AlphaLudoV9(context_length=context_length)

    learner = FastLearner(model, device, config)

    # Load weights
    if resume_path and os.path.exists(resume_path):
        learner.load_checkpoint(resume_path)
        print(f"[Learner] Resumed from {resume_path}")
    elif sl_weights_path and os.path.exists(sl_weights_path):
        checkpoint = torch.load(sl_weights_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        # Handle context_length mismatch
        if 'temporal_pos_embed.weight' in state_dict:
            ckpt_K = state_dict['temporal_pos_embed.weight'].shape[0]
            if ckpt_K != context_length:
                state_dict['temporal_pos_embed.weight'] = state_dict['temporal_pos_embed.weight'][:context_length]
        if 'causal_mask' in state_dict:
            ckpt_K = state_dict['causal_mask'].shape[0]
            if ckpt_K != context_length:
                state_dict['causal_mask'] = state_dict['causal_mask'][:context_length, :context_length]
        model.load_state_dict(state_dict)
        print(f"[Learner] Loaded SL weights from {sl_weights_path}")
    else:
        print("[Learner] Starting from random initialization")

    model.to(device)
    model.eval()

    # Export initial weights for actors
    learner.export_weights(weight_path)
    weight_version.value += 1

    # Paths
    main_ckpt = os.path.join(checkpoint_dir, 'model_latest.pt')
    best_ckpt = os.path.join(checkpoint_dir, 'model_best.pt')
    stats_path = os.path.join(checkpoint_dir, 'live_stats.json')
    metrics_path = os.path.join(checkpoint_dir, 'training_metrics.json')

    # Elo & Game DB (optional, created here since they're not process-safe)
    elo_tracker = None
    game_db = None
    try:
        from src.elo_tracker import EloTracker
        from src.game_db import GameDB
        elo_path = os.path.join(checkpoint_dir, 'elo_ratings.json')
        db_path = os.path.join(checkpoint_dir, 'game_history.db')
        elo_tracker = EloTracker(save_path=elo_path)
        game_db = GameDB(db_path)
    except Exception as e:
        print(f"[Learner] Warning: could not init elo/gamedb: {e}")

    # Rolling stats
    rolling_wins = deque(maxlen=500)
    start_time = time.time()
    last_save_time = time.time()
    last_weight_export = time.time()
    games_since_eval = 0
    eval_drops = 0
    games_at_start = learner.total_games

    eval_interval = config.get('eval_interval', 2000)
    eval_games = config.get('eval_games', 500)
    save_interval = config.get('save_interval', 300)
    weight_export_interval = config.get('weight_export_interval', 30)
    ghost_save_interval = config.get('ghost_save_interval', 2000)
    max_ghosts = config.get('max_ghosts', 20)
    early_stop_patience = config.get('early_stop_patience', 100)

    print(f"[Learner] Started on {device} "
          f"(buffer={config.get('ppo_buffer_steps', 4096)} steps, "
          f"minibatch={config.get('ppo_minibatch_size', 256)})")

    while not stop_event.is_set():
        # Batch-drain trajectories from queue for efficiency
        batch = []
        try:
            batch.append(trajectory_queue.get(timeout=1.0))
        except queue.Empty:
            continue
        # Grab more if available (non-blocking)
        while len(batch) < 200:
            try:
                batch.append(trajectory_queue.get_nowait())
            except queue.Empty:
                break

        update_result = None
        for game_data in batch:
            game_info = game_data['game_info']
            model_won = game_info['model_won']
            rolling_wins.append(1 if model_won else 0)
            games_since_eval += 1

            # Elo update
            if elo_tracker is not None:
                elo_tracker.update_from_game(
                    game_info['identities'],
                    game_info['winner'],
                    game_num=learner.total_games,
                )

            # Game DB
            if game_db is not None:
                game_db.add_game(
                    game_num=learner.total_games,
                    identities=game_info['identities'],
                    winner=game_info['winner'],
                    game_length=game_info.get('total_moves', 0),
                    avg_td_error=0.0,
                    model_player_idx=game_info['model_player'],
                )

            # Add to PPO buffer (may trigger update)
            result = learner.add_game_data(game_data)
            if result is not None:
                update_result = result
        total_games_counter.value = learner.total_games

        if update_result is not None:
            # PPO update happened — export weights for actors
            now = time.time()
            learner.export_weights(weight_path)
            weight_version.value += 1
            last_weight_export = now

            # Notify inference server of new weights
            if weight_update_queue is not None:
                try:
                    weight_update_queue.put_nowait(weight_path)
                except Exception:
                    pass

            # Print update
            win_rate = sum(rolling_wins) / max(1, len(rolling_wins))
            elapsed = now - start_time
            games_played = learner.total_games - games_at_start
            gpm = games_played / (elapsed / 60) if elapsed > 0 else 0

            entropy = learner.get_avg_entropy()
            main_elo = elo_tracker.get_rating('Model') if elo_tracker else 0

            print(f"[PPO {learner.total_updates:>5d}] "
                  f"G: {learner.total_games:>6d} | "
                  f"WR: {win_rate*100:5.1f}% | "
                  f"Ent: {entropy:.3f} | "
                  f"Elo: {main_elo:.0f} | "
                  f"GPM: {gpm:.0f} | "
                  f"pl: {update_result['policy_loss']:.4f} | "
                  f"vl: {update_result['value_loss']:.4f} | "
                  f"steps: {update_result['total_steps']}")

            learner.write_live_stats(
                stats_path, win_rate, gpm,
                elo_tracker=elo_tracker, game_db=game_db,
                ghosts_dir=ghosts_dir,
            )

        # Ghost saves
        learner.maybe_save_ghost(ghosts_dir, max_ghosts)

        # Periodic checkpoint
        now = time.time()
        if now - last_save_time > save_interval:
            learner.save_checkpoint(main_ckpt)
            if elo_tracker:
                elo_tracker.save()
            last_save_time = now
            print(f"[Learner] Checkpoint saved (game {learner.total_games})")

        # Evaluation
        if games_since_eval >= eval_interval:
            print(f"\n--- Evaluation ({eval_games} games) ---")
            try:
                from evaluate_v9 import evaluate_v9_model
                eval_results = evaluate_v9_model(
                    model, device, num_games=eval_games, verbose=False,
                    context_length=context_length,
                )
                eval_wr = eval_results['win_rate']
                print(f"--- Eval Win Rate: {eval_results['win_rate_percent']}% ---\n")

                is_best = eval_wr > learner.best_win_rate
                if is_best:
                    learner.best_win_rate = eval_wr
                    eval_drops = 0
                    print(f"  * New best: {eval_results['win_rate_percent']}%!")
                else:
                    eval_drops += 1
                    print(f"  No improvement ({eval_drops}/{early_stop_patience} patience)")

                learner.last_eval_wr = eval_wr
                learner.log_metrics(
                    metrics_path,
                    sum(rolling_wins) / max(1, len(rolling_wins)),
                    eval_win_rate=eval_wr,
                )
                learner.save_checkpoint(main_ckpt, is_best=is_best, best_path=best_ckpt)
                if elo_tracker:
                    elo_tracker.save()
            except Exception as e:
                print(f"[Learner] Eval failed: {e}")

            games_since_eval = 0
            last_save_time = time.time()

    # Final save
    print("[Learner] Shutting down...")
    learner.flush_buffer()
    learner.save_checkpoint(main_ckpt)
    learner.export_weights(weight_path)
    if elo_tracker:
        elo_tracker.save()

    elapsed = time.time() - start_time
    games_played = learner.total_games - games_at_start
    print(f"[Learner] Done: {games_played} games, {learner.total_updates} updates, "
          f"{elapsed/3600:.2f}h")

    # Send final elo rankings
    if elo_tracker:
        try:
            stats_queue.put({
                'type': 'final_elo',
                'rankings': elo_tracker.get_rankings(top_n=15),
                'total_games': learner.total_games,
                'total_updates': learner.total_updates,
                'best_win_rate': learner.best_win_rate,
            }, timeout=5.0)
        except Exception:
            pass
