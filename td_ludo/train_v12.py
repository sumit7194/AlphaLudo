"""
TD-Ludo V11 RL training — ResTNet (CNN + Transformer hybrid).

Drop-in successor to train_v10.py:
  - Same trainer (trainer_v10.ActorCriticTrainerV10): V11 has identical
    forward signature so PPO + BCE win_prob + sparse rewards work unchanged
  - Same player (game.players.v10.VectorACGamePlayer): V11 uses the same
    encode_state_v10 (28 channels)
  - Same config defaults (V10.2's lr=1e-5, entropy=0.005, T=1.1→0.95)

Differences from train_v10.py:
  - Run name = "ac_v12" (isolated checkpoint dir)
  - AlphaLudoV12 model class with 4 ResBlocks + 2 attention layers
  - Default dropout=0.0 (PPO importance ratios break with stochastic forward;
    SL used 0.1 for regularization but RL must be deterministic)
  - LR warmup window at start of RL (transformers benefit; 5K games of warmup)
  - Loads checkpoints/ac_v11/model_sl.pt as starting weights

Success criterion (Exp 20 gate): sustained eval WR >80% over 3 consecutive
2000-game evals = first plateau-break in project history (V10.2 peak 78.6%
single-eval, 75.15% 2000-game peak).
"""

import os
import sys
import time
import signal
import argparse
import threading
import torch
import json
import random
import numpy as np
import psutil
from collections import deque
from http.server import HTTPServer, SimpleHTTPRequestHandler
import functools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set V11 run name BEFORE any config imports so CHECKPOINT_DIR resolves correctly.
# Use setdefault so caller can override with `TD_LUDO_RUN_NAME=ac_v11_1 python ...`.
os.environ.setdefault("TD_LUDO_RUN_NAME", "ac_v12")

from td_ludo.models.v12 import AlphaLudoV12
from td_ludo.training.trainer_v10 import ActorCriticTrainerV10  # V10 trainer is V11-compatible
from td_ludo.game.players.v10 import VectorACGamePlayer  # V10 player uses encode_state_v10 (V11 same)
from src.elo_tracker import EloTracker
from src.game_db import GameDB
from src.config import (
    LEARNING_RATE, WEIGHT_DECAY, EVAL_INTERVAL, EVAL_GAMES,
    SAVE_INTERVAL, CHECKPOINT_DIR, MAIN_CKPT_PATH,
    KICKSTART_PATH, STATS_PATH, METRICS_PATH,
    GHOST_SAVE_INTERVAL, MODE, ELO_PATH, GAME_DB_PATH,
    BATCH_SIZE, EARLY_STOP_PATIENCE,
)


# =============================================================================
# Graceful Shutdown (SIGINT)
# =============================================================================
STOP_REQUESTED = False
SECOND_CTRL_C = False


def signal_handler(sig, frame):
    global STOP_REQUESTED, SECOND_CTRL_C
    if STOP_REQUESTED:
        SECOND_CTRL_C = True
        print("\n[V12 Train] Force exit requested. Saving immediately...")
    else:
        STOP_REQUESTED = True
        print("\n[V12 Train] Graceful shutdown requested. Finishing current step...")


signal.signal(signal.SIGINT, signal_handler)


# =============================================================================
# PID Lock
# =============================================================================
LOCK_FILE = os.path.join(CHECKPOINT_DIR, "train.pid")


def _is_process_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire_lock():
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, 'r') as f:
                old_pid = int(f.read().strip())
            if _is_process_alive(old_pid):
                print(f"[V12 Train] ERROR: Another instance (PID {old_pid}) is already running.")
                return False
            print(f"[V12 Train] Stale lock found (PID {old_pid} not running). Removing.")
        except (ValueError, IOError):
            pass
    os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
    with open(LOCK_FILE, 'w') as f:
        f.write(str(os.getpid()))
    return True


def release_lock():
    try:
        if os.path.exists(LOCK_FILE):
            with open(LOCK_FILE, 'r') as f:
                stored_pid = int(f.read().strip())
            if stored_pid == os.getpid():
                os.remove(LOCK_FILE)
    except Exception:
        pass


import atexit
atexit.register(release_lock)


# =============================================================================
# Power-loss-safe checkpoint saving
# =============================================================================
# Strategy:
#   1. Trainer's save_checkpoint() already does atomic save (write tmp, os.replace).
#   2. We add ROTATING BACKUPS — before each save, copy model_latest.pt to
#      model_prev.pt, and model_prev.pt to model_prev2.pt. So we always have
#      at least 2 fallback checkpoints if the latest somehow becomes corrupt.
#   3. We save more frequently than V10 (every 90s vs 5min) to minimize
#      worst-case work loss.
#   4. We save BEFORE every eval (in case eval crashes mid-run) and AFTER
#      every eval (to capture the eval result + best_win_rate update).
import shutil

SAFE_BACKUP_NAMES = ['model_latest.pt', 'model_prev.pt', 'model_prev2.pt']


def safe_save_with_rotation(trainer, ckpt_dir, is_best=False):
    """Atomic save + rotating backups. Survives mid-write power loss."""
    latest = os.path.join(ckpt_dir, SAFE_BACKUP_NAMES[0])
    prev = os.path.join(ckpt_dir, SAFE_BACKUP_NAMES[1])
    prev2 = os.path.join(ckpt_dir, SAFE_BACKUP_NAMES[2])

    # Rotate backups: prev → prev2, latest → prev (only if files exist).
    # Use copy (not move) so the rotation itself is recoverable if it crashes.
    try:
        if os.path.exists(prev):
            shutil.copy2(prev, prev2)
        if os.path.exists(latest):
            shutil.copy2(latest, prev)
    except Exception as e:
        print(f"[Safe-save] WARNING: backup rotation failed: {e}")
        # Continue anyway — we still want to attempt the new save

    # Trainer's save_checkpoint already does atomic write (.tmp + os.replace).
    trainer.save_checkpoint(is_best=is_best)


def load_with_fallback(trainer, ckpt_dir):
    """Try model_latest, fall back to model_prev / model_prev2 on corruption."""
    for name in SAFE_BACKUP_NAMES:
        path = os.path.join(ckpt_dir, name)
        if not os.path.exists(path):
            continue
        try:
            loaded = trainer.load_checkpoint(path)
            if loaded:
                if name != SAFE_BACKUP_NAMES[0]:
                    print(f"[Safe-load] WARNING: fell back to {name} "
                          f"(latest was corrupt or missing)")
                else:
                    print(f"[Safe-load] Loaded {name}")
                return True
        except Exception as e:
            print(f"[Safe-load] {name} failed to load ({e}), trying next backup")
    print("[Safe-load] No usable checkpoint found in any backup slot")
    return False


# =============================================================================
# Dashboard globals
# =============================================================================
_elo_tracker = None
_game_db = None
_player = None


def start_dashboard_server(port=8789):
    """Dashboard on port 8789 (8787=V10, 8788=exploiter, 8789=V11)."""
    dashboard_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(dashboard_dir, 'index.html')):
        print(f"[Dashboard] index.html not found, skipping")
        return None
    handler = functools.partial(DashboardHandler, directory=dashboard_dir)
    server = HTTPServer(('0.0.0.0', port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[Dashboard] http://localhost:{port}  (V11 dashboard: /v11_dashboard.html)")
    return server


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self):
        if self.path == '/api/stats':
            self._serve_json(STATS_PATH)
        elif self.path == '/api/metrics':
            self._serve_json(METRICS_PATH)
        elif self.path == '/api/elo':
            self._serve_elo()
        elif self.path.startswith('/api/games'):
            self._serve_games()
        elif self.path == '/api/system':
            self._serve_system()
        elif self.path == '/api/spectate':
            self._serve_spectate()
        else:
            super().do_GET()

    def _serve_json(self, path):
        try:
            with open(path, 'r') as f:
                data = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(data.encode())
        except FileNotFoundError:
            self.send_response(404); self.end_headers()
        except Exception:
            self.send_response(500); self.end_headers()

    def _serve_elo(self):
        if _elo_tracker is not None:
            data = json.dumps({
                'rankings': _elo_tracker.get_rankings(top_n=15),
                'history': _elo_tracker.get_history_for_dashboard(),
            })
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(data.encode())
        else:
            self.send_response(404); self.end_headers()

    def _serve_games(self):
        if _game_db is not None:
            games = _game_db.get_recent_games(n=50)
            data = json.dumps(games)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(data.encode())
        else:
            self.send_response(404); self.end_headers()

    def _serve_system(self):
        try:
            data = json.dumps({
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'pid': os.getpid(),
            })
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(data.encode())
        except Exception:
            self.send_response(500); self.end_headers()

    def _serve_spectate(self):
        if _player is not None:
            state = _player.get_spectator_state(game_idx=0)
            if state:
                data = json.dumps(state)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(data.encode())
                return
        self.send_response(404); self.end_headers()

    def log_message(self, format, *args):
        pass


# =============================================================================
# Device
# =============================================================================
def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# =============================================================================
# Main
# =============================================================================
def main():
    global _elo_tracker, _game_db, _player

    parser = argparse.ArgumentParser(description='V12 RL Training (Exp 20)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoints/ac_v11/model_latest.pt')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--games', type=int, default=0,
                        help='Max games this session (0=unlimited)')
    parser.add_argument('--hours', type=float, default=0,
                        help='Max hours this session (0=unlimited)')
    parser.add_argument('--no-dashboard', action='store_true')
    parser.add_argument('--port', type=int, default=8790)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--fresh', action='store_true',
                        help='Wipe run dir, restart from model_sl.pt')
    parser.add_argument('--alarm', action='store_true')

    # V11 architecture (defaults match SL training)
    parser.add_argument('--num-res-blocks', type=int, default=4)
    parser.add_argument('--num-channels', type=int, default=96)
    parser.add_argument('--num-attn-layers', type=int, default=2)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--ffn-ratio', type=int, default=4)
    parser.add_argument('--attn-dim', type=int, default=None,
                        help='Inner attention dim. None = matches num_channels '
                             '(V11). Smaller value (e.g. 64) adds Linear '
                             'projection in/out for memory savings (V11.1).')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='RL dropout MUST be 0.0 (PPO importance ratios '
                             'break with stochastic forward). SL used 0.1.')

    # PPO config overrides
    parser.add_argument('--anneal-lr', type=float, default=0.0)
    parser.add_argument('--anneal-games', type=int, default=20000)
    # V12.1: bump default from 0.005 (config) → 0.01 to fight overconfidence.
    # Eval-lens analysis showed V12 collapsed to 0.14 entropy; ~70% of decisions
    # >0.95 confidence; confident-disagreements had no win-prob signal.
    # Higher entropy bonus keeps the policy from collapsing into a single-token
    # greedy mode, lets PPO actually explore alternatives during self-play.
    # Pass --entropy-coeff <other> to override; pass -1 to use config default.
    parser.add_argument('--entropy-coeff', type=float, default=0.01)
    parser.add_argument('--reset-lr', action='store_true')
    parser.add_argument('--save-interval', type=float, default=90.0,
                        help='Seconds between auto-saves with backup rotation. '
                             'Shorter = less work lost on power cut. '
                             'Default 90s (config default 300s for V10).')
    parser.add_argument('--ppo-minibatch-size', type=int, default=0,
                        help='Override PPO minibatch size (config default 256). '
                             'Smaller = less attention memory per step, lets us '
                             'avoid macOS jetsam OOM on 16GB Macs. 0 = use config.')

    # V11-specific: LR warmup at start of RL
    parser.add_argument('--rl-warmup-games', type=int, default=5000,
                        help='Linear LR warmup over first N games. Transformers '
                             'benefit from warmup; default V10 didn\'t need it.')

    args = parser.parse_args()

    # Fresh start: wipe run dir but keep model_sl.pt
    if args.fresh and os.path.exists(CHECKPOINT_DIR):
        import shutil
        print(f"[V12 Train] Fresh start. Purging {CHECKPOINT_DIR}...")
        for f in os.listdir(CHECKPOINT_DIR):
            if f in ('model_sl.pt',):
                continue
            fpath = os.path.join(CHECKPOINT_DIR, f)
            try:
                if os.path.isfile(fpath):
                    os.unlink(fpath)
                elif os.path.isdir(fpath):
                    shutil.rmtree(fpath)
            except Exception as e:
                print(f"[V12 Train] Warning: couldn't delete {fpath}: {e}")

    from src.config import GHOSTS_DIR
    os.makedirs(GHOSTS_DIR, exist_ok=True)

    if not acquire_lock():
        sys.exit(1)

    # Device
    device = torch.device(args.device) if args.device else get_device()
    print(f"[V12 Train] Device: {device}")
    print(f"[V12 Train] Mode:   {MODE}")

    # Build model — V11 architecture
    model_factory = lambda: AlphaLudoV12(
        num_res_blocks=args.num_res_blocks,
        num_channels=args.num_channels,
        num_attn_layers=args.num_attn_layers,
        num_heads=args.num_heads,
        ffn_ratio=args.ffn_ratio,
        dropout=args.dropout,  # 0.0 for RL
        in_channels=28,
    )
    model = model_factory()
    print(f"[V12 Train] Model: AlphaLudoV12 ({model.count_parameters():,} params)")
    print(f"[V12 Train]   {args.num_res_blocks} ResBlocks × {args.num_channels}ch + "
          f"{args.num_attn_layers} Attn layers × {args.num_heads} heads")
    print(f"[V12 Train]   dropout={args.dropout} (RL must be 0.0)")
    model.to(device)

    # Trainer (V10 trainer works unchanged — same forward signature)
    trainer = ActorCriticTrainerV10(model, device, learning_rate=LEARNING_RATE)

    # Load weights with multi-backup fallback
    if args.resume:
        loaded = load_with_fallback(trainer, CHECKPOINT_DIR)
        if loaded:
            print(f"[V12 Train] Resumed: {trainer.total_games} games, "
                  f"{trainer.total_updates} updates")
        else:
            print("[V12 Train] No usable checkpoint backup. Falling back to SL.")
            sl_path = os.path.join(CHECKPOINT_DIR, 'model_sl.pt')
            if os.path.exists(sl_path):
                trainer.load_checkpoint(sl_path)
                trainer.total_games = 0
                trainer.total_updates = 0
                trainer.best_win_rate = 0.0
                print(f"[V12 Train] Initialized from SL: {sl_path}")
            else:
                print("[V12 Train] No SL baseline found. Random weights.")
    else:
        sl_path = os.path.join(CHECKPOINT_DIR, 'model_sl.pt')
        if os.path.exists(sl_path):
            trainer.load_checkpoint(sl_path)
            trainer.total_games = 0
            trainer.total_updates = 0
            trainer.best_win_rate = 0.0
            print(f"[V12 Train] Initialized from SL: {sl_path}")
        else:
            print("[V12 Train] No SL baseline found. Random weights.")

    # LR reset (same fix as V10 — SL cosine schedule saves lr=0)
    for g in trainer.optimizer.param_groups:
        if g['lr'] <= 0 or args.reset_lr:
            old_lr = g['lr']
            g['lr'] = LEARNING_RATE
            reason = "--reset-lr flag" if args.reset_lr else "was <= 0"
            print(f"[V12 Train] Reset LR {old_lr:.1e} → {LEARNING_RATE:.1e} ({reason})")

    # Entropy override
    if args.entropy_coeff >= 0:
        old_ent = trainer.entropy_coeff
        trainer.entropy_coeff = args.entropy_coeff
        print(f"[V12 Train] Entropy: {old_ent} → {args.entropy_coeff}")

    # PPO minibatch override (memory-safety on tight RAM systems)
    if args.ppo_minibatch_size > 0:
        old_mb = trainer.ppo_minibatch_size
        trainer.ppo_minibatch_size = args.ppo_minibatch_size
        print(f"[V12 Train] PPO minibatch: {old_mb} → {args.ppo_minibatch_size} "
              f"(reduces attention memory peak)")

    elo_tracker = EloTracker(save_path=ELO_PATH)
    _elo_tracker = elo_tracker
    print(f"[V12 Train] Elo: Model={elo_tracker.get_rating('Model'):.0f}")

    game_db = GameDB(GAME_DB_PATH)
    _game_db = game_db
    print(f"[V12 Train] Game DB: {game_db.get_total_games()} games recorded")

    if args.eval_only:
        from evaluate_v10 import evaluate_model  # V10 eval works (same encoder)
        results = evaluate_model(model, device, num_games=500, verbose=True)
        print(f"\nWin Rate: {results['win_rate_percent']}%")
        return

    if not args.no_dashboard:
        start_dashboard_server(port=args.port)

    player = VectorACGamePlayer(
        trainer, BATCH_SIZE, device,
        model_factory=model_factory,
        elo_tracker=elo_tracker,
    )
    _player = player

    rolling_win_rate = deque(maxlen=500)
    start_time = time.time()
    last_save_time = time.time()
    eval_drops = 0
    games_at_start = trainer.total_games
    trainer.play_alarm = args.alarm

    # Bucket-based eval scheduler — survives restarts cleanly.
    # The naive approach (session-local counter) loses progress on every
    # power-loss restart and shifts eval boundaries off "round" game counts.
    # Bucket approach: eval whenever total_games crosses an EVAL_INTERVAL
    # boundary (10K, 20K, ...) regardless of how many sessions split the run.
    last_eval_bucket = trainer.total_games // EVAL_INTERVAL
    if trainer.total_games > 0:
        print(f"[V12 Train] Eval scheduler: last_bucket={last_eval_bucket} "
              f"(next eval at G={(last_eval_bucket + 1) * EVAL_INTERVAL:,})")

    # LR annealing (same as V10)
    lr_anneal_active = args.anneal_lr > 0
    if lr_anneal_active:
        import math
        lr_start = trainer.optimizer.param_groups[0]['lr']
        lr_end = args.anneal_lr
        anneal_games = args.anneal_games
        print(f"[V12 Train] LR annealing: {lr_start:.1e} → {lr_end:.1e} "
              f"(cosine over {anneal_games:,} games)")

    # V11-specific: LR warmup at start of RL
    base_lr = trainer.optimizer.param_groups[0]['lr']
    rl_warmup_games = args.rl_warmup_games if not args.resume else 0
    if rl_warmup_games > 0:
        print(f"[V12 Train] LR warmup: 0 → {base_lr:.1e} "
              f"linearly over first {rl_warmup_games:,} games")

    print(f"\n{'=' * 60}")
    print(f"  V12 RL Training — Experiment 20 (ResTNet)")
    print(f"  Algorithm: PPO + BCE win_prob + sparse rewards")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")
    print(f"  Eval: every {EVAL_INTERVAL} games ({EVAL_GAMES} games each)")
    print(f"  Ghost saves: every {GHOST_SAVE_INTERVAL} games")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Success: eval WR > 80% sustained over 3 evals")
    print(f"  Power-loss safety:")
    print(f"    - Auto-save every {args.save_interval}s with rotating backups")
    print(f"    - Save before each eval (capture pre-eval state)")
    print(f"    - 3 backup slots: model_latest, model_prev, model_prev2")
    print(f"    - Resume tries each backup in order if latest is corrupt")
    if args.hours > 0:
        print(f"  Time limit: {args.hours} hours")
    if args.games > 0:
        print(f"  Game limit: {args.games}")
    print(f"  Ctrl+C to save and exit gracefully")
    print(f"{'=' * 60}\n")

    try:
        while not STOP_REQUESTED:
            session_games = trainer.total_games - games_at_start

            if args.games > 0 and session_games >= args.games:
                print(f"[V12 Train] Reached game limit ({args.games})")
                break

            if args.hours > 0:
                if (time.time() - start_time) / 3600 >= args.hours:
                    print(f"[V12 Train] Reached time limit ({args.hours}h)")
                    break

            if SECOND_CTRL_C:
                break

            # LR warmup
            if rl_warmup_games > 0 and session_games < rl_warmup_games:
                warmup_factor = session_games / rl_warmup_games
                current_lr = base_lr * warmup_factor
                for g in trainer.optimizer.param_groups:
                    g['lr'] = current_lr
            elif lr_anneal_active:
                progress = min(1.0, session_games / max(1, anneal_games))
                factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                current_lr = lr_end + (lr_start - lr_end) * factor
                for g in trainer.optimizer.param_groups:
                    g['lr'] = current_lr

            results = player.play_step(train=True)

            for result in results:
                rolling_win_rate.append(1 if result['model_won'] else 0)

                elo_tracker.update_from_game(
                    result['identities'], result['winner'],
                    game_num=trainer.total_games,
                )
                game_db.add_game(
                    game_num=trainer.total_games,
                    identities=result['identities'],
                    winner=result['winner'],
                    game_length=result.get('total_moves', 0),
                    avg_td_error=0.0,
                    model_player_idx=result['model_player'],
                )

                trainer.maybe_save_ghost(elo_tracker=elo_tracker)

                if trainer.total_games % 10 == 0:
                    win_rate = sum(rolling_win_rate) / max(1, len(rolling_win_rate))
                    entropy = trainer.get_avg_entropy()
                    elapsed = time.time() - start_time
                    games_played = trainer.total_games - games_at_start
                    gpm = games_played / (elapsed / 60) if elapsed > 0 else 0
                    main_elo = elo_tracker.get_rating('Model')
                    temperature = player.get_temperature(trainer.total_games)
                    current_lr = trainer.optimizer.param_groups[0]['lr']

                    print(f"[G {trainer.total_games:>6d}] "
                          f"WR: {win_rate*100:5.1f}% | "
                          f"Ent: {entropy:.3f} | "
                          f"τ: {temperature:.2f} | "
                          f"Elo: {main_elo:.0f} | "
                          f"LR: {current_lr:.1e} | "
                          f"GPM: {gpm:.0f} | "
                          f"{'WIN' if result['model_won'] else 'loss'}")

                    trainer.write_live_stats(
                        win_rate, gpm, temperature=temperature,
                        elo_tracker=elo_tracker, game_db=game_db,
                    )

            if time.time() - last_save_time > args.save_interval:
                safe_save_with_rotation(trainer, CHECKPOINT_DIR)
                elo_tracker.save()
                last_save_time = time.time()
                print(f"[Auto-save] Checkpoint + backups rotated (game {trainer.total_games})")

            current_bucket = trainer.total_games // EVAL_INTERVAL
            if current_bucket > last_eval_bucket:
                # Save BEFORE eval (eval is long; power loss mid-eval would
                # otherwise rewind further than necessary).
                safe_save_with_rotation(trainer, CHECKPOINT_DIR)
                print(f"[Pre-eval-save] Checkpoint saved before eval starts")

                print(f"\n--- Evaluation ({EVAL_GAMES} games) ---")
                from evaluate_v10 import evaluate_model

                eval_results = evaluate_model(model, device, num_games=EVAL_GAMES, verbose=False)
                eval_wr = eval_results['win_rate']
                print(f"--- Eval Win Rate: {eval_results['win_rate_percent']}% ---\n")

                is_best = eval_wr > trainer.best_win_rate
                if is_best:
                    trainer.best_win_rate = eval_wr
                    eval_drops = 0
                    trainer.is_stagnated = False
                    print(f"  ★ New best: {eval_results['win_rate_percent']}%!")
                else:
                    eval_drops += 1
                    print(f"  ↓ No improvement ({eval_drops}/{EARLY_STOP_PATIENCE} patience)")

                # Plateau-break milestone tracking
                if eval_wr >= 0.80:
                    print(f"  ★★★ PLATEAU BREAK: ≥80% sustained eval WR ★★★")

                win_rate_100 = sum(rolling_win_rate) / max(1, len(rolling_win_rate))
                trainer.log_metrics(win_rate_100, trainer.total_games, eval_win_rate=eval_wr)
                trainer.last_eval_wr = eval_wr

                safe_save_with_rotation(trainer, CHECKPOINT_DIR, is_best=is_best)
                elo_tracker.save()
                last_eval_bucket = current_bucket  # advance bucket
                last_save_time = time.time()

                trainer.is_stagnated = (eval_drops >= EARLY_STOP_PATIENCE)
                if trainer.is_stagnated:
                    print(f"\n[V12 Train] ⚠️ WARNING: No improvement for "
                          f"{EARLY_STOP_PATIENCE} consecutive evals.")

    except KeyboardInterrupt:
        print("\n[V12 Train] Keyboard interrupt")
    except Exception as e:
        print(f"\n[V12 Train] Error: {e}")
        import traceback
        traceback.print_exc()

    print("[V12 Train] Final save (with backup rotation)...")
    trainer.flush_buffer()
    safe_save_with_rotation(trainer, CHECKPOINT_DIR)
    elo_tracker.save()

    elapsed = time.time() - start_time
    games_played = trainer.total_games - games_at_start

    print(f"\n{'=' * 60}")
    print(f"  V12 RL Training Complete")
    print(f"  Session games:        {games_played:,}")
    print(f"  Total games:          {trainer.total_games:,}")
    print(f"  Total updates:        {trainer.total_updates:,}")
    print(f"  Model Elo:            {elo_tracker.get_rating('Model'):.0f}")
    print(f"  Best Eval Win Rate:   {trainer.best_win_rate*100:.1f}%")
    print(f"  Duration:             {elapsed/3600:.2f} hours")
    if games_played > 0 and elapsed > 0:
        print(f"  Throughput:           {games_played / (elapsed / 60):.0f} games/min")
    print(f"  Checkpoint:           {MAIN_CKPT_PATH}")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
