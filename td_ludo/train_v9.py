"""
TD-Ludo V9 Training Entry Point — Slim CNN + Temporal Transformer

Architecture: 14ch input, 5 ResBlocks (80ch), 4 Transformer layers (80-dim)
All ~912K params trainable end-to-end (no freeze/unfreeze needed).

Training loop with:
- AlphaLudoV9 model
- PPO (Proximal Policy Optimization) with dense v1 rewards
- Batched execution using VectorGameState + VectorV9GamePlayer
- Ghost checkpoint saving at regular intervals
- Elo rating tracking, SQLite game DB, live stats JSON
- Dashboard HTTP server
- Graceful shutdown (Ctrl+C saves everything)
- Resume support (--resume)
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

os.environ.setdefault('TD_LUDO_RUN_NAME', 'ac_v9_slim_transformer')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model_v9 import AlphaLudoV9
from src.trainer_v9 import V9Trainer
from src.game_player_v9 import VectorV9GamePlayer
from src.elo_tracker import EloTracker
from src.game_db import GameDB
from src.config import (
    LEARNING_RATE, WEIGHT_DECAY, EVAL_INTERVAL, EVAL_GAMES,
    SAVE_INTERVAL, CHECKPOINT_DIR, MAIN_CKPT_PATH,
    STATS_PATH, METRICS_PATH,
    GHOST_SAVE_INTERVAL, MODE, ELO_PATH, GAME_DB_PATH,
    BATCH_SIZE, EARLY_STOP_PATIENCE,
)

# SL checkpoint path (from train_sl_v9.py)
SL_CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'ac_v9')
SL_MODEL_PATH = os.path.join(SL_CHECKPOINT_DIR, 'model_sl_v9.pt')
SL_BEST_PATH = os.path.join(SL_CHECKPOINT_DIR, 'model_sl_v9_best.pt')

# =============================================================================
# Graceful Shutdown
# =============================================================================
STOP_REQUESTED = False
SECOND_CTRL_C = False

def signal_handler(sig, frame):
    global STOP_REQUESTED, SECOND_CTRL_C
    if STOP_REQUESTED:
        SECOND_CTRL_C = True
        print("\n[Train] Force exit requested. Saving immediately...")
    else:
        STOP_REQUESTED = True
        print("\n[Train] Graceful shutdown requested. Will finish current step and save...")

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
                print(f"[Train] ERROR: Another training instance (PID {old_pid}) is already running.")
                print(f"[Train] If this is wrong, delete {LOCK_FILE}")
                return False
            else:
                print(f"[Train] Stale lock found (PID {old_pid} not running). Removing...")
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
# Global references for API access
# =============================================================================
_elo_tracker = None
_game_db = None
_player = None

# =============================================================================
# Dashboard HTTP Server
# =============================================================================
def start_dashboard_server(port=8787):
    dashboard_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(dashboard_dir, 'index.html')):
        print(f"[Dashboard] Warning: index.html not found, skipping dashboard server")
        return None
    handler = functools.partial(DashboardHandler, directory=dashboard_dir)
    try:
        server = HTTPServer(('0.0.0.0', port), handler)
    except OSError as e:
        print(f"[Dashboard] Warning: Could not bind to port {port} ({e}). Dashboard disabled.")
        return None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[Dashboard] Server started at http://localhost:{port}")
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
            self.send_response(404)
            self.end_headers()
        except Exception:
            self.send_response(500)
            self.end_headers()

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
            self.send_response(404)
            self.end_headers()

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
            self.send_response(404)
            self.end_headers()

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
            self.send_response(500)
            self.end_headers()

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
        self.send_response(404)
        self.end_headers()

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
# Main Training Loop
# =============================================================================
def main():
    global _elo_tracker, _game_db, _player

    parser = argparse.ArgumentParser(description='TD-Ludo V9 Slim CNN+Transformer Training')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest V9 checkpoint')
    parser.add_argument('--device', type=str, default=None, help='Device override')
    parser.add_argument('--games', type=int, default=0, help='Max games (0=unlimited)')
    parser.add_argument('--hours', type=float, default=0, help='Max hours (0=unlimited)')
    parser.add_argument('--no-dashboard', action='store_true', help='Disable dashboard')
    parser.add_argument('--port', type=int, default=8787, help='Dashboard port')
    parser.add_argument('--eval-only', action='store_true', help='Run evaluation only')
    parser.add_argument('--fresh', action='store_true', help='Start fresh (delete run data)')
    parser.add_argument('--alarm', action='store_true', help='Audio alarm on stagnation')
    parser.add_argument('--context-length', type=int, default=16, help='Context window size')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--sl-weights', type=str, default=None,
                        help='Path to SL pre-trained weights (auto-detects if not specified)')
    parser.add_argument('--ppo-minibatch', type=int, default=0,
                        help='PPO minibatch size override')
    args = parser.parse_args()

    # Handle fresh start
    if args.fresh and os.path.exists(CHECKPOINT_DIR):
        import shutil
        print(f"[Train] Fresh start. Purging {CHECKPOINT_DIR}...")
        for f in os.listdir(CHECKPOINT_DIR):
            fpath = os.path.join(CHECKPOINT_DIR, f)
            try:
                if os.path.isfile(fpath):
                    os.unlink(fpath)
                elif os.path.isdir(fpath):
                    shutil.rmtree(fpath)
            except Exception as e:
                print(f"[Train] Warning: could not delete {fpath}: {e}")

    from src.config import GHOSTS_DIR
    os.makedirs(GHOSTS_DIR, exist_ok=True)

    if not acquire_lock():
        sys.exit(1)

    # Device
    device = torch.device(args.device) if args.device else get_device()
    print(f"[Train] Device: {device}")
    print(f"[Train] Mode: {MODE}")

    # Initialize model
    context_length = args.context_length
    model_factory = lambda: AlphaLudoV9(context_length=context_length)
    model = model_factory()

    total = model.count_all_parameters()
    print(f"[Train] Architecture: AlphaLudoV9 (14ch, 5res, 80ch, 4TF)")
    print(f"[Train] Total trainable params: {total:,}")

    # Load SL pre-trained weights if available and not resuming
    if not args.resume:
        sl_path = args.sl_weights
        if sl_path is None:
            # Auto-detect SL checkpoint
            for candidate in [SL_BEST_PATH, SL_MODEL_PATH]:
                if os.path.exists(candidate):
                    sl_path = candidate
                    break

        if sl_path and os.path.exists(sl_path):
            print(f"[Train] Loading SL pre-trained weights from {sl_path}")
            checkpoint = torch.load(sl_path, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            print(f"[Train] SL weights loaded successfully")
        else:
            print(f"[Train] No SL weights found. Starting from random initialization.")

    model.to(device)

    # JIT compile
    try:
        model = torch.compile(model, backend='aot_eager')
        print(f"[Train] torch.compile enabled (aot_eager)")
    except Exception as e:
        print(f"[Train] torch.compile unavailable ({e})")

    # Initialize trainer
    trainer = V9Trainer(model, device, learning_rate=LEARNING_RATE)

    if args.ppo_minibatch > 0:
        trainer.ppo_minibatch_size = args.ppo_minibatch
        print(f"[Train] PPO minibatch size override: {args.ppo_minibatch}")

    # Load checkpoint or start fresh
    if args.resume:
        loaded = trainer.load_checkpoint()
        if loaded:
            print(f"[Train] Resumed ({trainer.total_games} games, {trainer.total_updates} updates)")
        else:
            print("[Train] No V9 checkpoint found. Starting fresh.")

    # Elo Tracker
    elo_tracker = EloTracker(save_path=ELO_PATH)
    _elo_tracker = elo_tracker
    print(f"[Train] Elo Tracker (Model: {elo_tracker.get_rating('Model'):.0f})")

    # Game DB
    game_db = GameDB(GAME_DB_PATH)
    _game_db = game_db
    print(f"[Train] Game DB ({game_db.get_total_games()} games)")

    # Eval-only
    if args.eval_only:
        from evaluate_v9 import evaluate_v9_model
        results = evaluate_v9_model(model, device, num_games=500, verbose=True,
                                    context_length=context_length)
        print(f"\nWin Rate: {results['win_rate_percent']}%")
        return

    # Dashboard
    if not args.no_dashboard:
        start_dashboard_server(port=args.port)

    # Game player
    batch_size = args.batch_size
    player = VectorV9GamePlayer(
        trainer, batch_size, device,
        context_length=context_length,
        model_factory=model_factory,
        elo_tracker=elo_tracker,
    )
    _player = player

    # Stats
    rolling_win_rate = deque(maxlen=500)
    start_time = time.time()
    last_save_time = time.time()
    games_since_eval = 0
    eval_drops = 0
    games_at_start = trainer.total_games
    trainer.play_alarm = args.alarm

    print(f"\n{'='*60}")
    print(f"  TD-Ludo V9 Slim CNN+Transformer Training — {MODE} Mode")
    print(f"  Algorithm: PPO | All params trainable end-to-end")
    print(f"  Context: K={context_length} turns")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")
    print(f"  Eval: every {EVAL_INTERVAL} games ({EVAL_GAMES} games each)")
    print(f"  Ghost saves: every {GHOST_SAVE_INTERVAL} games")
    print(f"  Batch Size: {batch_size}")
    if args.hours > 0:
        print(f"  Time limit: {args.hours} hours")
    if args.games > 0:
        print(f"  Game limit: {args.games}")
    print(f"  Ctrl+C to save and exit gracefully")
    print(f"{'='*60}\n")

    try:
        while not STOP_REQUESTED:
            session_games = trainer.total_games - games_at_start
            if args.games > 0 and session_games >= args.games:
                print(f"[Train] Reached game limit ({args.games})")
                break

            if args.hours > 0:
                elapsed_hours = (time.time() - start_time) / 3600
                if elapsed_hours >= args.hours:
                    print(f"[Train] Reached time limit ({args.hours}h)")
                    break

            if SECOND_CTRL_C:
                break

            results = player.play_step(train=True)

            for result in results:
                games_since_eval += 1
                rolling_win_rate.append(1 if result['model_won'] else 0)

                elo_tracker.update_from_game(
                    result['identities'], result['winner'],
                    game_num=trainer.total_games
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

                    print(f"[G {trainer.total_games:>6d}] "
                          f"WR: {win_rate*100:5.1f}% | "
                          f"Ent: {entropy:.3f} | "
                          f"τ: {temperature:.2f} | "
                          f"Elo: {main_elo:.0f} | "
                          f"GPM: {gpm:.0f} | "
                          f"{'WIN' if result['model_won'] else 'loss'}")

                    trainer.write_live_stats(
                        win_rate, gpm,
                        temperature=temperature,
                        elo_tracker=elo_tracker, game_db=game_db
                    )

            if time.time() - last_save_time > SAVE_INTERVAL:
                trainer.save_checkpoint()
                elo_tracker.save()
                last_save_time = time.time()
                print(f"[Auto-save] Checkpoint saved (game {trainer.total_games})")

            if games_since_eval >= EVAL_INTERVAL:
                print(f"\n--- Evaluation ({EVAL_GAMES} games) ---")
                from evaluate_v9 import evaluate_v9_model

                eval_results = evaluate_v9_model(
                    model, device, num_games=EVAL_GAMES, verbose=False,
                    context_length=context_length
                )
                eval_wr = eval_results['win_rate']
                print(f"--- Eval Win Rate: {eval_results['win_rate_percent']}% ---\n")

                is_best = eval_wr > trainer.best_win_rate
                if is_best:
                    trainer.best_win_rate = eval_wr
                    eval_drops = 0
                    trainer.is_stagnated = False
                    print(f"  * New best: {eval_results['win_rate_percent']}%!")
                else:
                    eval_drops += 1
                    print(f"  No improvement ({eval_drops}/{EARLY_STOP_PATIENCE} patience)")

                win_rate_100 = sum(rolling_win_rate) / max(1, len(rolling_win_rate))
                trainer.log_metrics(win_rate_100, trainer.total_games, eval_win_rate=eval_wr)
                trainer.last_eval_wr = eval_wr

                trainer.save_checkpoint(is_best=is_best)
                elo_tracker.save()
                games_since_eval = 0
                last_save_time = time.time()

                trainer.is_stagnated = (eval_drops >= EARLY_STOP_PATIENCE)
                if trainer.is_stagnated:
                    print(f"\n[Train] WARNING: Stagnated for {EARLY_STOP_PATIENCE} evals.")

    except KeyboardInterrupt:
        print("\n[Train] Keyboard interrupt")
    except Exception as e:
        print(f"\n[Train] Error: {e}")
        import traceback
        traceback.print_exc()

    # Final save
    print("[Train] Final save...")
    trainer.flush_buffer()
    trainer.save_checkpoint()
    elo_tracker.save()

    elapsed = time.time() - start_time
    games_played = trainer.total_games - games_at_start

    print(f"\n{'='*60}")
    print(f"  V9 Training Complete ({MODE})")
    print(f"  Games This Session: {games_played}")
    print(f"  Total Games: {trainer.total_games}")
    print(f"  Total Updates: {trainer.total_updates}")
    print(f"  Model Elo: {elo_tracker.get_rating('Model'):.0f}")
    print(f"  Best Eval Win Rate: {trainer.best_win_rate*100:.1f}%")
    print(f"  Duration: {elapsed/3600:.2f} hours")
    if games_played > 0 and elapsed > 0:
        print(f"  Throughput: {games_played / (elapsed / 60):.0f} games/min")
    print(f"  DB Games Recorded: {game_db.get_total_games()}")
    print(f"  Checkpoint: {MAIN_CKPT_PATH}")
    print(f"{'='*60}")

    print(f"\n{elo_tracker}")


if __name__ == '__main__':
    main()
