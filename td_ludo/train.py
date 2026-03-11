"""
TD-Ludo Training Entry Point — Actor-Critic Edition

Training loop with:
- Batched execution using VectorGameState + VectorACGamePlayer
- Actor-Critic (REINFORCE + baseline) with Monte Carlo returns
- Trajectory collection for ALL players (model + bots)
- Ghost checkpoint saving at regular intervals
- Elo rating tracking for all agents
- SQLite game history database
- Live stats JSON for dashboard consumption
- Built-in HTTP server for dashboard UI
- Graceful shutdown (Ctrl+C finishes current batch step, saves everything)
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

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import AlphaLudoV5
from src.trainer import ActorCriticTrainer
from src.game_player import VectorACGamePlayer
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
# PID Lock — Prevent multiple training instances
# =============================================================================
LOCK_FILE = os.path.join(CHECKPOINT_DIR, "train.pid")

def _is_process_alive(pid):
    """Check if a process with given PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def acquire_lock():
    """Try to acquire the training lock."""
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
    """Remove the PID lock file."""
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
# Global references for API access (set in main())
# =============================================================================
_elo_tracker = None
_game_db = None
_player = None


# =============================================================================
# Dashboard HTTP Server (serves static files + live stats + Elo + games API)
# =============================================================================
def start_dashboard_server(port=8787):
    """Start HTTP server in background thread to serve dashboard."""
    # Dashboard index.html is in the td_ludo root directory
    dashboard_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(dashboard_dir, 'index.html')):
        print(f"[Dashboard] Warning: index.html not found in {dashboard_dir}, skipping dashboard server")
        return None
    handler = functools.partial(DashboardHandler, directory=dashboard_dir)
    server = HTTPServer(('0.0.0.0', port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[Dashboard] Server started at http://localhost:{port}")
    return server


class DashboardHandler(SimpleHTTPRequestHandler):
    """Custom handler that serves JSON APIs + static files."""
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
        pass  # Suppress HTTP logs


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
    
    parser = argparse.ArgumentParser(description='TD-Ludo Actor-Critic Training')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint')
    parser.add_argument('--device', type=str, default=None,
                        help='Device override (cpu/mps/cuda)')
    parser.add_argument('--games', type=int, default=0,
                        help='Max games to play (0=unlimited)')
    parser.add_argument('--hours', type=float, default=0,
                        help='Max hours to train (0=unlimited)')
    parser.add_argument('--no-dashboard', action='store_true',
                        help='Disable dashboard HTTP server')
    parser.add_argument('--port', type=int, default=8787,
                        help='Dashboard server port (default: 8787)')
    parser.add_argument('--eval-only', action='store_true',
                        help='Run evaluation only')
    parser.add_argument('--fresh', action='store_true',
                        help='Start fresh (delete existing run data)')
    parser.add_argument('--alarm', action='store_true',
                        help='Play audio alarm on dashboard when stagnated')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile to fuse operations (Mac MPS experimental)')
    args = parser.parse_args()
    
    # Handle fresh start (purge run directory)
    if args.fresh and os.path.exists(CHECKPOINT_DIR):
        import shutil
        print(f"[Train] Fresh start requested. Purging {CHECKPOINT_DIR}...")
        for f in os.listdir(CHECKPOINT_DIR):
            if f in ('model_sl.pt', 'checkpoint_sl.pt'):
                continue
            fpath = os.path.join(CHECKPOINT_DIR, f)
            try:
                if os.path.isfile(fpath):
                    os.unlink(fpath)
                elif os.path.isdir(fpath):
                    shutil.rmtree(fpath)
            except Exception as e:
                print(f"[Train] Warning: could not delete {fpath}: {e}")
    
    # Ensure required subdirectories exist
    from src.config import GHOSTS_DIR
    os.makedirs(GHOSTS_DIR, exist_ok=True)
    
    # Acquire training lock
    if not acquire_lock():
        sys.exit(1)
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    print(f"[Train] Device: {device}")
    print(f"[Train] Mode: {MODE}")
    
    # Initialize model — AlphaLudoV5 "V6 Big Brain" (10 blocks × 128 channels)
    model = AlphaLudoV5(num_res_blocks=10, num_channels=128)
    print(f"[Train] Architecture: AlphaLudoV6-Big (128ch, 10res, {model.count_parameters():,} params)")
    model.to(device)
    
    # Initialize trainer
    trainer = ActorCriticTrainer(model, device, learning_rate=LEARNING_RATE)
    
    # Load weights
    if args.resume:
        loaded = trainer.load_checkpoint()
        if loaded:
            print(f"[Train] Resumed from checkpoint ({trainer.total_games} games, {trainer.total_updates} updates)")
        else:
            print("[Train] No 'model_latest.pt' checkpoint found. Starting from scratch.")
    else:
        # User starting a fresh run. Kickstart from SL model!
        sl_path = os.path.join(CHECKPOINT_DIR, 'model_sl.pt')
        if os.path.exists(sl_path):
            trainer.load_checkpoint(sl_path)
            # trainer.load_checkpoint automatically zeroes the counters if it's a raw state dict, 
            # but we explicitly zero them here just to be safe so the training loop restarts correctly.
            trainer.total_games = 0
            trainer.total_updates = 0
            trainer.best_win_rate = 0.0
            print(f"[Train] Picked up SL baseline weights ({sl_path}) for the fresh RL run.")
        else:
            print("[Train] No SL base model found. Starting with completely random weights.")
            
    if args.compile:
        print("[Train] Attempting to torch.compile() model for fused MPS execution...")
        try:
            trainer.model = torch.compile(trainer.model)
            print("[Train] Graph compilation successful. (Expect a 2-5 min JIT pause when games start)")
        except Exception as e:
            print(f"[Train] torch.compile() failed (common on MPS). Falling back to dynamic graph: {e}")
    
    # Initialize Elo Tracker
    elo_tracker = EloTracker(save_path=ELO_PATH)
    _elo_tracker = elo_tracker
    print(f"[Train] Elo Tracker initialized (Main: {elo_tracker.get_rating('Main'):.0f})")
    
    # Initialize Game DB
    game_db = GameDB(GAME_DB_PATH)
    _game_db = game_db
    print(f"[Train] Game DB initialized ({game_db.get_total_games()} games recorded)")
    
    # Eval-only mode
    if args.eval_only:
        from evaluate import evaluate_model
        results = evaluate_model(model, device, num_games=500, verbose=True)
        print(f"\nWin Rate: {results['win_rate_percent']}%")
        return
    
    # Start dashboard
    if not args.no_dashboard:
        start_dashboard_server(port=args.port)
    
    # Initialize game player (Vectorized Actor-Critic)
    player = VectorACGamePlayer(trainer, BATCH_SIZE, device)
    _player = player
    
    # Stats tracking
    rolling_win_rate = deque(maxlen=500)
    start_time = time.time()
    last_save_time = time.time()
    games_since_eval = 0
    eval_drops = 0  # Early stopping: count consecutive eval drops
    games_at_start = trainer.total_games
    trainer.play_alarm = args.alarm
    
    print(f"\n{'='*60}")
    print(f"  TD-Ludo Actor-Critic Training — {MODE} Mode")
    print(f"  Algorithm: PPO (Proximal Policy Optimization)")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")
    print(f"  Eval: every {EVAL_INTERVAL} games ({EVAL_GAMES} games each)")
    print(f"  Ghost saves: every {GHOST_SAVE_INTERVAL} games")
    print(f"  Elo tracking: ON | Game DB: ON")
    print(f"  Batch Size: {BATCH_SIZE}")
    if args.hours > 0:
        print(f"  Time limit: {args.hours} hours")
    if args.games > 0:
        print(f"  Game limit: {args.games}")
    print(f"  Ctrl+C to save and exit gracefully")
    print(f"{'='*60}\n")
    
    try:
        while not STOP_REQUESTED:
            # ---- Check limits ----
            session_games = trainer.total_games - games_at_start
            if args.games > 0 and session_games >= args.games:
                print(f"[Train] Reached game limit ({args.games} games this session)")
                break
            
            if args.hours > 0:
                elapsed_hours = (time.time() - start_time) / 3600
                if elapsed_hours >= args.hours:
                    print(f"[Train] Reached time limit ({args.hours} hours)")
                    break
            
            if SECOND_CTRL_C:
                break
            
            # ---- Play one batch step ----
            results = player.play_step(train=True)
            
            # Process results of finished games
            for result in results:
                games_since_eval += 1
                
                # Track stats
                rolling_win_rate.append(1 if result['model_won'] else 0)
                
                # ---- Elo update ----
                elo_tracker.update_from_game(
                    result['identities'], result['winner'],
                    game_num=trainer.total_games
                )
                
                # ---- Game DB record ----
                game_db.add_game(
                    game_num=trainer.total_games,
                    identities=result['identities'],
                    winner=result['winner'],
                    game_length=result.get('total_moves', 0),
                    avg_td_error=0.0,  # No TD error in AC
                    model_player_idx=result['model_player'],
                )
                
                # ---- Ghost saving ----
                trainer.maybe_save_ghost(elo_tracker=elo_tracker)
                
                # ---- Logging (every 10 games) ----
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
                    
                    # Write live stats for dashboard
                    trainer.write_live_stats(
                        win_rate, gpm,
                        temperature=temperature,
                        elo_tracker=elo_tracker, game_db=game_db
                    )
            
            # ---- Time-based auto-save ----
            if time.time() - last_save_time > SAVE_INTERVAL:
                trainer.save_checkpoint()
                elo_tracker.save()
                last_save_time = time.time()
                print(f"[Auto-save] Checkpoint saved (game {trainer.total_games})")

            # ---- Periodic evaluation ----
            if games_since_eval >= EVAL_INTERVAL:
                print(f"\n--- Evaluation ({EVAL_GAMES} games) ---")
                from evaluate import evaluate_model
                
                eval_results = evaluate_model(model, device, num_games=EVAL_GAMES, verbose=False)
                eval_wr = eval_results['win_rate']
                print(f"--- Eval Win Rate: {eval_results['win_rate_percent']}% ---\n")
                
                is_best = eval_wr > trainer.best_win_rate
                if is_best:
                    trainer.best_win_rate = eval_wr
                    eval_drops = 0  # Reset drop counter on new best
                    trainer.is_stagnated = False  # Clear stagnation flag
                    print(f"  ★ New best: {eval_results['win_rate_percent']}%!")
                else:
                    eval_drops += 1
                    print(f"  ↓ No improvement ({eval_drops}/{EARLY_STOP_PATIENCE} patience)")
                
                win_rate_100 = sum(rolling_win_rate) / max(1, len(rolling_win_rate))
                trainer.log_metrics(win_rate_100, trainer.total_games, eval_win_rate=eval_wr)
                trainer.last_eval_wr = eval_wr  # Cache it so live_stats.json picks it up on the next tick
                
                trainer.save_checkpoint(is_best=is_best)
                elo_tracker.save()
                games_since_eval = 0
                last_save_time = time.time()
                
                # Stagnation warning (no actual early stopping)
                trainer.is_stagnated = (eval_drops >= EARLY_STOP_PATIENCE)
                if trainer.is_stagnated:
                    print(f"\n[Train] ⚠️ WARNING: Model stuck! No improvement for {EARLY_STOP_PATIENCE} consecutive evals.")
            
    except KeyboardInterrupt:
        print("\n[Train] Keyboard interrupt")
    except Exception as e:
        print(f"\n[Train] Error: {e}")
        import traceback
        traceback.print_exc()
    
    # ---- Final save ----
    print("[Train] Final save...")
    trainer.flush_buffer()  # Train on any remaining buffered PPO data
    trainer.save_checkpoint()
    elo_tracker.save()
    
    elapsed = time.time() - start_time
    games_played = trainer.total_games - games_at_start
    
    print(f"\n{'='*60}")
    print(f"  Training Complete ({MODE})")
    print(f"  Games This Session: {games_played}")
    print(f"  Total Games: {trainer.total_games}")
    print(f"  Total Updates: {trainer.total_updates}")
    print(f"  Main Elo: {elo_tracker.get_rating('Main'):.0f}")
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
