"""
TD-Ludo Training Entry Point

Training loop with:
- Batched execution using VectorGameState + VectorTDGamePlayer
- Automatic kickstart weight loading
- Online TD(0) updates + periodic experience replay
- Ghost checkpoint saving at regular intervals
- Elo rating tracking for all agents
- SQLite game history database
- Live stats JSON for dashboard consumption
- Built-in HTTP server for dashboard UI
- Graceful shutdown (Ctrl+C finishes current batch step, saves everything)
- Hours-based and games-based limits
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

from src.model import AlphaLudoV3
from src.trainer import TDTrainer
from src.game_player import VectorTDGamePlayer, TDGamePlayer
from src.elo_tracker import EloTracker
from src.game_db import GameDB
from src.config import (
    LEARNING_RATE, WEIGHT_DECAY, EVAL_INTERVAL, EVAL_GAMES,
    SAVE_INTERVAL, CHECKPOINT_DIR, MAIN_CKPT_PATH,
    KICKSTART_PATH, STATS_PATH, METRICS_PATH,
    REPLAY_EVERY_N_GAMES, USE_EXPERIENCE_BUFFER,
    GHOST_SAVE_INTERVAL, MODE, ELO_PATH, GAME_DB_PATH,
    BATCH_SIZE,
)


# =============================================================================
# Graceful Shutdown
# =============================================================================
STOP_REQUESTED = False
SECOND_CTRL_C = False

def signal_handler(sig, frame):
    global STOP_REQUESTED, SECOND_CTRL_C
    if STOP_REQUESTED:
        # Second Ctrl+C = force exit
        SECOND_CTRL_C = True
        print("\n[Train] Force shutdown! Saving emergency checkpoint...")
        return
    STOP_REQUESTED = True
    print("\n[Train] Received Ctrl+C — finishing current batch and saving...")
    print("[Train] Press Ctrl+C again to force exit (may lose progress)")

signal.signal(signal.SIGINT, signal_handler)


# =============================================================================
# PID Lock — Prevent multiple training instances
# =============================================================================
LOCK_FILE = os.path.join(CHECKPOINT_DIR, "train.pid")

def _is_process_alive(pid):
    """Check if a process with given PID is still running."""
    try:
        os.kill(pid, 0)  # Signal 0 = check if alive, don't actually kill
        return True
    except (ProcessLookupError, PermissionError):
        return False

def acquire_lock():
    """
    Try to acquire the training lock. Returns True if successful.
    If another training instance is already running, prints an error and returns False.
    """
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, 'r') as f:
                old_pid = int(f.read().strip())
            if _is_process_alive(old_pid):
                print(f"\n{'='*60}")
                print(f"  ❌ ANOTHER TRAINING INSTANCE IS ALREADY RUNNING!")
                print(f"  PID: {old_pid}")
                print(f"  Lock file: {LOCK_FILE}")
                print(f"")
                print(f"  To force start, either:")
                print(f"    1. Kill the other process: kill {old_pid}")
                print(f"    2. Remove the lock: rm {LOCK_FILE}")
                print(f"{'='*60}\n")
                return False
            else:
                # Stale lock file — process died without cleanup
                print(f"[Train] Removing stale lock (PID {old_pid} no longer running)")
        except (ValueError, IOError):
            print("[Train] Removing corrupt lock file")
    
    # Write our PID
    with open(LOCK_FILE, 'w') as f:
        f.write(str(os.getpid()))
    return True

def release_lock():
    """Remove the PID lock file."""
    try:
        if os.path.exists(LOCK_FILE):
            with open(LOCK_FILE, 'r') as f:
                pid = int(f.read().strip())
            # Only remove if it's our lock
            if pid == os.getpid():
                os.remove(LOCK_FILE)
    except (ValueError, IOError, OSError):
        pass

import atexit
atexit.register(release_lock)


# =============================================================================
# Global references for API access (set in main())
# =============================================================================
_elo_tracker = None
_game_db = None


# =============================================================================
# Dashboard HTTP Server (serves static files + live stats + Elo + games API)
# =============================================================================
def start_dashboard_server(port=8787):
    """Start HTTP server in background thread to serve dashboard."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    handler = functools.partial(DashboardHandler, directory=project_root)
    try:
        server = HTTPServer(('0.0.0.0', port), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"[Dashboard] http://localhost:{port}/index.html")
        return server
    except OSError as e:
        print(f"[Dashboard] Failed to start (port {port} in use?): {e}")
        return None


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
        elif self.path == '/api/games':
            self._serve_games()
        elif self.path == '/api/spectate':
            self._serve_spectate()
        elif self.path == '/api/system':
            self._serve_system()
        elif self.path == '/api/history':
            self._serve_games()
        elif self.path == '/api/live':
            self._serve_spectate()
        else:
            super().do_GET()
    
    def _serve_json(self, path):
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(data.encode())
            else:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{}')
        except Exception:
            self.send_response(500)
            self.end_headers()
    
    def _serve_elo(self):
        """Serve Elo tracker data as JSON."""
        try:
            data = {}
            if _elo_tracker is not None:
                data = _elo_tracker.to_dict()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        except Exception:
            self.send_response(500)
            self.end_headers()
    
    def _serve_games(self):
        """Serve recent game history from DB as JSON."""
        try:
            data = {}
            if _game_db is not None:
                data = _game_db.to_dict()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        except Exception:
            self.send_response(500)
            self.end_headers()
    
    def _serve_system(self):
        """Serve system resource usage."""
        try:
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            data = {"cpu": cpu, "ram": ram, "mode": MODE}
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        except Exception:
            self.send_response(500)
            self.end_headers()

    def _serve_spectate(self):
        """Serve live game state for spectator."""
        try:
            data = {}
            if _player is not None:
                data = _player.get_spectator_state(0) # Watch game 0
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        except Exception:
            self.send_response(500)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress HTTP access logs


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
    
    parser = argparse.ArgumentParser(description='TD-Ludo Training')
    parser.add_argument('--kickstart', type=str, default=None,
                        help='Path to kickstart model (default: pretrained/model_kickstart.pt)')
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
    parser.add_argument('--clear-buffer', action='store_true',
                        help='Resume model weights but clear experience buffer and reset optimizer')
    parser.add_argument('--reset-heads', action='store_true',
                        help='Resume backbone weights but reinitialize value/policy heads, optimizer, and buffer')
    args = parser.parse_args()
    
    # Handle fresh start (purge run directory)
    if args.fresh and os.path.exists(CHECKPOINT_DIR):
        import shutil
        print(f"[Train] Fresh start requested. Purging {CHECKPOINT_DIR}...")
        for f in os.listdir(CHECKPOINT_DIR):
            fpath = os.path.join(CHECKPOINT_DIR, f)
            try:
                if os.path.isfile(fpath):
                    os.unlink(fpath)
                elif os.path.isdir(fpath):
                    shutil.rmtree(fpath)
            except Exception as e:
                print(f"[Train] Warning: could not delete {fpath}: {e}")
    
    # Ensure required subdirectories exist (important for resumes if they were deleted)
    from src.config import GHOSTS_DIR
    os.makedirs(GHOSTS_DIR, exist_ok=True)
    
    # Acquire training lock — prevent duplicate instances
    if not acquire_lock():
        sys.exit(1)
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    print(f"[Train] Device: {device}")
    print(f"[Train] Mode: {MODE}")
    
    # Initialize model
    model = AlphaLudoV3(num_res_blocks=10, num_channels=128)
    model.to(device)
    
    # Initialize trainer
    trainer = TDTrainer(model, device, learning_rate=LEARNING_RATE)
    
    # Load weights
    if args.resume or args.clear_buffer or args.reset_heads:
        loaded = trainer.load_checkpoint()
        if loaded:
            print(f"[Train] Resumed from checkpoint ({trainer.total_games} games, {trainer.total_updates} updates)")
            if args.reset_heads:
                # Keep backbone (conv_input, bn_input, res_blocks), reset heads
                print("[Train] Resetting value/policy/aux heads (keeping backbone)...")
                for name, module in model.named_children():
                    if name in ('policy_fc1', 'policy_fc2', 'value_fc1', 'value_fc2', 'aux_fc1', 'aux_fc2'):
                        module.reset_parameters()
                        print(f"  Reset: {name}")
                # Reset optimizer (Adam state is stale)
                trainer.optimizer = torch.optim.Adam(
                    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
                )
                # Clear buffer (old data has wrong value targets)
                if trainer.experience_buffer is not None:
                    trainer.experience_buffer.clear()
                from src.config import BUFFER_PATH
                if os.path.exists(BUFFER_PATH):
                    os.remove(BUFFER_PATH)
                print(f"[Train] Heads reset, optimizer reset, buffer cleared")
            elif args.clear_buffer:
                # Keep model weights, reset optimizer and buffer
                trainer.optimizer = torch.optim.Adam(
                    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
                )
                if trainer.experience_buffer is not None:
                    trainer.experience_buffer.clear()
                from src.config import BUFFER_PATH
                if os.path.exists(BUFFER_PATH):
                    os.remove(BUFFER_PATH)
                print(f"[Train] Cleared buffer and reset optimizer (model weights kept)")
        else:
            print("[Train] No checkpoint found, starting fresh")
            _try_kickstart(trainer, args.kickstart)
    elif args.kickstart:
        trainer.load_kickstart(args.kickstart)
    else:
        _try_kickstart(trainer, None)
    
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
    dashboard_server = None
    if not args.no_dashboard:
        dashboard_server = start_dashboard_server(port=args.port)
    
    # Initialize game player (Vectorized)
    # Note: We use VectorTDGamePlayer for batch training efficiency
    player = VectorTDGamePlayer(trainer, BATCH_SIZE, device)
    _player = player
    
    # Stats tracking
    rolling_win_rate = deque(maxlen=500)
    rolling_td_error = deque(maxlen=500)
    start_time = time.time()
    last_save_time = time.time()
    games_since_eval = 0
    games_since_replay = 0
    games_at_start = trainer.total_games
    
    print(f"\n{'='*60}")
    print(f"  TD-Ludo Training — {MODE} Mode")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")
    print(f"  Eval: every {EVAL_INTERVAL} games ({EVAL_GAMES} games each)")
    print(f"  Ghost saves: every {GHOST_SAVE_INTERVAL} games")
    print(f"  Elo tracking: ON | Game DB: ON")
    if USE_EXPERIENCE_BUFFER:
        print(f"  Experience buffer: ON (replay every {REPLAY_EVERY_N_GAMES} games)")
    else:
        print(f"  Experience buffer: OFF (pure online learning)")
    print(f"  Batch Size: {BATCH_SIZE} (Configured)")
    if args.hours > 0:
        print(f"  Time limit: {args.hours} hours")
    if args.games > 0:
        print(f"  Game limit: {args.games}")
    print(f"  Ctrl+C to save and exit gracefully")
    print(f"{'='*60}\n")
    
    try:
        while not STOP_REQUESTED:
            # ---- Check limits ----
            if args.games > 0 and trainer.total_games >= args.games:
                print(f"[Train] Reached game limit ({args.games})")
                break
            
            if args.hours > 0:
                elapsed_hours = (time.time() - start_time) / 3600
                if elapsed_hours >= args.hours:
                    print(f"[Train] Reached time limit ({args.hours} hours)")
                    break
            
            if SECOND_CTRL_C:
                break
            
            # ---- Play one batch step ----
            # This advances all 32 games by 1 step, gathering transitions and doing updates.
            # It returns a list of results for any games that finished in this step.
            epsilon = player.get_epsilon(trainer.total_games)
            results = player.play_step(epsilon=epsilon, train=True)
            
            # Process results of finished games (if any)
            for result in results:
                trainer.total_games += 1
                games_since_eval += 1
                games_since_replay += 1
                
                # Track stats
                rolling_win_rate.append(1 if result['model_won'] else 0)
                if result['avg_td_error'] > 0:
                    rolling_td_error.append(result['avg_td_error'])
                
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
                    avg_td_error=result['avg_td_error'],
                    model_player_idx=result['model_player'],
                )
                
                # ---- Experience replay ----
                if USE_EXPERIENCE_BUFFER and games_since_replay >= REPLAY_EVERY_N_GAMES:
                    replay_loss = trainer.replay_experience()
                    games_since_replay = 0
                
                # ---- Ghost saving (with Elo-based pruning) ----
                trainer.maybe_save_ghost(elo_tracker=elo_tracker)
                
                # ---- Logging (every 10 games) ----
                if trainer.total_games % 10 == 0:
                    win_rate = sum(rolling_win_rate) / max(1, len(rolling_win_rate))
                    avg_td = np.mean(list(rolling_td_error)) if rolling_td_error else 0
                    elapsed = time.time() - start_time
                    games_played = trainer.total_games - games_at_start
                    gpm = games_played / (elapsed / 60) if elapsed > 0 else 0
                    
                    buf_size = len(trainer.experience_buffer) if trainer.experience_buffer else 0
                    main_elo = elo_tracker.get_rating('Main')
                    
                    print(f"[G {trainer.total_games:>6d}] "
                          f"WR: {win_rate*100:5.1f}% | "
                          f"TD: {avg_td:.4f} | "
                          f"ε: {epsilon:.3f} | "
                          f"Elo: {main_elo:.0f} | "
                          f"GPM: {gpm:.0f} | "
                          f"Buf: {buf_size:>5d} | "
                          f"{'WIN' if result['model_won'] else 'loss'}")
                    
                    # Write live stats for dashboard (enriched with Elo + opponent stats)
                    trainer.write_live_stats(
                        win_rate, avg_td, epsilon, gpm,
                        elo_tracker=elo_tracker, game_db=game_db
                    )
            
            # ---- Periodic Elo save (every 50 games) ----
            # Moved out of result loop? No, better check periodically independently?
            # Or check using trainer.total_games (which updates inside loop).
            # But if loop processes 30 games at once, we might skip a multiple of 50?
            # > if trainer.total_games % 50 == 0
            # If total_games jumps from 40 to 60, we miss 50.
            # Better: if trainer.total_games >= next_elo_save
            # For now, simplistic modulo check inside loop (done below) is fine for frequent events.
            
            # Check periodic events that need to happen regardless of game finishes
            # (e.g. time-based auto-save)
            if time.time() - last_save_time > SAVE_INTERVAL:
                trainer.save_checkpoint()
                elo_tracker.save()
                last_save_time = time.time()
                print(f"[Auto-save] Checkpoint saved (game {trainer.total_games})")

            # ---- Periodic evaluation (based on games count) ----
            # This should be inside result loop or check games_since_eval
            if games_since_eval >= EVAL_INTERVAL:
                print(f"\n--- Evaluation ({EVAL_GAMES} games) ---")
                from evaluate import evaluate_model
                
                # Evaluation uses single-game play (usually) for simplicity/correctness
                # We can use our VectorTDGamePlayer logic, but evaluate_model might expect TDGamePlayer?
                # evaluate_model likely imports TDGamePlayer.
                # Since we didn't delete TDGamePlayer, it's fine.
                eval_results = evaluate_model(model, device, num_games=EVAL_GAMES, verbose=False)
                eval_wr = eval_results['win_rate']
                print(f"--- Eval Win Rate: {eval_results['win_rate_percent']}% ---\n")
                
                is_best = eval_wr > trainer.best_win_rate
                if is_best:
                    trainer.best_win_rate = eval_wr
                    print(f"  ★ New best: {eval_results['win_rate_percent']}%!")
                
                win_rate_100 = sum(rolling_win_rate) / max(1, len(rolling_win_rate))
                avg_td = np.mean(list(rolling_td_error)) if rolling_td_error else 0
                trainer.log_metrics(win_rate_100, avg_td, epsilon, trainer.total_games, eval_win_rate=eval_wr)
                
                trainer.save_checkpoint(is_best=is_best)
                elo_tracker.save()
                games_since_eval = 0
                last_save_time = time.time()
            
    except KeyboardInterrupt:
        print("\n[Train] Keyboard interrupt")
    except Exception as e:
        print(f"\n[Train] Error: {e}")
        import traceback
        traceback.print_exc()
    
    # ---- Final save (always runs) ----
    print("[Train] Final save...")
    trainer.flush_gradients()
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
    
    # Print Elo rankings
    print(f"\n{elo_tracker}")


def _try_kickstart(trainer, explicit_path):
    """Try to load kickstart weights from explicit path or default location."""
    path = explicit_path or KICKSTART_PATH
    if os.path.exists(path):
        trainer.load_kickstart(path)
    else:
        print(f"[Train] No kickstart found at {path}, starting with random weights")


if __name__ == '__main__':
    main()
