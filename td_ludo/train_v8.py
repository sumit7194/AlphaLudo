"""
TD-Ludo V8 Training Entry Point — V6 CNN + Temporal Transformer

Takes the proven V6 CNN backbone, freezes it, wraps with temporal transformer.
PPO trains only the transformer + heads (~400K params) while the CNN provides
proven spatial features from 382K games of V6 training.

Training loop with:
- AlphaLudoV8 model (V6 CNN backbone + temporal transformer, K=16 context)
- PPO (Proximal Policy Optimization) with dense v1 rewards
- Batched execution using VectorGameState + VectorV8GamePlayer
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

# V8 uses its own checkpoint directory
os.environ.setdefault('TD_LUDO_RUN_NAME', 'ac_v8_cnn_transformer')

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model_v8 import AlphaLudoV8
from src.trainer_v8 import V8Trainer
from src.game_player_v8 import VectorV8GamePlayer
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
# Global references for API access (set in main())
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
        print(f"[Dashboard] Warning: index.html not found in {dashboard_dir}, skipping dashboard server")
        return None
    handler = functools.partial(DashboardHandler, directory=dashboard_dir)
    try:
        server = HTTPServer(('0.0.0.0', port), handler)
    except OSError as e:
        print(f"[Dashboard] Warning: Could not bind to port {port} ({e}). Dashboard disabled — training continues.")
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
# V6 Weights Path
# =============================================================================
V6_WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'checkpoints', 'ac_v6_big', 'backups', 'model_final_v6_382k_70pct.pt'
)


# =============================================================================
# Main Training Loop
# =============================================================================
def main():
    global _elo_tracker, _game_db, _player

    parser = argparse.ArgumentParser(description='TD-Ludo V8 CNN+Transformer Training')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest V8 checkpoint')
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
    parser.add_argument('--freeze-cnn', action='store_true', default=True,
                        help='Freeze CNN backbone (default: True)')
    parser.add_argument('--unfreeze-cnn', action='store_true',
                        help='Unfreeze CNN backbone for end-to-end training')
    parser.add_argument('--v6-weights', type=str, default=V6_WEIGHTS_PATH,
                        help='Path to V6 CNN weights')
    parser.add_argument('--context-length', type=int, default=16,
                        help='Context window size (default: 16)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size (default: 256, with CNN caching only 1 new turn per game per step)')
    parser.add_argument('--unfreeze-after', type=int, default=0,
                        help='Auto-unfreeze CNN after N total games (0=never). Uses 0.1x LR for CNN.')
    parser.add_argument('--ppo-minibatch', type=int, default=0,
                        help='PPO minibatch size override (0=use config default). Larger = fewer but faster minibatches.')
    args = parser.parse_args()

    if args.unfreeze_cnn:
        args.freeze_cnn = False

    # Handle fresh start
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

    from src.config import GHOSTS_DIR
    os.makedirs(GHOSTS_DIR, exist_ok=True)

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
    context_length = args.context_length
    model_factory = lambda: AlphaLudoV8(context_length=context_length)
    model = model_factory()

    # Load V6 CNN weights
    if not args.resume:
        if os.path.exists(args.v6_weights):
            model.load_v6_weights(args.v6_weights)
        else:
            print(f"[Train] WARNING: V6 weights not found at {args.v6_weights}")
            print(f"[Train] Starting CNN from random weights.")

    # Freeze/unfreeze CNN
    if args.freeze_cnn:
        model.freeze_cnn()
        trainable = model.count_parameters()
        total = model.count_all_parameters()
        print(f"[Train] Architecture: AlphaLudoV8 (V6 CNN frozen + Transformer)")
        print(f"[Train] Total params: {total:,} | Trainable: {trainable:,}")
    else:
        total = model.count_parameters()
        print(f"[Train] Architecture: AlphaLudoV8 (V6 CNN unfrozen + Transformer)")
        print(f"[Train] Total trainable params: {total:,}")

    model.to(device)

    # JIT compile for faster forward passes (especially transformer in PPO)
    try:
        model = torch.compile(model, backend='aot_eager')
        print(f"[Train] torch.compile enabled (aot_eager backend)")
    except Exception as e:
        print(f"[Train] torch.compile unavailable ({e}), continuing without JIT")

    # Initialize trainer
    trainer = V8Trainer(model, device, learning_rate=LEARNING_RATE)

    # Override PPO minibatch size if specified
    if args.ppo_minibatch > 0:
        trainer.ppo_minibatch_size = args.ppo_minibatch
        print(f"[Train] PPO minibatch size override: {args.ppo_minibatch}")

    # Load checkpoint or start fresh
    if args.resume:
        loaded = trainer.load_checkpoint()
        if loaded:
            print(f"[Train] Resumed from checkpoint ({trainer.total_games} games, {trainer.total_updates} updates)")
        else:
            print("[Train] No V8 checkpoint found. Starting from scratch with V6 CNN weights.")
    else:
        print("[Train] Fresh V8 start. Transformer weights are random, CNN weights from V6.")

    # Initialize Elo Tracker
    elo_tracker = EloTracker(save_path=ELO_PATH)
    _elo_tracker = elo_tracker
    print(f"[Train] Elo Tracker initialized (Model: {elo_tracker.get_rating('Model'):.0f})")

    # Initialize Game DB
    game_db = GameDB(GAME_DB_PATH)
    _game_db = game_db
    print(f"[Train] Game DB initialized ({game_db.get_total_games()} games recorded)")

    # Eval-only mode
    if args.eval_only:
        from evaluate_v8 import evaluate_v8_model
        results = evaluate_v8_model(model, device, num_games=500, verbose=True,
                                    context_length=context_length)
        print(f"\nWin Rate: {results['win_rate_percent']}%")
        return

    # Start dashboard
    if not args.no_dashboard:
        start_dashboard_server(port=args.port)

    # Initialize game player (V8 uses smaller batch than V6 due to K×CNN overhead)
    batch_size = args.batch_size
    player = VectorV8GamePlayer(
        trainer, batch_size, device,
        context_length=context_length,
        model_factory=model_factory,
        elo_tracker=elo_tracker,
    )
    _player = player

    # Stats tracking
    rolling_win_rate = deque(maxlen=500)
    start_time = time.time()
    last_save_time = time.time()
    games_since_eval = 0
    eval_drops = 0
    games_at_start = trainer.total_games
    trainer.play_alarm = args.alarm

    print(f"\n{'='*60}")
    print(f"  TD-Ludo V8 CNN+Transformer Training — {MODE} Mode")
    print(f"  Algorithm: PPO (Proximal Policy Optimization)")
    print(f"  CNN: {'FROZEN' if args.freeze_cnn else 'UNFROZEN'} (V6 backbone)")
    print(f"  Context: K={context_length} turns")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")
    print(f"  Eval: every {EVAL_INTERVAL} games ({EVAL_GAMES} games each)")
    print(f"  Ghost saves: every {GHOST_SAVE_INTERVAL} games")
    print(f"  Elo tracking: ON | Game DB: ON")
    print(f"  Batch Size: {batch_size} (CNN passes/step: {batch_size}×{context_length}={batch_size*context_length})")
    if args.hours > 0:
        print(f"  Time limit: {args.hours} hours")
    if args.games > 0:
        print(f"  Game limit: {args.games}")
    if args.unfreeze_after > 0:
        print(f"  Auto-unfreeze CNN: at game {args.unfreeze_after} (CNN LR: 0.1x)")
    print(f"  Ctrl+C to save and exit gracefully")
    print(f"{'='*60}\n")

    try:
        while not STOP_REQUESTED:
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

            # Auto-unfreeze CNN at specified game count
            if (args.unfreeze_after > 0
                    and trainer.total_games >= args.unfreeze_after
                    and model.cnn_frozen):
                print(f"\n[Train] === Auto-unfreezing CNN at game {trainer.total_games} ===")
                model.unfreeze_cnn()
                trainer.rebuild_optimizer(cnn_lr_scale=0.1)
                total = model.count_parameters()
                print(f"[Train] CNN unfrozen. Total trainable params: {total:,}")
                print(f"[Train] CNN LR: 0.1x base | Throughput will decrease (no CNN caching)\n")

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
                from evaluate_v8 import evaluate_v8_model

                eval_results = evaluate_v8_model(
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
                    print(f"  ★ New best: {eval_results['win_rate_percent']}%!")
                else:
                    eval_drops += 1
                    print(f"  ↓ No improvement ({eval_drops}/{EARLY_STOP_PATIENCE} patience)")

                win_rate_100 = sum(rolling_win_rate) / max(1, len(rolling_win_rate))
                trainer.log_metrics(win_rate_100, trainer.total_games, eval_win_rate=eval_wr)
                trainer.last_eval_wr = eval_wr

                trainer.save_checkpoint(is_best=is_best)
                elo_tracker.save()
                games_since_eval = 0
                last_save_time = time.time()

                trainer.is_stagnated = (eval_drops >= EARLY_STOP_PATIENCE)
                if trainer.is_stagnated:
                    print(f"\n[Train] WARNING: Model stuck! No improvement for {EARLY_STOP_PATIENCE} consecutive evals.")

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
    print(f"  V8 Training Complete ({MODE})")
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
