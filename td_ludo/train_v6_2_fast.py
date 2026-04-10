#!/usr/bin/env python3
"""
TD-Ludo V9 Fast Training — Multi-Process Actor-Learner Architecture

Architecture:
  N Actor processes (CPU) → trajectory_queue → 1 Learner process (MPS/CUDA)
                          ← weight_file (periodic sync) ←

Actors play games on CPU with batched inference. Learner runs PPO on MPS.
Actors periodically reload updated weights from the learner.
Dashboard runs in the main process, reading stats from files.

Usage:
  ./td_env/bin/python3 train_v9_fast.py --fresh
  ./td_env/bin/python3 train_v9_fast.py --resume
  ./td_env/bin/python3 train_v9_fast.py --actors 6 --actor-batch 64
"""

import os
import sys
import time
import signal
import argparse
import json
import threading
import functools
import multiprocessing as mp
from http.server import HTTPServer, SimpleHTTPRequestHandler

# Set multiprocessing start method before anything else
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass  # already set

os.environ.setdefault('TD_LUDO_RUN_NAME', 'ac_v6_2_transformer')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# Paths (computed early, before spawning processes)
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RUN_NAME = os.environ.get('TD_LUDO_RUN_NAME', 'ac_v6_2_transformer')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints', RUN_NAME)
GHOSTS_DIR = os.path.join(CHECKPOINT_DIR, 'ghosts')
MAIN_CKPT_PATH = os.path.join(CHECKPOINT_DIR, 'model_latest.pt')
BEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, 'model_best.pt')
STATS_PATH = os.path.join(CHECKPOINT_DIR, 'live_stats.json')
WEIGHT_SYNC_PATH = os.path.join(CHECKPOINT_DIR, 'actor_weights.pt')

SL_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints', RUN_NAME)
SL_BEST_PATH = os.path.join(SL_CHECKPOINT_DIR, 'model_sl.pt')
SL_MODEL_PATH = os.path.join(SL_CHECKPOINT_DIR, 'model_sl_v6_2.pt')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(GHOSTS_DIR, exist_ok=True)

# =============================================================================
# Graceful Shutdown
# =============================================================================
STOP_EVENT = mp.Event()


def signal_handler(sig, frame):
    if STOP_EVENT.is_set():
        print("\n[Main] Force exit.")
        sys.exit(1)
    print("\n[Main] Graceful shutdown requested...")
    STOP_EVENT.set()


signal.signal(signal.SIGINT, signal_handler)

# =============================================================================
# Dashboard HTTP Server (runs in main process)
# =============================================================================
class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self):
        if self.path == '/api/stats':
            self._serve_json(STATS_PATH)
        elif self.path == '/api/metrics':
            self._serve_json(os.path.join(CHECKPOINT_DIR, 'training_metrics.json'))
        elif self.path == '/api/elo':
            self._serve_json(os.path.join(CHECKPOINT_DIR, 'elo_ratings.json'))
        elif self.path == '/api/system':
            self._serve_system()
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

    def _serve_system(self):
        try:
            import psutil
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

    def log_message(self, format, *args):
        pass


def start_dashboard_server(port=8787):
    dashboard_dir = PROJECT_ROOT
    if not os.path.exists(os.path.join(dashboard_dir, 'index.html')):
        print(f"[Dashboard] Warning: index.html not found, skipping")
        return None
    handler = functools.partial(DashboardHandler, directory=dashboard_dir)
    try:
        server = HTTPServer(('0.0.0.0', port), handler)
    except OSError as e:
        print(f"[Dashboard] Warning: Could not bind to port {port} ({e})")
        return None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[Dashboard] Server started at http://localhost:{port}")
    return server


# =============================================================================
# Device Detection
# =============================================================================
def get_device_str():
    import torch
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='TD-Ludo V9 Fast Multi-Process Training'
    )
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint')
    parser.add_argument('--fresh', action='store_true',
                        help='Start fresh (purge run data)')
    parser.add_argument('--device', type=str, default=None,
                        help='Learner device override')
    parser.add_argument('--actors', type=int, default=2,
                        help='Number of actor processes (0=auto)')
    parser.add_argument('--actor-batch', type=int, default=32,
                        help='Games per actor process')
    parser.add_argument('--context-length', type=int, default=16,
                        help='Transformer context window')
    parser.add_argument('--ppo-buffer', type=int, default=4096,
                        help='PPO buffer size in steps')
    parser.add_argument('--ppo-minibatch', type=int, default=256,
                        help='PPO minibatch size')
    parser.add_argument('--games', type=int, default=0,
                        help='Max games (0=unlimited)')
    parser.add_argument('--hours', type=float, default=0,
                        help='Max hours (0=unlimited)')
    parser.add_argument('--no-dashboard', action='store_true')
    parser.add_argument('--port', type=int, default=8787)
    parser.add_argument('--sl-weights', type=str, default=None,
                        help='Path to SL pre-trained weights')
    parser.add_argument('--queue-size', type=int, default=500,
                        help='Max trajectory queue size')
    parser.add_argument('--gpu-actors', action='store_true', default=False,
                        help='Use GPU inference server for actors (experimental)')
    parser.add_argument('--no-gpu-actors', dest='gpu_actors', action='store_false',
                        help='Use CPU actors (default)')
    parser.add_argument('--quantize-actors', action='store_true', default=False,
                        help='Use int8 quantized models for actor inference')
    parser.add_argument('--coreml', action='store_true', default=False,
                        help='Use Core ML Neural Engine for CNN feature pre-computation')
    args = parser.parse_args()

    # Fresh start
    if args.fresh and os.path.exists(CHECKPOINT_DIR):
        import shutil
        print(f"[Main] Fresh start. Purging {CHECKPOINT_DIR}...")
        for f in os.listdir(CHECKPOINT_DIR):
            fpath = os.path.join(CHECKPOINT_DIR, f)
            try:
                if os.path.isfile(fpath):
                    os.unlink(fpath)
                elif os.path.isdir(fpath):
                    shutil.rmtree(fpath)
            except Exception as e:
                print(f"[Main] Warning: could not delete {fpath}: {e}")
        os.makedirs(GHOSTS_DIR, exist_ok=True)

    # Device
    device_str = args.device or get_device_str()
    print(f"[Main] Learner device: {device_str}")

    # Number of actors
    num_actors = args.actors
    if num_actors <= 0:
        num_actors = max(2, min(os.cpu_count() - 2, 8))
    actor_batch = args.actor_batch
    total_parallel = num_actors * actor_batch
    print(f"[Main] Actors: {num_actors} x {actor_batch} games = "
          f"{total_parallel} parallel games")

    # SL weights
    sl_path = args.sl_weights
    if sl_path is None and not args.resume:
        for candidate in [SL_BEST_PATH, SL_MODEL_PATH]:
            if os.path.exists(candidate):
                sl_path = candidate
                break

    # Resume path
    resume_path = MAIN_CKPT_PATH if args.resume else None

    # Shared state
    weight_version = mp.Value('i', 0)
    total_games_counter = mp.Value('i', 0)  # updated by learner, read by main + actors
    trajectory_queue = mp.Queue(maxsize=args.queue_size)
    stats_queue = mp.Queue(maxsize=100)

    # Build config dict for workers
    config = {
        'context_length': args.context_length,
        'learning_rate': 1e-5,
        'weight_decay': 1e-4,
        'max_grad_norm': 1.0,
        'entropy_coeff': 0.005,
        'value_loss_coeff': 0.5,
        'clip_epsilon': 0.2,
        'ppo_epochs': 3,
        'ppo_buffer_steps': args.ppo_buffer,
        'ppo_minibatch_size': args.ppo_minibatch,
        'num_active_players': 2,
        'game_composition': {
            'SelfPlay': 0.40,
            'Expert': 0.25,
            'Heuristic': 0.15,
            'Aggressive': 0.10,
            'Defensive': 0.10,
        },
        'ghosts_dir': GHOSTS_DIR,
        'selfplay_ghost_fraction': 0.50,
        'selfplay_ghost_strategy': 'matched',
        'max_moves': 10000,
        'temp_start': 1.1,
        'temp_end': 0.95,
        'temp_decay_games': 20000,
        'eval_interval': 2000,
        'eval_games': 500,
        'save_interval': 300,
        'weight_export_interval': 30,
        'ghost_save_interval': 2000,
        'max_ghosts': 20,
        'early_stop_patience': 100,
        'quantize_actors': args.quantize_actors,
        'use_coreml': args.coreml,
    }

    # Dashboard
    if not args.no_dashboard:
        start_dashboard_server(port=args.port)

    use_gpu_actors = args.gpu_actors
    arch_label = (f"{num_actors} GPU-backed actors + 1 {device_str} inference server + 1 {device_str} learner"
                  if use_gpu_actors
                  else f"{num_actors} CPU actors + 1 {device_str} learner")

    print(f"\n{'='*60}")
    print(f"  TD-Ludo V9 FAST Training — Multi-Process")
    print(f"  Architecture: {arch_label}")
    print(f"  Parallel games: {total_parallel}")
    print(f"  PPO buffer: {args.ppo_buffer} steps")
    print(f"  Context: K={args.context_length}")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")
    if sl_path:
        print(f"  SL weights: {sl_path}")
    if resume_path and os.path.exists(resume_path):
        print(f"  Resuming from: {resume_path}")
    if args.hours > 0:
        print(f"  Time limit: {args.hours}h")
    if args.games > 0:
        print(f"  Game limit: {args.games}")
    print(f"  Ctrl+C to save and exit gracefully")
    print(f"{'='*60}\n")

    # Import worker functions (must be importable for spawn)
    from src.fast_learner_v62 import learner_worker

    # Weight update queue for inference server
    weight_update_queue = mp.Queue(maxsize=10) if use_gpu_actors else None

    # Start learner process
    learner_proc = mp.Process(
        target=learner_worker,
        args=(
            trajectory_queue, stats_queue,
            WEIGHT_SYNC_PATH, weight_version, total_games_counter,
            STOP_EVENT,
            device_str, config, CHECKPOINT_DIR, GHOSTS_DIR,
            resume_path, sl_path,
        ),
        kwargs={'weight_update_queue': weight_update_queue},
        name='learner',
        daemon=False,
    )
    learner_proc.start()
    print(f"[Main] Learner process started (PID {learner_proc.pid})")

    # Wait for learner to export initial weights
    print("[Main] Waiting for initial weights...")
    for _ in range(60):
        if os.path.exists(WEIGHT_SYNC_PATH) and weight_version.value > 0:
            break
        time.sleep(0.5)
    else:
        print("[Main] Warning: timed out waiting for initial weights")

    # Start inference server and actors
    inference_proc = None
    actor_procs = []

    if use_gpu_actors:
        from src.inference_server import inference_server_worker
        from src.fast_actor_v62 import actor_worker_gpu

        # Create per-actor response queues
        request_queue = mp.Queue(maxsize=num_actors * 2)
        response_queues = [mp.Queue(maxsize=4) for _ in range(num_actors)]

        # Start inference server
        inference_proc = mp.Process(
            target=inference_server_worker,
            args=(
                args.context_length, device_str,
                request_queue, response_queues,
                weight_update_queue, STOP_EVENT,
                WEIGHT_SYNC_PATH,
            ),
            name='inference-server',
            daemon=False,
        )
        inference_proc.start()
        print(f"[Main] Inference server started (PID {inference_proc.pid})")
        time.sleep(1.0)  # Let server initialize

        # Start GPU-backed actors
        for i in range(num_actors):
            proc = mp.Process(
                target=actor_worker_gpu,
                args=(
                    i, actor_batch, args.context_length,
                    trajectory_queue, stats_queue,
                    request_queue, response_queues[i],
                    total_games_counter, STOP_EVENT,
                    config,
                ),
                name=f'actor-{i}',
                daemon=False,
            )
            proc.start()
            actor_procs.append(proc)
            print(f"[Main] Actor {i} started (PID {proc.pid}, GPU-backed)")
    else:
        from src.fast_actor_v62 import actor_worker

        # Original CPU actor mode
        for i in range(num_actors):
            proc = mp.Process(
                target=actor_worker,
                args=(
                    i, actor_batch, args.context_length,
                    trajectory_queue, stats_queue,
                    WEIGHT_SYNC_PATH, weight_version, total_games_counter,
                    STOP_EVENT,
                    config,
                ),
                name=f'actor-{i}',
                daemon=False,
            )
            proc.start()
            actor_procs.append(proc)
            print(f"[Main] Actor {i} started (PID {proc.pid})")

    start_time = time.time()

    # Main monitoring loop
    print("\n[Main] Training running. Press Ctrl+C to stop.\n")
    try:
        while not STOP_EVENT.is_set():
            time.sleep(2.0)

            # Check if learner is still alive
            if not learner_proc.is_alive():
                print("[Main] Learner process died! Shutting down...")
                STOP_EVENT.set()
                break

            # Check game/time limits
            if args.hours > 0:
                elapsed = (time.time() - start_time) / 3600
                if elapsed >= args.hours:
                    print(f"[Main] Time limit reached ({args.hours}h)")
                    STOP_EVENT.set()
                    break

            # Drain stats queue (for final elo, etc.)
            while True:
                try:
                    stat = stats_queue.get_nowait()
                    if stat.get('type') == 'final_elo':
                        print(f"\nElo Rankings:")
                        for name, elo in stat.get('rankings', []):
                            print(f"  {name}: {elo:.0f}")
                except Exception:
                    break

            # Game limit (learner tracks authoritative count)
            if args.games > 0 and total_games_counter.value >= args.games:
                print(f"[Main] Game limit reached ({args.games})")
                STOP_EVENT.set()
                break

    except KeyboardInterrupt:
        print("\n[Main] Caught interrupt in main loop")
        STOP_EVENT.set()

    # Shutdown
    print("[Main] Waiting for processes to finish...")

    # Wait for actors (with timeout)
    for i, proc in enumerate(actor_procs):
        proc.join(timeout=10)
        if proc.is_alive():
            print(f"[Main] Actor {i} didn't stop, terminating...")
            proc.terminate()

    # Wait for inference server
    if inference_proc is not None:
        inference_proc.join(timeout=10)
        if inference_proc.is_alive():
            print("[Main] Inference server didn't stop, terminating...")
            inference_proc.terminate()

    # Wait for learner (give it more time to save)
    learner_proc.join(timeout=30)
    if learner_proc.is_alive():
        print("[Main] Learner didn't stop, terminating...")
        learner_proc.terminate()

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"  V9 Fast Training Complete")
    print(f"  Duration: {elapsed/3600:.2f} hours")
    print(f"  Checkpoint: {MAIN_CKPT_PATH}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
