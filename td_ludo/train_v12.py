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
from td_ludo.game.players.v11 import VectorACGamePlayer  # V12.1: V11 player uses encode_state_v11 (33ch: V10 + idle + streak)
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


def get_composition_mix(name):
    """Return the GAME_COMPOSITION dict for a named preset, or None if unknown.
    Centralized so curriculum-mode swap can re-apply mixes without
    duplicating the dicts inside the if/elif chain in main()."""
    if name == 'v122':
        return {"SelfPlay": 0.75, "Expert": 0.15, "Heuristic": 0.05,
                "Aggressive": 0.03, "Defensive": 0.02}
    if name == 'v122_hist':
        return {"SelfPlay": 0.60, "Expert": 0.15, "Heuristic": 0.05,
                "Aggressive": 0.03, "Defensive": 0.02,
                "Hist_V12_2": 0.05, "Hist_V10": 0.05,
                "Hist_V6_3": 0.03, "Hist_V6_1": 0.02}
    if name == 'v122_hist_v2':
        return {"SelfPlay": 0.60, "Expert": 0.07, "Heuristic": 0.03,
                "Hist_V12_2": 0.15, "Hist_V10": 0.10,
                "Hist_V6_3": 0.03, "Hist_V6_1": 0.02}
    return None


def apply_composition(name):
    """Monkey-patch GAME_COMPOSITION in src.config and the live player module.
    Returns True on success. Used at startup AND during curriculum swap."""
    mix = get_composition_mix(name)
    if mix is None:
        return False
    import src.config as _cfg
    _cfg.GAME_COMPOSITION = mix
    import td_ludo.game.players.v11 as _v11mod
    _v11mod.GAME_COMPOSITION = mix
    return True


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
    """Dashboard on port 8789 (8787=V10, 8788=exploiter, 8789=V11, 8790=V12 default)."""
    dashboard_dir = os.path.dirname(os.path.abspath(__file__))
    # V13/V12: check for specific dashboard HTML files
    landing = None
    for cand in ('v13_dashboard.html', 'v12_dashboard.html', 'v11_dashboard.html', 'index.html'):
        if os.path.exists(os.path.join(dashboard_dir, cand)):
            landing = cand
            break
    if landing is None:
        print(f"[Dashboard] No dashboard HTML found, skipping")
        return None
    handler = functools.partial(DashboardHandler, directory=dashboard_dir,
                                landing_page=landing)
    server = HTTPServer(('0.0.0.0', port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[Dashboard] http://localhost:{port}  (default: /{landing})")
    return server


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, landing_page='index.html', **kwargs):
        self._landing = landing_page
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self):
        # V12.2: redirect '/' to the chain-aware dashboard if available.
        if self.path == '/' or self.path == '':
            self.path = '/' + self._landing
            return super().do_GET()
        if self.path == '/api/stats':
            self._serve_json(STATS_PATH)
        elif self.path == '/api/metrics':
            self._serve_json(METRICS_PATH)
        elif self.path == '/api/elo':
            self._serve_elo()
        elif self.path == '/api/sl_stats':
            # V12.2: SL warm-up progress (only present during/after stage 1)
            self._serve_json(os.path.join(os.path.dirname(STATS_PATH), 'sl_stats.json'))
        elif self.path == '/api/chain':
            # V12.2: pipeline status (data-gen / SL / RL)
            self._serve_json(os.path.join(os.path.dirname(STATS_PATH), 'chain_status.json'))
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
    # V13: switch model architecture from V12 (CNN+attn, 33ch) to
    # MinimalCNN14 (10×128 deep CNN, 14ch raw input). When model-arch is
    # 'v13_minimal', V12-specific args (num-attn-layers, num-heads,
    # ffn-ratio, attn-dim) are ignored; defaults for num-res-blocks /
    # num-channels are overridden to 10/128. Use --game-composition v13
    # together with this flag to get the V13-tuned opponent mix.
    parser.add_argument('--curriculum-mode', default='off',
                        choices=['off', 'auto'],
                        help="auto: starts with --game-composition (typically "
                             "v122 = bots only) and swaps to --curriculum-target "
                             "after 3 consecutive evals ≥ --curriculum-eval-thresh.")
    parser.add_argument('--curriculum-target', default='v122_hist_v2',
                        help='Game composition to swap to once gating triggers.')
    parser.add_argument('--curriculum-eval-thresh', type=float, default=0.80,
                        help='Eval WR threshold (decimal, e.g. 0.80) for gating.')
    parser.add_argument('--curriculum-window', type=int, default=3,
                        help='How many recent evals must all clear the threshold.')
    parser.add_argument('--model-arch', default='v12',
                        choices=['v12', 'v13_minimal', 'v131_aux', 'v132', 'v14_scalar', 'v13_5'],
                        help="Model architecture. 'v12' = AlphaLudoV12 "
                             "(CNN+attn, 33ch encoder). 'v13_minimal' = "
                             "MinimalCNN14 (10×128 pure CNN, 14ch raw). "
                             "'v14_scalar' = V14ScalarDeepSets (no CNN, no "
                             "attention; per-token MLPs + DeepSets pool over "
                             "V12.2-equivalent scalar features). 'v13_5' = "
                             "V135ProductionAdapter wrapping V135Symmetric "
                             "(10×128, V18 token-symmetric encoder, "
                             "rank-indexed output mapped back to token-id).")
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

    # V12.2: opponent-mix preset. Bots are saturated for V12-class models;
    # self-play vs ghosts gives more useful gradient than easy bot wins.
    parser.add_argument('--game-composition', default='default',
                        choices=['default', 'v122', 'v122_hist', 'v122_hist_v2', 'v123', 'v13', 'v13_5_no_bots'],
                        help="'default' = config PROD mix (40/25/15/10/10). "
                             "'v122' = SelfPlay 75 / Expert 15 / Heuristic 5 "
                             "/ Aggressive 3 / Defensive 2. "
                             "'v122_hist' = v122 + 15%% historical mix: "
                             "SelfPlay 60 / Expert 15 / Heuristic 5 / "
                             "Aggressive 3 / Defensive 2 / Hist_V12_2 5 / "
                             "Hist_V10 5 / Hist_V6_3 3 / Hist_V6_1 2. "
                             "'v123' = historical models replace bots: "
                             "SelfPlay 67 / Hist_V10 18 / Hist_V6_3 8 / "
                             "Hist_V6_1 4 / Hist_V6_big 3 (no Random). "
                             "'v13' = V13-tuned mix with V12.2 as strong "
                             "external opponent: SelfPlay 55 / Hist_V12_2 "
                             "20 / Hist_V10 15 / Hist_V6_3 10. No bots. "
                             "'v13_5_no_bots' = V13.5-tuned mix targeting "
                             "DNA diversity for V13.5 RL: SelfPlay 50 + "
                             "ghost rotation / Hist_V13_2 25 / Hist_V13_5_SL "
                             "10 / Hist_V12_2 10 / Hist_V10 5. No bots.")

    # Exp 24: search-during-training (depth-1 expectimax → aux policy target).
    parser.add_argument('--search-enabled', action='store_true',
                        help='Run depth-1 expectimax on a fraction of training '
                             'states; use the search argmax as auxiliary CE '
                             'target for the policy head. (Exp 24)')
    parser.add_argument('--search-target-fraction', type=float, default=0.25,
                        help='Fraction of training states to search per turn '
                             '(default 0.25). Cost scales linearly.')
    parser.add_argument('--alpha-search', type=float, default=0.5,
                        help='Weight of the search-target CE loss term '
                             '(default 0.5). Only active with --search-enabled.')
    parser.add_argument('--search-label-smoothing', type=float, default=0.1,
                        help='Label smoothing applied to the one-hot target '
                             '(default 0.1; argmax gets 0.9, rest legal share '
                             '0.1 uniformly).')

    # Progress shaping (Exp 39). Per-step shaping reward = α * (ΔΦ_own − ΔΦ_opp)
    # where Φ = total progress score (Σ S(pos) over a player's tokens, with
    # S(pos) the non-linear curve in td_ludo.game.progress_score). The shaping
    # is potential-based and policy-invariant in expectation, but provides
    # much denser per-step signal than the +1/-1 terminal reward alone.
    parser.add_argument('--shaping-coeff', type=float, default=0.0,
                        help='Per-step potential-based shaping coefficient α. '
                             '0 disables shaping. Reasonable starting values: '
                             '0.02-0.05. Bigger = more weight on per-step '
                             'progress vs terminal outcome.')

    # V13.5 progress aux head loss (only meaningful when --model-arch=v13_5,
    # since other archs don't expose a progress head).
    parser.add_argument('--progress-coeff', type=float, default=0.0,
                        help='V13.5 progress aux loss coefficient. >0 trains '
                             "the per-rank progress head against player_v11's "
                             'progress_target (S(pos) per canonical rank). '
                             '0 disables. Recommended start: 0.05-0.1.')

    # Eval cadence overrides (default comes from src.config — usually
    # 25000/2500). Lower interval = more frequent feedback at the cost of
    # more eval-game throughput. Lower per-eval games = noisier WR estimate.
    parser.add_argument('--eval-interval', type=int, default=0,
                        help='Run evaluation every N training games. '
                             '0 = use config default (typically 25000).')
    parser.add_argument('--eval-games', type=int, default=0,
                        help='Number of games per evaluation round. '
                             '0 = use config default (typically 2500).')

    args = parser.parse_args()

    # CLI overrides for eval cadence. Shadow the imported config constants so
    # the bucket / eval-call sites use the user-specified values when given.
    global EVAL_INTERVAL, EVAL_GAMES
    if args.eval_interval > 0:
        EVAL_INTERVAL = args.eval_interval
    if args.eval_games > 0:
        EVAL_GAMES = args.eval_games

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

    # Build model — V12 (CNN+attn, 33ch) or V13/V13.1 (MinimalCNN14[Aux], 14ch).
    if args.model_arch == 'v13_minimal':
        from experiments.distillation_14ch.model_14ch import MinimalCNN14
        # V13 defaults: 10×128 deep CNN, 14ch raw input.
        v13_blocks = args.num_res_blocks if args.num_res_blocks != 4 else 10
        v13_channels = args.num_channels if args.num_channels != 96 else 128
        model_factory = lambda: MinimalCNN14(
            num_res_blocks=v13_blocks,
            num_channels=v13_channels,
            in_channels=14,
        )
        model = model_factory()
        print(f"[V12 Train] Model: MinimalCNN14 (V13) "
              f"({model.count_parameters():,} params)")
        print(f"[V12 Train]   {v13_blocks} ResBlocks × {v13_channels}ch, "
              f"pure CNN, 14ch raw input")
        import td_ludo_cpp as _ludo
        encoder_fn = _ludo.encode_state_v14_minimal
    elif args.model_arch == 'v132':
        # V13.2: MinimalCNN14 with 17ch input (V14 + 3 V11 static channels).
        # Default 10×128 (~3M params). No aux heads. The static board layout
        # is provided as input rather than learned via aux losses.
        from experiments.distillation_14ch.model_14ch import MinimalCNN14
        from td_ludo.game.encoder_v17 import encode_state_v17, V17_CHANNELS
        v132_blocks = args.num_res_blocks if args.num_res_blocks != 4 else 10
        v132_channels = args.num_channels if args.num_channels != 96 else 128
        model_factory = lambda: MinimalCNN14(
            num_res_blocks=v132_blocks,
            num_channels=v132_channels,
            in_channels=V17_CHANNELS,
        )
        model = model_factory()
        print(f"[V12 Train] Model: MinimalCNN14 (V13.2) "
              f"({model.count_parameters():,} params)")
        print(f"[V12 Train]   {v132_blocks} ResBlocks × {v132_channels}ch, "
              f"{V17_CHANNELS}ch input (V14 + 3 static), no aux heads")
        encoder_fn = encode_state_v17
    elif args.model_arch == 'v14_scalar':
        # V14_scalar: DeepSets over V12.2-equivalent scalar features.
        # No CNN, no attention. ~225K params at default sizes.
        # Encoder returns a (FLAT_DIM=73, 1, 1) tensor; the model unpacks
        # it back into the per-token + global dict structure internally so
        # the existing tensor-based RL pipeline works unchanged.
        from td_ludo.models.v14_scalar import V14ScalarDeepSets
        from td_ludo.game.encoder_v14_scalar import (
            encode_state_v14_scalar_flat, FLAT_DIM,
        )
        model_factory = lambda: V14ScalarDeepSets()
        model = model_factory()
        print(f"[V12 Train] Model: V14ScalarDeepSets (V14_scalar) "
              f"({model.count_parameters():,} params)")
        print(f"[V12 Train]   DeepSets, no CNN/attn, "
              f"input shape ({FLAT_DIM}, 1, 1) flat tensor")
        encoder_fn = encode_state_v14_scalar_flat
    elif args.model_arch == 'v13_5':
        # V13.5: token-symmetric V18 encoder + V135Symmetric backbone
        # (10×128, rank-indexed output). The production-pipeline adapter
        # packs (V18 base + rank masks + token_to_rank) into a single 21ch
        # tensor and exposes the standard token-id-indexed forward signature
        # so trainer_v10 / VectorACGamePlayer can drive it unchanged. Tokens
        # at the same canonical rank get equal probability (architectural
        # invariance preserved through the wrapper).
        from td_ludo.models.v13_5_production import V135ProductionAdapter
        from td_ludo.game.encoder_v18_production import (
            encode_state_v18_production, V18_PROD_CHANNELS,
        )
        v135_blocks = args.num_res_blocks if args.num_res_blocks != 4 else 10
        v135_channels = args.num_channels if args.num_channels != 96 else 128
        model_factory = lambda: V135ProductionAdapter(
            num_res_blocks=v135_blocks, num_channels=v135_channels,
        )
        model = model_factory()
        print(f"[V12 Train] Model: V135ProductionAdapter (V13.5) "
              f"({model.count_parameters():,} params)")
        print(f"[V12 Train]   {v135_blocks} ResBlocks × {v135_channels}ch, "
              f"{V18_PROD_CHANNELS}ch packed input (V18 base + rank masks + "
              f"token_to_rank), rank-indexed inner output")
        encoder_fn = encode_state_v18_production
    elif args.model_arch == 'v131_aux':
        # V13.1: MinimalCNN14Aux (12×160 default, 14ch raw + 3 aux heads).
        # During RL we use the same forward_policy_only path as V13 — aux
        # heads exist in state_dict but are not exercised.
        from td_ludo.models.v13_1 import MinimalCNN14Aux
        v131_blocks = args.num_res_blocks if args.num_res_blocks != 4 else 12
        v131_channels = args.num_channels if args.num_channels != 96 else 160
        model_factory = lambda: MinimalCNN14Aux(
            num_res_blocks=v131_blocks,
            num_channels=v131_channels,
            in_channels=14,
        )
        model = model_factory()
        print(f"[V12 Train] Model: MinimalCNN14Aux (V13.1) "
              f"({model.count_parameters():,} params)")
        print(f"[V12 Train]   {v131_blocks} ResBlocks × {v131_channels}ch, "
              f"14ch raw input + 3 aux heads (frozen during RL)")
        import td_ludo_cpp as _ludo
        encoder_fn = _ludo.encode_state_v14_minimal
    else:
        # V12.1 / V12.2: V11 encoder (33ch), CNN + token attention.
        model_factory = lambda: AlphaLudoV12(
            num_res_blocks=args.num_res_blocks,
            num_channels=args.num_channels,
            num_attn_layers=args.num_attn_layers,
            num_heads=args.num_heads,
            ffn_ratio=args.ffn_ratio,
            dropout=args.dropout,  # 0.0 for RL
            in_channels=33,  # V12.1: V10 (28) + idle (4) + streak (1)
        )
        model = model_factory()
        print(f"[V12 Train] Model: AlphaLudoV12 ({model.count_parameters():,} params)")
        print(f"[V12 Train]   {args.num_res_blocks} ResBlocks × {args.num_channels}ch + "
              f"{args.num_attn_layers} Attn layers × {args.num_heads} heads")
        print(f"[V12 Train]   dropout={args.dropout} (RL must be 0.0)")
        encoder_fn = None  # use player default (encode_state_v11)
    model.to(device)

    # Trainer (V10 trainer works unchanged — same forward signature)
    # Exp 24: pass alpha_search to enable the auxiliary loss term. 0.0 when
    # --search-enabled is not set, so the loss is identical to V12.2 baseline.
    alpha_search_eff = args.alpha_search if args.search_enabled else 0.0
    trainer = ActorCriticTrainerV10(
        model, device, learning_rate=LEARNING_RATE,
        alpha_search=alpha_search_eff,
        progress_coeff=args.progress_coeff,
    )

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

    # V12.2: opponent-mix preset override.
    # `_random_composition` looks up GAME_COMPOSITION from its module's namespace
    # at call time, so we can safely monkey-patch already-imported modules.
    if args.game_composition == 'v122':
        V122_MIX = {
            "SelfPlay":   0.75,
            "Expert":     0.15,
            "Heuristic":  0.05,
            "Aggressive": 0.03,
            "Defensive":  0.02,
        }
        import src.config as _cfg
        _cfg.GAME_COMPOSITION = V122_MIX
        import td_ludo.game.players.v11 as _v11mod
        _v11mod.GAME_COMPOSITION = V122_MIX
        print(f"[V12 Train] Game composition: V12.2 mix → {V122_MIX}")
    elif args.game_composition == 'v123':
        # Historical-model opponents replace the saturated bot mix.
        # Each Hist_* tag is dispatched at play_step time through
        # OpponentRegistry, which loads the right architecture + encoder.
        # Curriculum mix prioritises the strongest historicals (V11, V10)
        # since those produce competitive games against V12.2; V6.x are
        # included for *style diversity* (different defect profiles), not
        # competitive challenge.
        V123_MIX = {
            "SelfPlay":    0.67,    # main self-play + ghost (unchanged)
            "Hist_V10":    0.18,    # strongest available historical
            "Hist_V6_3":   0.08,    # bonus-turn-aware older model
            "Hist_V6_1":   0.04,    # base V6 (no bonus-turn channel)
            "Hist_V6_big": 0.03,    # old V5-era 17ch model
        }
        # Note: NO Random in the mix — trained models all crush Random
        # ~95%, so games against it carry zero gradient signal and
        # 2% × any meaningful run = thousands of wasted games. The
        # historical-opponent WRs (especially Hist_V10) act as the real
        # collapse-detector if the policy ever degrades.
        # Note: V11 (token-attention) and V6.2 (temporal transformer)
        # are intentionally absent — see opponent_registry.py for why.
        import src.config as _cfg
        _cfg.GAME_COMPOSITION = V123_MIX
        import td_ludo.game.players.v11 as _v11mod
        _v11mod.GAME_COMPOSITION = V123_MIX
        print(f"[V12 Train] Game composition: V12.3 mix → {V123_MIX}")
    elif args.game_composition == 'v122_hist':
        # v122 mix + small dose of historical opponents.
        # Bots stay (V13 trains well against them), small hist exposure
        # gives "stronger-than-bot" gradient signal. Hist_V12_2 included
        # for ceiling reference (~peer to V13). Hist_V10/V6_3/V6_1 give
        # "should-be-winning" gradient (V13 ≥ V10 ≥ V6_3 ≥ V6_1 in
        # tournament data: 56.9 / 49.0 / 44.5 / 40.9 % aggregate WR).
        V122_HIST_MIX = {
            "SelfPlay":    0.60,
            "Expert":      0.15,
            "Heuristic":   0.05,
            "Aggressive":  0.03,
            "Defensive":   0.02,
            "Hist_V12_2":  0.05,
            "Hist_V10":    0.05,
            "Hist_V6_3":   0.03,
            "Hist_V6_1":   0.02,
        }
        import src.config as _cfg
        _cfg.GAME_COMPOSITION = V122_HIST_MIX
        import td_ludo.game.players.v11 as _v11mod
        _v11mod.GAME_COMPOSITION = V122_HIST_MIX
        print(f"[V12 Train] Game composition: V12.2 + hist mix → {V122_HIST_MIX}")
    elif args.game_composition == 'v122_hist_v2':
        # Stronger opponent mix for the bias-penalty V12.2 lineage.
        # Doubles historical share (15% → 30%), trims bot share to 10%.
        # H2H tracking goal: V12.2-bias-vN beats V12.2-pre-search by 5pp+.
        # Increases per-game inference cost (hist models run on CPU).
        V122_HIST_V2_MIX = {
            "SelfPlay":    0.60,
            "Expert":      0.07,
            "Heuristic":   0.03,
            "Hist_V12_2":  0.15,
            "Hist_V10":    0.10,
            "Hist_V6_3":   0.03,
            "Hist_V6_1":   0.02,
        }
        import src.config as _cfg
        _cfg.GAME_COMPOSITION = V122_HIST_V2_MIX
        import td_ludo.game.players.v11 as _v11mod
        _v11mod.GAME_COMPOSITION = V122_HIST_V2_MIX
        print(f"[V12 Train] Game composition: V12.2 + hist v2 mix → {V122_HIST_V2_MIX}")
    elif args.game_composition == 'v13':
        # V13-tuned mix: V12.2 added as strong external opponent
        # (Distill14-vs-V12.2 plays ~50/50, every game is a real test).
        # Self-play reduced from v123's 67% → 55% to limit the
        # over-fit-to-self failure mode V12.2 hit late in training.
        # Bots dropped entirely (saturated, all ~50% vs each other).
        V13_MIX = {
            "SelfPlay":    0.55,    # main self-play + ghost
            "Hist_V12_2":  0.20,    # strongest external (V12.2 final)
            "Hist_V10":    0.15,    # 2nd strongest external
            "Hist_V6_3":   0.10,    # diversity / older bonus-turn model
        }
        import src.config as _cfg
        _cfg.GAME_COMPOSITION = V13_MIX
        import td_ludo.game.players.v11 as _v11mod
        _v11mod.GAME_COMPOSITION = V13_MIX
        print(f"[V12 Train] Game composition: V13 mix → {V13_MIX}")
    elif args.game_composition == 'v13_5_no_bots':
        # V13.5-tuned mix: target DNA diversity by mixing in models from
        # different lineages (V13.2/V12.2 = "old DNA from teacher chain";
        # Hist_V13_5_SL = "different SL endpoint of same architecture";
        # SelfPlay+ghost = "current student / past selves"). NO bots —
        # they're saturated and don't push the policy past V12.2 lineage.
        V13_5_NO_BOTS_MIX = {
            # Phase L (2026-05-13): "tough-opponents" rebalance after search-
            # teacher experiment damaged the model. Less self-play (50 → 20)
            # so the policy doesn't drift into self-overfit; more weight on
            # the toughest external opponents (V13_2 and V13_5_SL) to
            # extract whatever discriminative signal remains in the pool.
            "Hist_V13_2":       0.40,    # tough — V13-line, different arch family
            "Hist_V13_5_SL":    0.30,    # toughest (different SL endpoint, same arch)
            "SelfPlay":         0.20,    # cut from 0.50 — less self-overfit
            "Hist_V12_2":       0.10,    # legacy, kept for DNA diversity
        }
        import src.config as _cfg
        _cfg.GAME_COMPOSITION = V13_5_NO_BOTS_MIX
        import td_ludo.game.players.v11 as _v11mod
        _v11mod.GAME_COMPOSITION = V13_5_NO_BOTS_MIX
        print(f"[V12 Train] Game composition: v13_5_no_bots mix → {V13_5_NO_BOTS_MIX}")
    else:
        from src.config import GAME_COMPOSITION as _gc_default
        print(f"[V12 Train] Game composition: PROD default → {_gc_default}")

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
        from evaluate_v11 import evaluate_model  # V12.1+: 33ch V11 encoder
        results = evaluate_model(model, device, num_games=500, verbose=True, encoder_fn=encoder_fn)
        print(f"\nWin Rate: {results['win_rate_percent']}%")
        return

    if not args.no_dashboard:
        start_dashboard_server(port=args.port)

    # Initial composition uses what the user passed; curriculum mode also
    # enables historicals upfront so the registry is loaded before the swap.
    historical_opponents_enabled = (
        args.game_composition in ('v123', 'v13', 'v122_hist', 'v122_hist_v2', 'v13_5_no_bots')
        or (args.curriculum_mode == 'auto'
            and args.curriculum_target in ('v123', 'v13', 'v122_hist', 'v122_hist_v2', 'v13_5_no_bots'))
    )
    # Progress aux is only meaningful for V13.5 (it has the per-rank progress
    # head). For other archs the player still passes the kwarg but the trainer
    # ignores `progress_target` since the model doesn't return a 4th tensor.
    progress_target_enabled_eff = (
        args.progress_coeff > 0.0 and args.model_arch == 'v13_5'
    )
    player = VectorACGamePlayer(
        trainer, BATCH_SIZE, device,
        model_factory=model_factory,
        elo_tracker=elo_tracker,
        search_enabled=args.search_enabled,
        search_target_fraction=args.search_target_fraction,
        search_label_smoothing=args.search_label_smoothing,
        historical_opponents_enabled=historical_opponents_enabled,
        encoder_fn=encoder_fn,
        shaping_coeff=args.shaping_coeff,
        progress_target_enabled=progress_target_enabled_eff,
    )
    _player = player

    if args.search_enabled:
        print(f"[V12 Train] Exp 24 search-during-training: ENABLED "
              f"(fraction={args.search_target_fraction}, "
              f"alpha={alpha_search_eff}, "
              f"label_smoothing={args.search_label_smoothing})")
    else:
        print("[V12 Train] Exp 24 search-during-training: DISABLED "
              "(use --search-enabled to turn on)")

    if historical_opponents_enabled:
        print(f"[V12 Train] Historical opponents: ENABLED "
              f"(tags: {player.opp_registry.available_tags()})")
    else:
        print("[V12 Train] Historical opponents: DISABLED "
              "(use --game-composition v123 to turn on)")

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

    # Curriculum-mode tracking. eval_wr_history collects all evals this
    # session; curriculum_swapped is one-shot — once we swap we don't keep
    # checking. Persist nothing across restarts (intentional: if you resume
    # past the swap you should manually pass the post-swap composition).
    eval_wr_history = []
    curriculum_swapped = False
    if args.curriculum_mode == 'auto':
        print(f"[V12 Train] Curriculum: ENABLED. Will swap "
              f"{args.game_composition} → {args.curriculum_target} after "
              f"{args.curriculum_window} consecutive evals ≥ "
              f"{args.curriculum_eval_thresh:.0%}.")

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
                from evaluate_v11 import evaluate_model  # V12.1+: 33ch V11 encoder

                eval_results = evaluate_model(model, device, num_games=EVAL_GAMES, verbose=False, encoder_fn=encoder_fn)
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

                # Curriculum gating: swap composition once last N evals all clear threshold.
                if args.curriculum_mode == 'auto' and not curriculum_swapped:
                    eval_wr_history.append(eval_wr)
                    recent = eval_wr_history[-args.curriculum_window:]
                    if (len(recent) >= args.curriculum_window
                            and all(w >= args.curriculum_eval_thresh for w in recent)):
                        if apply_composition(args.curriculum_target):
                            curriculum_swapped = True
                            mix = get_composition_mix(args.curriculum_target)
                            print(f"\n  ★★★ CURRICULUM TRIGGER: swapping "
                                  f"{args.game_composition} → {args.curriculum_target} ★★★")
                            print(f"      Last {args.curriculum_window} evals: "
                                  f"{[f'{w:.1%}' for w in recent]}")
                            print(f"      New mix: {mix}\n")
                        else:
                            print(f"  [curriculum] FAILED: unknown target "
                                  f"'{args.curriculum_target}'")

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
