"""Online Supervised Learning (SL) Distillation for AlphaLudo 4-Player Model.

Key Principles:
1. Live in-memory game streaming — ZERO transitions stored on disk (zero disk clutter).
2. Hardware accelerated on Mac via Apple Silicon MPS (or CPU fallback).
3. Live HTTP dashboard on http://localhost:8844 serving dashboard_4p.html.
4. Minimal disk storage: only model_latest.pt and model_best.pt are kept (~10 MB total).
"""
from __future__ import annotations

import argparse
import collections
import http.server
import json
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Path resolution
_REPO_ROOT = Path(__file__).resolve().parent.parent
_V15_ROOT = _REPO_ROOT / "td_ludo_v15"
_TD_ROOT = _REPO_ROOT / "td_ludo"

for p in (str(_V15_ROOT), str(_TD_ROOT), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import td_ludo_cpp as cpp
from td_ludo.game.heuristic_bot import ExpertBot, HeuristicLudoBot
from td_ludo.game.strong_bots import ExpectimaxBot
from td_ludo.game.strong_bots_v2 import AggressiveExpectimaxBot

from td_ludo_v15.game.cells import (
    BOARD_SIZE,
    NUM_BOARD_CELLS,
    cell_to_index,
    index_to_cell,
    position_to_cell_in_pov,
)
from td_ludo_v15.game.encoder_4p import encode_frame_4p
from td_ludo_v15.models.v15_4p import V15_4P_GraphTransformer

_BASE_POS = -1
_HOME_POS = 99


# ─── 1. In-Memory Transition Buffer ──────────────────────────────────────────

class Transition:
    __slots__ = ("frame", "legal_mask", "target_cell", "relative_winner")

    def __init__(self, frame: np.ndarray, legal_mask: np.ndarray, target_cell: int, relative_winner: int):
        self.frame = frame                        # (15, 15, 5) int8
        self.legal_mask = legal_mask              # (225,) float32
        self.target_cell = target_cell            # int (0..224)
        self.relative_winner = relative_winner    # int (0..3)


# ─── 2. Bot Game Generator (Runs in background thread) ────────────────────────

def create_bot_instance(bot_name: str, player_id: int):
    if bot_name == "Expert":
        return ExpertBot(player_id=player_id)
    if bot_name == "Heuristic":
        return HeuristicLudoBot(player_id=player_id)
    if bot_name == "Expectimax":
        return ExpectimaxBot(player_id=player_id)
    if bot_name == "AggressiveExpectimax":
        return AggressiveExpectimaxBot(player_id=player_id)
    return ExpertBot(player_id=player_id)


def play_one_game_transitions(bot_names: List[str]) -> Tuple[List[dict], int]:
    """Simulate one 4-player game among bots. Returns (trajectory, winner_seat)."""
    bots = [create_bot_instance(name, p) for p, name in enumerate(bot_names)]
    state = cpp.create_initial_state()
    consec_sixes = [0, 0, 0, 0]
    mc = 0
    trajectory: List[dict] = []

    while not state.is_terminal and mc < 800:
        cp = int(state.current_player)

        # Handle dice roll
        if state.current_dice_roll == 0:
            d = random.randint(1, 6)
            if d == 6:
                consec_sixes[cp] += 1
                if consec_sixes[cp] >= 3:
                    consec_sixes[cp] = 0
                    state.current_player = (cp + 1) % 4
                    state.current_dice_roll = 0
                    continue
            else:
                consec_sixes[cp] = 0
            state.current_dice_roll = d

        legal = cpp.get_legal_moves(state)
        if not legal:
            state.current_player = (cp + 1) % 4
            state.current_dice_roll = 0
            continue

        # Compute legal source mask (225,)
        legal_mask = np.zeros(NUM_BOARD_CELLS, dtype=np.float32)
        for t in legal:
            pos = int(state.player_positions[cp][t])
            c = position_to_cell_in_pov(_BASE_POS if pos == _BASE_POS else pos, cp, cp)
            legal_mask[cell_to_index(*c)] = 1.0

        # Encode state in mover's POV
        frame = encode_frame_4p(state, pov_player=cp)

        # Bot selects token
        action = bots[cp].select_move(state, list(legal))

        # Target source cell
        chosen_pos = int(state.player_positions[cp][action])
        chosen_c = position_to_cell_in_pov(_BASE_POS if chosen_pos == _BASE_POS else chosen_pos, cp, cp)
        target_cell = cell_to_index(*chosen_c)

        trajectory.append({
            "frame": frame,
            "legal_mask": legal_mask,
            "target_cell": target_cell,
            "mover": cp,
        })

        # Apply move
        state = cpp.apply_move(state, int(action))
        mc += 1

    winner = int(cpp.get_winner(state)) if state.is_terminal else -1
    return trajectory, winner


# ─── 3. Background Data Generation Worker ─────────────────────────────────────

def data_generator_worker(
    buffer: collections.deque,
    lock: threading.Lock,
    stop_event: threading.Event,
    stats_dict: dict,
    max_buffer_size: int = 30000,
):
    """Continuously generates 4P games and pushes transitions to RAM buffer."""
    teacher_pool = ["Expert", "Heuristic", "Expectimax", "AggressiveExpectimax"]

    while not stop_event.is_set():
        with lock:
            lag = stats_dict.get("states_generated", 0) - stats_dict.get("total_states", 0)
        if lag >= max_buffer_size:
            time.sleep(0.005)
            continue

        # Shuffle bots among seats for variance
        bot_assignment = random.sample(teacher_pool, 4)
        traj, winner = play_one_game_transitions(bot_assignment)

        if winner >= 0:
            with lock:
                stats_dict["games_generated"] = stats_dict.get("games_generated", 0) + 1
                stats_dict["states_generated"] = stats_dict.get("states_generated", 0) + len(traj)
                for step in traj:
                    rel_winner = (winner - step["mover"]) % 4
                    buffer.append(
                        Transition(
                            frame=step["frame"],
                            legal_mask=step["legal_mask"],
                            target_cell=step["target_cell"],
                            relative_winner=rel_winner,
                        )
                    )


# ─── 4. Evaluation Function ───────────────────────────────────────────────────

def evaluate_student_vs_bots(
    model: nn.Module,
    device: torch.device,
    n_games: int = 40,
) -> float:
    """Evaluate student model playing against 3 bots (Expert, Heuristic, AggressiveExpectimax).
    
    Student rotates equally across all 4 seats: 0, 1, 2, 3.
    Returns win rate in percent (0..100).
    """
    model.eval()
    bot_names = ["Expert", "Heuristic", "AggressiveExpectimax"]
    student_wins = 0

    with torch.no_grad():
        for g in range(n_games):
            student_seat = g % 4
            # Assign other seats to bots
            bots = {}
            b_idx = 0
            for s in range(4):
                if s != student_seat:
                    bots[s] = create_bot_instance(bot_names[b_idx % len(bot_names)], s)
                    b_idx += 1

            state = cpp.create_initial_state()
            consec_sixes = [0, 0, 0, 0]
            mc = 0

            while not state.is_terminal and mc < 600:
                cp = int(state.current_player)

                if state.current_dice_roll == 0:
                    d = random.randint(1, 6)
                    if d == 6:
                        consec_sixes[cp] += 1
                        if consec_sixes[cp] >= 3:
                            consec_sixes[cp] = 0
                            state.current_player = (cp + 1) % 4
                            state.current_dice_roll = 0
                            continue
                    else:
                        consec_sixes[cp] = 0
                    state.current_dice_roll = d

                legal = cpp.get_legal_moves(state)
                if not legal:
                    state.current_player = (cp + 1) % 4
                    state.current_dice_roll = 0
                    continue

                if cp == student_seat:
                    # Model move
                    frame = encode_frame_4p(state, pov_player=cp)
                    x_t = torch.from_numpy(frame).unsqueeze(0).to(device)  # (1, 15, 15, 5)

                    legal_mask = np.zeros(NUM_BOARD_CELLS, dtype=np.float32)
                    for t in legal:
                        pos = int(state.player_positions[cp][t])
                        c = position_to_cell_in_pov(_BASE_POS if pos == _BASE_POS else pos, cp, cp)
                        legal_mask[cell_to_index(*c)] = 1.0
                    mask_t = torch.from_numpy(legal_mask).unsqueeze(0).to(device)

                    policy, _ = model(x_t, legal_mask=mask_t)
                    best_cell_idx = int(policy[0].argmax().item())
                    chosen_cell = index_to_cell(best_cell_idx)

                    # Map chosen cell to a legal token
                    chosen_token = legal[0]
                    for t in legal:
                        pos = int(state.player_positions[cp][t])
                        c = position_to_cell_in_pov(_BASE_POS if pos == _BASE_POS else pos, cp, cp)
                        if c == chosen_cell:
                            chosen_token = t
                            break
                    action = chosen_token
                else:
                    action = bots[cp].select_move(state, list(legal))

                state = cpp.apply_move(state, int(action))
                mc += 1

            if state.is_terminal and int(cpp.get_winner(state)) == student_seat:
                student_wins += 1

    model.train()
    return (student_wins / n_games) * 100.0


# ─── 5. Dashboard HTTP Server ─────────────────────────────────────────────────

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, stats_getter=None, metrics_getter=None, html_path=None, **kwargs):
        self.stats_getter = stats_getter
        self.metrics_getter = metrics_getter
        self.html_path = html_path
        super().__init__(*args, **kwargs)

    def log_message(self, *args):
        return  # silence terminal log clutter

    def do_GET(self):
        if self.path in ("/", ""):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open(self.html_path, "rb") as f:
                self.wfile.write(f.read())
            return

        if self.path == "/api/stats":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            data = json.dumps(self.stats_getter()).encode()
            self.wfile.write(data)
            return

        if self.path == "/api/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            data = json.dumps(self.metrics_getter()).encode()
            self.wfile.write(data)
            return

        self.send_response(404)
        self.end_headers()


def start_dashboard_server(stats_getter, metrics_getter, html_path: str, port: int = 8844):
    handler = lambda *args, **kwargs: DashboardHandler(
        *args, stats_getter=stats_getter, metrics_getter=metrics_getter, html_path=html_path, **kwargs
    )
    server = http.server.HTTPServer(("0.0.0.0", port), handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


# ─── 6. Main Online SL Training Loop ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AlphaLudo 4P Online SL Distillation")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--target-states", type=int, default=1_500_000, help="Target decision states to train on")
    parser.add_argument("--buffer-size", type=int, default=25000, help="Max transitions stored in RAM ring buffer")
    parser.add_argument("--eval-interval", type=int, default=300, help="Batches between live bot evaluations")
    parser.add_argument("--eval-games", type=int, default=40, help="Games per evaluation round")
    parser.add_argument("--port", type=int, default=8844, help="Dashboard port")
    parser.add_argument("--device", default=None, help="Device (mps, cpu, cuda)")
    parser.add_argument("--save-dir", default="checkpoints/v15_4p_sl", help="Directory for models")
    parser.add_argument("--resume", action="store_true", help="Resume from save-dir/model_latest.pt if exists")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of concurrent game generator threads")
    args = parser.parse_args()

    # Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"🚀 AlphaLudo 4-Player Online SL Distillation")
    print(f"💻 Device: {device} | Batch Size: {args.batch_size} | Buffer RAM Cap: {args.buffer_size}")
    print(f"🌐 Dashboard will be live at: http://localhost:{args.port}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize 4P GraphTransformer with fresh weights (or resume)
    model = V15_4P_GraphTransformer(d_model=128, n_heads=4, n_layers=4, ffn_dim=256)
    if args.resume and (save_dir / "model_latest.pt").exists():
        print(f"🔄 Resuming weights from {save_dir / 'model_latest.pt'}")
        st = torch.load(save_dir / "model_latest.pt", map_location=device, weights_only=True)
        model.load_state_dict(st)
    else:
        print("🌱 Initializing with fresh random weights")
    model.to(device)
    print(f"🧠 Model Initialized: {model.count_parameters():,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Shared buffer and state
    buffer: collections.deque = collections.deque(maxlen=args.buffer_size)
    lock = threading.Lock()
    stop_event = threading.Event()

    stats = {
        "phase": "ONLINE SL",
        "device": str(device),
        "total_states": 0,
        "states_generated": 0,
        "batches": 0,
        "games_generated": 0,
        "games_played": 0,
        "policy_loss": 0.0,
        "policy_acc": 0.0,
        "val_loss": 0.0,
        "val_acc": 0.0,
        "eval_wr": 25.0,
        "best_eval_wr": 25.0,
        "gpm": 0.0,
        "fps": 0.0,
        "lr": args.lr,
        "elapsed": "00:00:00",
    }
    metrics_history: List[dict] = []

    # Start dashboard server
    html_file = _V15_ROOT / "td_ludo_v15" / "rich" / "dashboard_4p.html"
    start_dashboard_server(lambda: stats, lambda: metrics_history, str(html_file), port=args.port)
    print(f"📊 Dashboard server started at http://localhost:{args.port}")

    # Start game generator threads
    threads = []
    for _ in range(args.num_workers):
        t = threading.Thread(
            target=data_generator_worker,
            args=(buffer, lock, stop_event, stats, args.buffer_size),
            daemon=True,
        )
        t.start()
        threads.append(t)
    print(f"⏳ Warming up in-memory buffer with {args.num_workers} parallel bot generator threads...")

    # Wait for warm-up
    while len(buffer) < 2000:
        time.sleep(0.5)
        print(f"   Buffer: {len(buffer)} / 2000 transitions...", end="\r")
    print(f"\n✅ Warm-up complete! Buffer ready with {len(buffer)} transitions. Starting training...")

    t_start = time.time()
    t_last_stat = time.time()
    states_last_stat = 0
    games_last_stat = 0

    best_wr = 25.0
    total_states_trained = 0
    batch_idx = 0

    try:
        while total_states_trained < args.target_states:
            # Sample batch from buffer
            with lock:
                if len(buffer) < args.batch_size:
                    time.sleep(0.01)
                    continue
                indices = [random.randint(0, len(buffer) - 1) for _ in range(args.batch_size)]
                samples = [buffer[idx] for idx in indices]

            batch_x = np.stack([s.frame for s in samples])            # (B, 15, 15, 5)
            batch_mask = np.stack([s.legal_mask for s in samples])     # (B, 225)
            batch_target = np.array([s.target_cell for s in samples], dtype=np.int64)
            batch_winner = np.array([s.relative_winner for s in samples], dtype=np.int64)

            x_t = torch.from_numpy(batch_x).to(device)
            mask_t = torch.from_numpy(batch_mask).to(device)
            target_t = torch.from_numpy(batch_target).to(device)
            winner_t = torch.from_numpy(batch_winner).to(device)

            optimizer.zero_grad(set_to_none=True)
            policy, value, pol_logits, val_logits = model(x_t, legal_mask=mask_t, return_logits=True)

            loss_pol = F.cross_entropy(pol_logits, target_t)
            loss_val = F.cross_entropy(val_logits, winner_t)
            total_loss = loss_pol + 0.5 * loss_val

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Metrics
            with torch.no_grad():
                pol_acc = (pol_logits.argmax(dim=-1) == target_t).float().mean().item()
                val_acc = (val_logits.argmax(dim=-1) == winner_t).float().mean().item()

            total_states_trained += args.batch_size
            batch_idx += 1

            # Update stats
            now = time.time()
            elapsed_sec = int(now - t_start)
            stats["total_states"] = total_states_trained
            stats["batches"] = batch_idx
            stats["policy_loss"] = float(loss_pol.item())
            stats["policy_acc"] = float(pol_acc)
            stats["val_loss"] = float(loss_val.item())
            stats["val_acc"] = float(val_acc)
            stats["elapsed"] = time.strftime("%H:%M:%S", time.gmtime(elapsed_sec))

            # Update rates every 2 seconds
            if now - t_last_stat >= 2.0:
                dt = now - t_last_stat
                d_states = total_states_trained - states_last_stat
                d_games = stats["games_generated"] - games_last_stat
                stats["fps"] = d_states / dt
                stats["gpm"] = (d_games / dt) * 60.0
                t_last_stat = now
                states_last_stat = total_states_trained
                games_last_stat = stats["games_generated"]

            # Periodic Live Bot Evaluation
            if batch_idx % args.eval_interval == 0:
                eval_wr = evaluate_student_vs_bots(model, device, n_games=args.eval_games)
                stats["eval_wr"] = eval_wr
                if eval_wr > best_wr:
                    best_wr = eval_wr
                    stats["best_eval_wr"] = best_wr
                    torch.save(model.state_dict(), save_dir / "model_best.pt")
                    print(f"\n🌟 NEW BEST EVAL WR: {best_wr:.1f}% (saved to model_best.pt)!")

                # Always save latest
                torch.save(model.state_dict(), save_dir / "model_latest.pt")

                metrics_history.append({
                    "batches": batch_idx,
                    "states": total_states_trained,
                    "policy_loss": round(stats["policy_loss"], 4),
                    "policy_acc": round(stats["policy_acc"], 4),
                    "value_loss": round(stats["val_loss"], 4),
                    "eval_wr": round(eval_wr, 1),
                    "best_eval_wr": round(best_wr, 1),
                })
                # Keep history trimmed in RAM
                if len(metrics_history) > 300:
                    metrics_history.pop(0)

                print(
                    f"[{stats['elapsed']}] Batch {batch_idx:5d} | "
                    f"States: {total_states_trained:,} | "
                    f"Pol Acc: {pol_acc * 100:4.1f}% | "
                    f"Val Acc: {val_acc * 100:4.1f}% | "
                    f"Eval WR: {eval_wr:4.1f}% (Best: {best_wr:4.1f}%)"
                )

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user.")
    finally:
        stop_event.set()
        torch.save(model.state_dict(), save_dir / "model_latest.pt")
        print(f"💾 Checkpoint saved to {save_dir / 'model_latest.pt'}")
        print("Done!")


if __name__ == "__main__":
    main()
