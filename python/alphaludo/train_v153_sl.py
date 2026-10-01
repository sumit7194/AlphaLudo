"""AlphaLudo V15.3 — Supervised Learning from Rust Search Teacher.

Trains a V15 GraphTransformer (history_len=1, ~588K params) tabula rasa from
random weights on 1 Million games of high-depth teacher play (Expecti-MCTS N=3,000
vs Depth-2 Expectimax).

Features:
  - Random weights initialization (tabula rasa).
  - Masked cross-entropy on 225-cell source-cell policy.
  - Balanced MSE on terminal outcome (+1.0 / -1.0).
  - Streaming shard loader: consumes new binary shards as Rust produces them.
  - Periodic head-to-head evaluation against bot suite (Heuristic, Aggressive, Expert, MCTS).
  - Live rich interactive web dashboard on port 8790.
  - Clean interruptability via 'stop' file.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path
import random
import sys
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
TD_LUDO_DIR = ROOT_DIR / "td_ludo"
PYTHON_DIR = ROOT_DIR / "python"
V15_ROOT = ROOT_DIR / "td_ludo_v15"
for p in (str(TD_LUDO_DIR), str(PYTHON_DIR), str(V15_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import alphaludo_rs
import td_ludo_cpp as ludo_cpp
import td_ludo_v15_cpp as v15_cpp
from sl_dataset_v153 import V153TeacherDataset
from td_ludo.game.heuristic_bot import get_bot
from td_ludo_v15.game.cells import NUM_BOARD_CELLS, cell_to_index, position_to_cell_in_pov
from td_ludo_v15.game.encoder import encode_frame
from td_ludo_v15.models.v15 import V15GraphTransformer

_BASE_POS = v15_cpp.BASE_POS


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dashboard HTTP Server
# ─────────────────────────────────────────────────────────────────────────────
class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, stats_path=None, metrics_path=None, landing="v13_dashboard.html", **kw):
        self._stats_path = stats_path
        self._metrics_path = metrics_path
        self._landing = landing
        super().__init__(*args, directory=directory, **kw)

    def log_message(self, *a, **kw):
        return

    def do_GET(self):
        try:
            if self.path in ("/", ""):
                self.path = "/" + self._landing
                return super().do_GET()
            if self.path == "/api/stats":
                return self._serve_json_file(self._stats_path)
            if self.path == "/api/metrics":
                return self._serve_json_file(self._metrics_path)
            if self.path == "/api/elo":
                return self._send_json(b'{"rankings": [], "history": {}}')
            if self.path.startswith("/api/games"):
                return self._send_json(b'{"games": []}')
            if self.path == "/api/system":
                return self._serve_system()
            return super().do_GET()
        except (ConnectionResetError, BrokenPipeError):
            pass

    def _serve_json_file(self, path):
        if not path or not os.path.exists(path):
            self.send_response(404)
            self.end_headers()
            return
        try:
            with open(path) as f:
                data = f.read()
        except OSError:
            self.send_response(500)
            self.end_headers()
            return
        self._send_json(data.encode())

    def _serve_system(self):
        try:
            import psutil
            payload = {
                "cpu_percent": float(psutil.cpu_percent(interval=None)),
                "memory_percent": float(psutil.virtual_memory().percent),
                "pid": os.getpid(),
            }
        except Exception:
            payload = {"cpu_percent": 0.0, "memory_percent": 0.0, "pid": os.getpid()}
        self._send_json(json.dumps(payload).encode())

    def _send_json(self, data: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def start_dashboard(port: int, stats_path: str, metrics_path: str, dashboard_dir: str):
    landing = "v13_dashboard.html"
    handler = functools.partial(
        DashboardHandler,
        directory=dashboard_dir,
        stats_path=stats_path,
        metrics_path=metrics_path,
        landing=landing,
    )
    server = HTTPServer(("0.0.0.0", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[Dashboard] Server live at http://localhost:{port}/{landing}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Benchmark Evaluation (Zero-Search Greedy Policy)
# ─────────────────────────────────────────────────────────────────────────────
class RustMCTSBot:
    def __init__(self, player_id: int, num_simulations: int = 500):
        self.player_id = player_id
        self.num_simulations = num_simulations

    def select_move(self, state, legal_moves):
        if not legal_moves:
            return 0
        if len(legal_moves) == 1:
            return legal_moves[0]
        pos = np.ascontiguousarray(state.player_positions, dtype=np.int8)
        act = alphaludo_rs.select_mcts_move_2p(
            pos,
            int(self.player_id if self.player_id is not None else state.current_player),
            int(state.current_dice_roll),
            self.num_simulations,
        )
        if act not in legal_moves:
            return legal_moves[0]
        return act


def evaluate_model(model: nn.Module, device: torch.device, games_per_bot: int = 15):
    """Evaluates greedy model policy (zero search) with cyclic seat rotation."""
    model.eval()
    bot_names = ["Heuristic", "Aggressive", "Expert", "MCTS"]
    results = {}
    total_wins = 0
    total_games = 0

    for bot_name in bot_names:
        bot_wins = 0
        for g_idx in range(games_per_bot):
            model_seat = 0 if (g_idx % 2 == 0) else 2
            opp_seat = 2 if model_seat == 0 else 0

            if bot_name == "MCTS":
                bot = RustMCTSBot(player_id=opp_seat, num_simulations=500)
            else:
                bot = get_bot(bot_name, player_id=opp_seat)

            state = ludo_cpp.create_initial_state_2p()
            move_count = 0

            while not state.is_terminal and move_count < 350:
                cp = int(state.current_player)
                d = random.randint(1, 6)
                state.current_dice_roll = d

                legal = ludo_cpp.get_legal_moves(state)
                if not legal:
                    nxt = 2 if cp == 0 else 0
                    state.current_player = nxt
                    state.current_dice_roll = 0
                    continue

                if cp == model_seat:
                    if len(legal) == 1:
                        action = legal[0]
                    else:
                        v15_x = np.zeros((1, 15, 15, 3), dtype=np.float32)
                        v15_x[0] = encode_frame(state, pov_player=cp)
                        v15_legal = np.zeros(NUM_BOARD_CELLS, dtype=np.float32)
                        legal_cells = []
                        for t in legal:
                            pos = int(state.player_positions[cp][t])
                            c = position_to_cell_in_pov(_BASE_POS if pos == _BASE_POS else pos, cp, cp)
                            v15_legal[cell_to_index(*c)] = 1.0
                            legal_cells.append((t, c))
                        with torch.no_grad():
                            xt = torch.from_numpy(v15_x).unsqueeze(0).to(device)
                            mt = torch.from_numpy(v15_legal).unsqueeze(0).to(device)
                            pol, _ = model(xt, mt)
                            chosen_idx = int(pol.argmax(dim=-1).item())
                        chosen_cell = divmod(chosen_idx, 15)
                        action = legal[0]
                        for t, c in legal_cells:
                            if c == chosen_cell:
                                action = t
                                break
                else:
                    action = bot.select_move(state, list(legal))

                state = ludo_cpp.apply_move(state, int(action))
                move_count += 1

            if state.is_terminal and ludo_cpp.get_winner(state) == model_seat:
                bot_wins += 1

        wr = bot_wins / max(1, games_per_bot)
        results[bot_name] = wr
        total_wins += bot_wins
        total_games += games_per_bot

    overall_wr = total_wins / max(1, total_games)
    model.train()
    return overall_wr, results


# ─────────────────────────────────────────────────────────────────────────────
# 3. Main SL Training Loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=str, default="data/sl_teacher_v153")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/v15_3")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-every-steps", type=int, default=1000)
    parser.add_argument("--eval-games", type=int, default=60)
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    device = torch.device(args.device if torch.backends.mps.is_available() and args.device == "mps" else "cpu")
    print(f"[Device] Using {device}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    stats_path = str(ckpt_dir / "stats.json")
    metrics_path = str(ckpt_dir / "metrics.json")
    stop_file = ROOT_DIR / "stop"

    start_dashboard(port=args.port, stats_path=stats_path, metrics_path=metrics_path, dashboard_dir=str(TD_LUDO_DIR))

    # Initialize V15 GraphTransformer tabula rasa from random weights
    model = V15GraphTransformer(d_model=128, n_heads=4, n_layers=4, ffn_dim=256, history_len=1).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[Model] Initialized V15.3 Graph Transformer: {param_count:,} params (Random Weights)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Initialize stats
    stats_payload = {
        "run_info": {
            "model_name": "AlphaLudo V15.3 (Search-Distilled GraphTransformer)",
            "badge": "MCTS Teacher (N=3000) -> GraphTransformer",
            "arch_summary": f"V15.3 GraphTransformer (4x128) · {param_count:,} params · Zero Search",
            "description": "Distilling 1,000,000 games of MCTS (3,000 sims) vs Depth-2 Expectimax into GraphTransformer.",
            "target_wr": 0.85,
            "target_label": "Mastery Target (85% WR)",
            "baseline_wr": 0.50,
            "baseline_label": "Parity Baseline (50% WR)",
            "eval_games": args.eval_games,
        },
        "timestamp": time.time(),
        "total_games": 0,
        "total_states": 0,
        "games_per_minute": 0.0,
        "states_per_minute": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "total_loss": 0.0,
        "policy_entropy": 0.69,
        "temperature": 0.0,
        "main_elo": 1200,
        "eval_win_rate": 0.0,
        "opp_breakdown": {"Heuristic": 0.0, "Aggressive": 0.0, "Expert": 0.0, "MCTS": 0.0},
        "recent_opponent_stats": {},
        "elo_rankings": [
            {"name": "MCTSBot", "elo": 1600},
            {"name": "ExpertBot", "elo": 1500},
            {"name": "HeuristicBot", "elo": 1400},
            {"name": "AggressiveBot", "elo": 1350},
            {"name": "Model", "elo": 1200},
        ],
    }
    with open(stats_path, "w") as f:
        json.dump(stats_payload, f, indent=2)

    metrics_history = []
    total_states_trained = 0
    total_steps = 0
    best_eval_wr = 0.0

    print("\n" + "=" * 80)
    print("🚀 V15.3 GRAPH TRANSFORMER SL TRAINING PIPELINE LAUNCHED")
    print(f"   Architecture: GraphTransformer (~588K params, d=128, 4 layers)")
    print(f"   Teacher Data: Shards from {args.shard_dir}")
    print(f"   Dashboard:    http://localhost:{args.port}/v13_dashboard.html")
    print("=" * 80 + "\n")

    t_start = time.time()
    loaded_shards = 0
    last_pi_loss = 0.0
    last_v_loss = 0.0
    last_entropy = 0.69

    while True:
        if stop_file.exists() or (ckpt_dir / "stop").exists():
            print("\n[Stop] 'stop' file detected! Gracefully saving checkpoint and exiting...", flush=True)
            if stop_file.exists(): stop_file.unlink(missing_ok=True)
            if (ckpt_dir / "stop").exists(): (ckpt_dir / "stop").unlink(missing_ok=True)
            break

        # Check for new shards
        shard_path = Path(args.shard_dir)
        current_shards = sorted(list(shard_path.glob("shard_*.bin")))

        if not current_shards:
            print(f"[Waiting] Waiting for first teacher shard in {args.shard_dir}...", flush=True)
            time.sleep(5)
            continue

        dataset = V153TeacherDataset(args.shard_dir)
        loaded_shards = len(dataset.shard_files)
        total_records = len(dataset)

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True,
        )

        running_pi_loss = 0.0
        running_v_loss = 0.0
        running_entropy = 0.0
        correct_top1 = 0
        total_samples = 0
        t_batch_start = time.time()

        for batch in loader:
            if stop_file.exists() or (ckpt_dir / "stop").exists():
                break

            total_steps += 1
            x = batch["x"].to(device)
            target = batch["target"].to(device)
            legal_mask = batch["legal_mask"].to(device)
            target_val = batch["target_value"].to(device)

            optimizer.zero_grad(set_to_none=True)
            policy_logits, val_pred = model(x, legal_mask)

            # Masked Cross-Entropy Loss
            masked_logits = policy_logits.masked_fill(legal_mask == 0, -1e9)
            log_p = F.log_softmax(masked_logits, dim=-1)
            policy_loss = -log_p.gather(1, target.unsqueeze(1)).squeeze(1).mean()

            # Value MSE Loss
            value_loss = F.mse_loss(val_pred.squeeze(-1), target_val)

            # Entropy over valid legal actions
            with torch.no_grad():
                probs = F.softmax(masked_logits, dim=-1)
                log_probs_safe = torch.where(legal_mask > 0, log_p, torch.zeros_like(log_p))
                step_entropy = -(probs * log_probs_safe).sum(dim=-1).mean().item()

            total_loss = policy_loss + 0.5 * value_loss
            total_loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Metrics
            pred_cell = policy_logits.argmax(dim=-1)
            correct_top1 += (pred_cell == target).sum().item()
            total_samples += target.size(0)
            total_states_trained += target.size(0)

            running_pi_loss += policy_loss.item()
            running_v_loss += value_loss.item()
            running_entropy += step_entropy

            if total_steps % 100 == 0:
                elapsed = time.time() - t_batch_start
                sps = total_samples / max(0.1, elapsed)
                top1_acc = correct_top1 / max(1, total_samples) * 100
                last_pi_loss = running_pi_loss / 100
                last_v_loss = running_v_loss / 100
                last_entropy = running_entropy / 100

                print(
                    f"Step {total_steps:5d} | {loaded_shards} Shards ({total_records:,} st) | "
                    f"Top-1 Acc: {top1_acc:5.1f}% | Pi Loss: {last_pi_loss:.4f} | V Loss: {last_v_loss:.4f} | "
                    f"Entropy: {last_entropy:.3f} | Speed: {sps:.0f} SPS",
                    flush=True
                )
                running_pi_loss = 0.0
                running_v_loss = 0.0
                running_entropy = 0.0
                correct_top1 = 0
                total_samples = 0
                t_batch_start = time.time()

                # Frequent dashboard stats update
                stats_payload.update({
                    "timestamp": time.time(),
                    "total_games": total_states_trained // 50,
                    "total_states": total_states_trained,
                    "games_per_minute": (total_states_trained // 50) / max(0.1, (time.time() - t_start) / 60.0),
                    "states_per_minute": total_states_trained / max(0.1, (time.time() - t_start) / 60.0),
                    "policy_loss": last_pi_loss,
                    "value_loss": last_v_loss,
                    "total_loss": last_pi_loss + last_v_loss,
                    "policy_entropy": last_entropy,
                })
                with open(stats_path, "w") as f:
                    json.dump(stats_payload, f, indent=2)

            # Periodic Evaluation
            if total_steps % args.eval_every_steps == 0:
                print(f"\n[Eval @ Step {total_steps}] Evaluating zero-search policy against bot suite...", flush=True)
                eval_wr, opp_breakdown = evaluate_model(model, device, games_per_bot=args.eval_games // 4)
                print(
                    f"[Eval #{total_steps}] Overall WR: {eval_wr:.1%} | "
                    f"Heuristic: {opp_breakdown.get('Heuristic', 0):.1%}, "
                    f"Aggressive: {opp_breakdown.get('Aggressive', 0):.1%}, "
                    f"Expert: {opp_breakdown.get('Expert', 0):.1%}, "
                    f"MCTS: {opp_breakdown.get('MCTS', 0):.1%}\n",
                    flush=True
                )

                if eval_wr > best_eval_wr:
                    best_eval_wr = eval_wr
                    torch.save(
                        {
                            "step": total_steps,
                            "states": total_states_trained,
                            "best_eval_wr": best_eval_wr,
                            "model_state_dict": model.state_dict(),
                        },
                        ckpt_dir / "model_best.pt",
                    )
                    print(f"🌟 New best model saved to {ckpt_dir / 'model_best.pt'} (WR: {best_eval_wr:.1%})!")

                # Update stats & metrics
                model_elo = 1200 + int(eval_wr * 600)
                stats_payload.update({
                    "timestamp": time.time(),
                    "total_games": total_states_trained // 50,
                    "total_states": total_states_trained,
                    "games_per_minute": (total_states_trained // 50) / max(0.1, (time.time() - t_start) / 60.0),
                    "states_per_minute": total_states_trained / max(0.1, (time.time() - t_start) / 60.0),
                    "policy_loss": last_pi_loss,
                    "value_loss": last_v_loss,
                    "total_loss": last_pi_loss + last_v_loss,
                    "policy_entropy": last_entropy,
                    "main_elo": model_elo,
                    "eval_win_rate": float(eval_wr),
                    "opp_breakdown": opp_breakdown,
                    "recent_opponent_stats": {k: {"win_rate": v * 100, "wins": int(v * (args.eval_games // 4)), "games": args.eval_games // 4} for k, v in opp_breakdown.items()},
                    "elo_rankings": [
                        {"name": "MCTSBot", "elo": 1600},
                        {"name": "ExpertBot", "elo": 1500},
                        {"name": "HeuristicBot", "elo": 1400},
                        {"name": "Model", "elo": model_elo},
                        {"name": "AggressiveBot", "elo": 1350},
                    ],
                })
                with open(stats_path, "w") as f:
                    json.dump(stats_payload, f, indent=2)

                metrics_history.append({
                    "step": total_steps,
                    "games": total_states_trained // 50,
                    "states": total_states_trained,
                    "eval_win_rate": float(eval_wr),
                    "opponents": opp_breakdown,
                })
                with open(metrics_path, "w") as f:
                    json.dump(metrics_history, f, indent=2)

        # Save checkpoint after processing shards
        torch.save(
            {
                "step": total_steps,
                "states": total_states_trained,
                "best_eval_wr": best_eval_wr,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            ckpt_dir / "model_latest.pt",
        )


if __name__ == "__main__":
    main()
