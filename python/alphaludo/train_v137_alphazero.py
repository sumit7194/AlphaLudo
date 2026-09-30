"""AlphaLudo V13.7 — Tabula Rasa AlphaZero 2-Player Self-Play Trainer.

Features:
  - 100% Pure AlphaZero tabula rasa: initialized from random weights.
  - Deep Expecti-MCTS exploration at N = 3,000 simulations per move in native multi-threaded Rust.
  - Zero disk space generation: states generated directly into RAM replay buffer.
  - Pure terminal rewards: z in {+1.0, -1.0}.
  - Minimal Dual-Head ResNet (V13.7): 17-channel input, 6 ResBlocks x 128ch.
  - Serves rich interactive V13 dashboard on port 8790 with live training & eval metrics.
  - Clean interruptability via 'stop' file and seamless resumption from checkpoints.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import random
import sys
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
TD_LUDO_DIR = ROOT_DIR / "td_ludo"
PYTHON_DIR = ROOT_DIR / "python"
sys.path.insert(0, str(TD_LUDO_DIR))
sys.path.insert(0, str(PYTHON_DIR))

import alphaludo_rs
import td_ludo_cpp as ludo_cpp
from td_ludo.game.encoder_v17 import encode_state_v17
from td_ludo.game.heuristic_bot import get_bot
from alphaludo.model_v137 import AlphaLudoV137


# ─────────────────────────────────────────────────────────────────────────────
# 1. Circular Replay Buffer in RAM
# ─────────────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.size = 0
        self.ptr = 0

        # Pre-allocate pinned / CPU tensors
        self.frames = torch.zeros((capacity, 17, 15, 15), dtype=torch.float32)
        self.masks = torch.zeros((capacity, 4), dtype=torch.float32)
        self.pis = torch.zeros((capacity, 4), dtype=torch.float32)
        self.values = torch.zeros((capacity,), dtype=torch.float32)

    def push(
        self,
        frames: np.ndarray,
        masks: np.ndarray,
        pis: np.ndarray,
        values: np.ndarray,
    ):
        n = len(frames)
        f_tensor = torch.from_numpy(frames)
        m_tensor = torch.from_numpy(masks)
        p_tensor = torch.from_numpy(pis)
        v_tensor = torch.from_numpy(values)

        if n >= self.capacity:
            # Take newest capacity items
            f_tensor = f_tensor[-self.capacity:]
            m_tensor = m_tensor[-self.capacity:]
            p_tensor = p_tensor[-self.capacity:]
            v_tensor = v_tensor[-self.capacity:]
            n = self.capacity
            self.ptr = 0
            self.size = self.capacity
            self.frames[:] = f_tensor
            self.masks[:] = m_tensor
            self.pis[:] = p_tensor
            self.values[:] = v_tensor
            return

        end = self.ptr + n
        if end <= self.capacity:
            self.frames[self.ptr:end] = f_tensor
            self.masks[self.ptr:end] = m_tensor
            self.pis[self.ptr:end] = p_tensor
            self.values[self.ptr:end] = v_tensor
        else:
            first_part = self.capacity - self.ptr
            second_part = n - first_part
            self.frames[self.ptr:self.capacity] = f_tensor[:first_part]
            self.masks[self.ptr:self.capacity] = m_tensor[:first_part]
            self.pis[self.ptr:self.capacity] = p_tensor[:first_part]
            self.values[self.ptr:self.capacity] = v_tensor[:first_part]

            self.frames[:second_part] = f_tensor[first_part:]
            self.masks[:second_part] = m_tensor[first_part:]
            self.pis[:second_part] = p_tensor[first_part:]
            self.values[:second_part] = v_tensor[first_part:]

        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int):
        idx = torch.randint(0, self.size, (batch_size,))
        return (
            self.frames[idx].to(self.device),
            self.masks[idx].to(self.device),
            self.pis[idx].to(self.device),
            self.values[idx].to(self.device),
        )



# ─────────────────────────────────────────────────────────────────────────────
# 2. Rich Dashboard Server
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
# 3. Model Benchmark Evaluation
# ─────────────────────────────────────────────────────────────────────────────
class RustMCTSBot:
    """Native Rust 2-Player Zero-Sum Expecti-MCTS bot for evaluation."""
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


def evaluate_model(
    model: nn.Module,
    device: torch.device,
    games_per_opponent: int = 20,
) -> tuple[float, dict[str, float]]:
    """Evaluates greedy model policy against bot suite with cyclic seat rotation."""
    model.eval()
    bot_names = ["Heuristic", "Aggressive", "Expert", "MCTS"]
    results = {}
    total_wins = 0
    total_games = 0

    for bot_name in bot_names:
        bot_wins = 0
        for g_idx in range(games_per_opponent):
            # Alternate seats: even index model is P0, odd index model is P2
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
                        enc = encode_state_v17(state)
                        mask = np.zeros(4, dtype=np.float32)
                        for m in legal:
                            mask[m] = 1.0

                        with torch.no_grad():
                            x_t = torch.from_numpy(enc).unsqueeze(0).to(device)
                            m_t = torch.from_numpy(mask).unsqueeze(0).to(device)
                            logits, _ = model(x_t, m_t)
                            action = int(logits.argmax(dim=1).item())
                        if action not in legal:
                            action = random.choice(legal)
                else:
                    action = bot.select_move(state, list(legal))

                state = ludo_cpp.apply_move(state, int(action))
                move_count += 1

            if state.is_terminal and ludo_cpp.get_winner(state) == model_seat:
                bot_wins += 1

        wr = bot_wins / max(1, games_per_opponent)
        results[bot_name] = wr
        total_wins += bot_wins
        total_games += games_per_opponent

    overall_wr = total_wins / max(1, total_games)
    model.train()
    return overall_wr, results


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main AlphaZero Training Loop
# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    # Setup device
    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[Device] Using {device}")

    # Output paths
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    stats_path = str(ckpt_dir / "stats.json")
    metrics_path = str(ckpt_dir / "metrics.json")
    stop_file = ROOT_DIR / "stop"

    # Start dashboard server
    start_dashboard(
        port=args.port,
        stats_path=stats_path,
        metrics_path=metrics_path,
        dashboard_dir=str(TD_LUDO_DIR),
    )

    # Initialize model
    model = AlphaLudoV137(
        in_channels=17,
        num_res_blocks=args.num_res_blocks,
        num_channels=args.num_channels,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[Model] Initialized V13.7 ResNet ({args.num_res_blocks} blocks x {args.num_channels} ch): {param_count:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Resume if checkpoint exists
    start_iter = 0
    total_states_seen = 0
    total_games_seen = 0
    best_eval_wr = 0.0
    metrics_history = []

    last_eval_wr = best_eval_wr
    last_opp_breakdown = {}

    latest_ckpt = ckpt_dir / "model_latest.pt"
    if latest_ckpt.exists():
        print(f"[Resume] Loading existing checkpoint from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_iter = ckpt.get("iteration", 0)
        total_states_seen = ckpt.get("total_states", 0)
        total_games_seen = ckpt.get("total_games", 0)
        best_eval_wr = ckpt.get("best_eval_wr", 0.0)
        last_eval_wr = best_eval_wr
        if (ckpt_dir / "metrics.json").exists():
            try:
                with open(ckpt_dir / "metrics.json") as f:
                    metrics_history = json.load(f)
                    if metrics_history:
                        last_eval_wr = metrics_history[-1].get("eval_win_rate", best_eval_wr)
                        last_opp_breakdown = metrics_history[-1].get("opponents", {})
            except Exception:
                pass
        print(f"[Resume] Resumed at iteration {start_iter}, states={total_states_seen:,}, best WR={best_eval_wr:.1%}")

    initial_recent_opp = {
        name: {
            "win_rate": float(wr * 100.0),
            "wins": int(round(wr * (args.eval_games // max(1, len(last_opp_breakdown))))),
            "games": int(args.eval_games // max(1, len(last_opp_breakdown))),
        }
        for name, wr in last_opp_breakdown.items()
    }

    # Write initial stats payload so dashboard connects immediately
    stats_payload = {
        "run_info": {
            "model_name": "AlphaLudo V13.7 (AlphaZero 2-Player)",
            "badge": f"Tabula Rasa MCTS (N={args.mcts_sims})",
            "arch_summary": f"V13.7 Minimal ResNet ({args.num_res_blocks}x{args.num_channels}) · 1.8M params · Terminal Only",
            "description": f"Pure AlphaZero self-play on 2-Player Ludo (P0 vs P2). N={args.mcts_sims} simulations, zero-sum terminal rewards z in {{-1, +1}}.",
            "target_wr": 0.85,
            "target_label": "Mastery Target (85% WR)",
            "baseline_wr": 0.50,
            "baseline_label": "Even Baseline (50% WR)",
            "eval_games": args.eval_games,
        },
        "timestamp": time.time(),
        "total_games": total_games_seen,
        "total_states": total_states_seen,
        "games_per_minute": 0.0,
        "states_per_minute": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "total_loss": 0.0,
        "policy_entropy": 0.70,
        "temperature": 1.0,
        "main_elo": 1200 + int(last_eval_wr * 600),
        "eval_win_rate": float(last_eval_wr),
        "opp_breakdown": last_opp_breakdown,
        "recent_opponent_stats": initial_recent_opp,
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

    # CPU shadow model for fast single-step evaluation without GPU stream syncs
    cpu_model = AlphaLudoV137(
        in_channels=17,
        num_res_blocks=args.num_res_blocks,
        num_channels=args.num_channels,
    ).to("cpu")

    # Replay Buffer
    buffer = ReplayBuffer(capacity=args.replay_capacity, device=device)

    start_time = time.time()
    last_log_time = time.time()

    print("\n" + "=" * 80, flush=True)
    print(f"🚀 ALPHAZERO V13.7 SELF-PLAY RUN LAUNCHED", flush=True)
    print(f"   Mode: 100% Tabula Rasa (Random Weights Init)", flush=True)
    print(f"   Search: Expecti-MCTS N={args.mcts_sims} simulations/move", flush=True)
    print(f"   Rewards: Terminal Only z in {{-1.0, +1.0}}", flush=True)
    print(f"   RAM Buffer: {args.replay_capacity:,} states", flush=True)
    print(f"   Dashboard: http://localhost:{args.port}/v13_dashboard.html", flush=True)
    print("=" * 80 + "\n", flush=True)

    iteration = start_iter
    while True:
        # Check for clean interrupt
        if stop_file.exists() or (ckpt_dir / "stop").exists():
            print("\n[Stop] 'stop' file detected! Gracefully saving checkpoint and exiting...", flush=True)
            if stop_file.exists():
                stop_file.unlink(missing_ok=True)
            if (ckpt_dir / "stop").exists():
                (ckpt_dir / "stop").unlink(missing_ok=True)
            break

        iter_start = time.time()
        iteration += 1

        # ── 1. Self-Play Phase (Native Multi-Threaded Rust MCTS) ─────────
        gen_t0 = time.time()
        frames, masks, pis, values, actions = alphaludo_rs.generate_alphazero_2p_batch(
            min_states=args.states_per_iter,
            num_simulations=args.mcts_sims,
            temperature_cutoff=args.temp_cutoff,
            use_heuristic_prior=False, # Tabula rasa: uniform + Dirichlet
        )
        gen_dt = time.time() - gen_t0

        n_new = len(frames)
        n_games = max(1, n_new // 50)
        total_states_seen += n_new
        total_games_seen += n_games

        buffer.push(frames, masks, pis, values)

        # ── 2. Training Optimization Phase ─────────────────────────────
        model.train()
        train_t0 = time.time()
        running_pi_loss = torch.zeros((), device=device)
        running_v_loss = torch.zeros((), device=device)
        running_ent = torch.zeros((), device=device)
        num_steps = args.train_steps_per_iter

        for _ in range(num_steps):
            b_frames, b_masks, b_pis, b_values = buffer.sample(args.batch_size)

            optimizer.zero_grad(set_to_none=True)
            logits, value = model(b_frames, b_masks)

            # Policy Loss: Cross Entropy with MCTS visit distribution
            log_probs = F.log_softmax(logits, dim=-1)
            clean_log_probs = torch.where(b_pis > 0, log_probs, torch.zeros_like(log_probs))
            policy_loss = -(b_pis * clean_log_probs).sum(dim=-1).mean()

            # Value Loss: MSE with terminal payoff z
            value_loss = F.mse_loss(value, b_values)

            # Policy Entropy over legal actions
            probs = F.softmax(logits, dim=-1)
            clean_log_p = torch.where(probs > 0, torch.log(probs.clamp_min(1e-8)), torch.zeros_like(probs))
            entropy = -(probs * clean_log_p).sum(dim=-1).mean()

            total_loss = policy_loss + value_loss
            total_loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_pi_loss += policy_loss.detach()
            running_v_loss += value_loss.detach()
            running_ent += entropy.detach()

        train_dt = time.time() - train_t0
        avg_pi_loss = (running_pi_loss / num_steps).item()
        avg_v_loss = (running_v_loss / num_steps).item()
        avg_entropy = (running_ent / num_steps).item()
        total_loss = avg_pi_loss + avg_v_loss

        # ── 3. Periodic Evaluation Phase (Fast CPU Evaluation) ─────────
        eval_wr = None
        opp_breakdown = {}
        if iteration % args.eval_every == 0 or iteration == 1:
            eval_t0 = time.time()
            # Copy latest weights to CPU model for zero-overhead inference
            cpu_model.load_state_dict(model.state_dict())
            eval_wr, opp_breakdown = evaluate_model(
                model=cpu_model,
                device=torch.device("cpu"),
                games_per_opponent=args.eval_games // 4,
            )
            eval_dt = time.time() - eval_t0
            last_eval_wr = eval_wr
            last_opp_breakdown = opp_breakdown

            print(
                f"[Eval #{iteration}] WR: {eval_wr:.1%} | "
                f"Heuristic: {opp_breakdown.get('Heuristic', 0):.1%}, "
                f"Aggressive: {opp_breakdown.get('Aggressive', 0):.1%}, "
                f"Expert: {opp_breakdown.get('Expert', 0):.1%}, "
                f"MCTS: {opp_breakdown.get('MCTS', 0):.1%} ({eval_dt:.1f}s)",
                flush=True
            )


            # Update metrics history
            metric_entry = {
                "step": iteration,
                "games": total_games_seen,
                "states": total_states_seen,
                "eval_win_rate": float(eval_wr),
                "policy_loss": float(avg_pi_loss),
                "value_loss": float(avg_v_loss),
                "entropy": float(avg_entropy),
                "opponents": opp_breakdown,
            }
            metrics_history.append(metric_entry)
            with open(metrics_path, "w") as f:
                json.dump(metrics_history, f, indent=2)

            # Checkpoint best
            if eval_wr > best_eval_wr:
                best_eval_wr = eval_wr
                torch.save(
                    {
                        "iteration": iteration,
                        "total_states": total_states_seen,
                        "total_games": total_games_seen,
                        "best_eval_wr": best_eval_wr,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    ckpt_dir / "model_best.pt",
                )
                print(f"⭐ New Best Model Saved: {best_eval_wr:.1%} WR", flush=True)

        # Save Latest Checkpoint
        torch.save(
            {
                "iteration": iteration,
                "total_states": total_states_seen,
                "total_games": total_games_seen,
                "best_eval_wr": best_eval_wr,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            latest_ckpt,
        )

        # ── 4. Update Live Stats for Dashboard ─────────────────────────
        elapsed = time.time() - start_time
        gpm = (total_games_seen / max(1.0, elapsed)) * 60.0
        spm = (total_states_seen / max(1.0, elapsed)) * 60.0

        elo_rankings = [
            {"name": "MCTSBot", "elo": 1600},
            {"name": "ExpertBot", "elo": 1500},
            {"name": "HeuristicBot", "elo": 1400},
            {"name": "AggressiveBot", "elo": 1350},
            {"name": "Model", "elo": 1200 + int(last_eval_wr * 600)},
        ]
        elo_rankings.sort(key=lambda x: x["elo"], reverse=True)

        recent_opp_stats = {
            name: {
                "win_rate": float(wr * 100.0),
                "wins": int(round(wr * (args.eval_games // max(1, len(last_opp_breakdown))))),
                "games": int(args.eval_games // max(1, len(last_opp_breakdown))),
            }
            for name, wr in last_opp_breakdown.items()
        }

        stats_payload = {
            "run_info": {
                "model_name": "AlphaLudo V13.7 (AlphaZero 2-Player)",
                "badge": f"Tabula Rasa MCTS (N={args.mcts_sims})",
                "arch_summary": f"V13.7 Minimal ResNet ({args.num_res_blocks}x{args.num_channels}) · 1.8M params · Terminal Only",
                "description": f"Pure AlphaZero self-play on 2-Player Ludo (P0 vs P2). N={args.mcts_sims} simulations, zero-sum terminal rewards z in {{-1, +1}}.",
                "target_wr": 0.85,
                "target_label": "Mastery Target (85% WR)",
                "baseline_wr": 0.50,
                "baseline_label": "Even Baseline (50% WR)",
                "eval_games": args.eval_games,
            },
            "timestamp": time.time(),
            "total_games": total_games_seen,
            "total_states": total_states_seen,
            "games_per_minute": float(gpm),
            "states_per_minute": float(spm),
            "policy_loss": float(avg_pi_loss),
            "value_loss": float(avg_v_loss),
            "total_loss": float(total_loss),
            "policy_entropy": float(avg_entropy),
            "temperature": 1.0 if iteration < 20 else 0.0,
            "main_elo": 1200 + int(last_eval_wr * 600),
            "eval_win_rate": float(last_eval_wr),
            "opp_breakdown": last_opp_breakdown,
            "recent_opponent_stats": recent_opp_stats,
            "elo_rankings": elo_rankings,
        }
        with open(stats_path, "w") as f:
            json.dump(stats_payload, f, indent=2)


        # ── 5. Memory Management & Logging ────────────────────────────
        if device.type == "mps":
            torch.mps.empty_cache()

        iter_dt = time.time() - iter_start
        print(
            f"Iter {iteration:4d} | +{n_new:4d} st ({gen_dt:.2f}s) | "
            f"Loss: {total_loss:.4f} (Pi: {avg_pi_loss:.4f}, V: {avg_v_loss:.4f}) | "
            f"Ent: {avg_entropy:.3f} | Total: {total_states_seen:,} st | {iter_dt:.2f}s",
            flush=True
        )



def parse_args():
    parser = argparse.ArgumentParser(description="AlphaLudo V13.7 Tabula Rasa AlphaZero Trainer")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=60000)
    parser.add_argument("--mcts-sims", type=int, default=3000)
    parser.add_argument("--states-per-iter", type=int, default=2000)
    parser.add_argument("--train-steps-per-iter", type=int, default=250)
    parser.add_argument("--temp-cutoff", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-res-blocks", type=int, default=6)
    parser.add_argument("--num-channels", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-games", type=int, default=80)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/v13_7")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--device", type=str, default="mps")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
