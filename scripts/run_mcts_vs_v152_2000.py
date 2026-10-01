#!/usr/bin/env python3
"""
2,000-Game Showdown: Expecti-MCTS Search (N=300) vs V15.2 GraphTransformer.
Executes 2,000 parallel games across 8 worker processes with 50/50 seat balance:
  - 1,000 games: MCTS (N=300) as Player 0, V15.2 as Player 2
  - 1,000 games: V15.2 as Player 0, MCTS (N=300) as Player 2
Reports Wilson 95% confidence intervals, seat balance, and game dynamics.
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
TD_LUDO_DIR = ROOT_DIR / "td_ludo"
PYTHON_DIR = ROOT_DIR / "python"
V15_ROOT = ROOT_DIR / "td_ludo_v15"
for p in (str(TD_LUDO_DIR), str(PYTHON_DIR), str(V15_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import alphaludo_rs
import td_ludo_cpp as cpp
import td_ludo_v15_cpp as v15_cpp
from td_ludo_v15.models.v15 import V15GraphTransformer
from td_ludo_v15.game.cells import NUM_BOARD_CELLS, cell_to_index, position_to_cell_in_pov
from td_ludo_v15.game.encoder import encode_frame

_BASE_POS = v15_cpp.BASE_POS

_WORKER_V152 = None
_NUM_SIMS = 300


def worker_init(num_sims):
    global _WORKER_V152, _NUM_SIMS
    _NUM_SIMS = num_sims
    torch.set_num_threads(1)

    v152_ck = torch.load(ROOT_DIR / "td_ludo/play/model_weights/v15_2/model_best.pt", map_location="cpu", weights_only=False)
    m152 = V15GraphTransformer(d_model=128, n_heads=4, n_layers=4, ffn_dim=256, history_len=1)
    m152.load_state_dict(v152_ck["model_state_dict"])
    m152.eval()
    for p in m152.parameters():
        p.requires_grad = False
    _WORKER_V152 = m152


def play_one_game(task_tuple):
    seed, mcts_seat = task_tuple
    random.seed(seed)
    np.random.seed(seed)
    state = cpp.create_initial_state_2p()
    moves = 0
    max_moves = 350

    while not state.is_terminal and moves < max_moves:
        cp = int(state.current_player)
        d = random.randint(1, 6)
        state.current_dice_roll = d
        legal = cpp.get_legal_moves(state)
        if not legal:
            state.current_player = 2 if cp == 0 else 0
            state.current_dice_roll = 0
            continue

        if cp == mcts_seat:
            # Native Rust Expecti-MCTS
            if len(legal) == 1:
                act = legal[0]
            else:
                pos = np.ascontiguousarray(state.player_positions, dtype=np.int8)
                act = alphaludo_rs.select_mcts_move_2p(pos, int(cp), int(d), _NUM_SIMS)
                if act not in legal:
                    act = legal[0]
        else:
            # V15.2 GraphTransformer
            if len(legal) == 1:
                act = legal[0]
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
                    xt = torch.from_numpy(v15_x).unsqueeze(0)
                    mt = torch.from_numpy(v15_legal).unsqueeze(0)
                    pol, _ = _WORKER_V152(xt, mt)
                    chosen_idx = int(pol.argmax(dim=-1).item())
                chosen_cell = divmod(chosen_idx, 15)
                act = legal[0]
                for t, c in legal_cells:
                    if c == chosen_cell:
                        act = t
                        break

        state = cpp.apply_move(state, int(act))
        moves += 1

    winner = cpp.get_winner(state) if state.is_terminal else -1
    mcts_won = (winner == mcts_seat)
    v152_won = (winner != -1 and winner != mcts_seat)
    return {
        "mcts_seat": mcts_seat,
        "winner": winner,
        "mcts_won": mcts_won,
        "v152_won": v152_won,
        "moves": moves,
    }


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95):
    if total == 0:
        return 0.0, 0.0, 0.0
    z = 1.95996  # 95% confidence
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    return p, lower, upper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-games", type=int, default=2000, help="Total games to play (50/50 seat rotation)")
    parser.add_argument("--sims", type=int, default=300, help="MCTS simulations per decision")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker processes")
    args = parser.parse_args()

    n_per_orientation = args.total_games // 2
    total_games = n_per_orientation * 2

    print("=" * 80)
    print(f"🥊 LARGE-SCALE HEAD-TO-HEAD: Native Expecti-MCTS (N={args.sims}) vs V15.2 (GraphTransformer)")
    print(f"   Target Games: {total_games:,} ({n_per_orientation:,} as P0 / {n_per_orientation:,} as P2)")
    print(f"   Worker Processes: {args.workers}")
    print("=" * 80)

    tasks = []
    # Orientation 1: MCTS as P0, V15.2 as P2
    for i in range(n_per_orientation):
        tasks.append((200000 + i, 0))
    # Orientation 2: V15.2 as P0, MCTS as P2
    for i in range(n_per_orientation):
        tasks.append((600000 + i, 2))

    t0 = time.time()
    ctx = mp.get_context("spawn")
    results = []

    print(f"\n[Tournament] Starting {total_games:,} games across {args.workers} workers...")
    with ctx.Pool(args.workers, initializer=worker_init, initargs=(args.sims,)) as pool:
        for idx, res in enumerate(pool.imap_unordered(play_one_game, tasks, chunksize=25), 1):
            results.append(res)
            if idx % 200 == 0 or idx == total_games:
                elapsed = time.time() - t0
                gps = idx / max(0.1, elapsed)
                mcts_wins = sum(1 for r in results if r["mcts_won"])
                v152_wins = sum(1 for r in results if r["v152_won"])
                print(
                    f"  Progress: {idx:4d}/{total_games} games ({idx/total_games*100:5.1f}%) | "
                    f"MCTS(N={args.sims}): {mcts_wins:4d} ({mcts_wins/idx*100:4.1f}%) vs "
                    f"V15.2: {v152_wins:4d} ({v152_wins/idx*100:4.1f}%) | "
                    f"Speed: {gps:.1f} games/s | "
                    f"Elapsed: {elapsed:.1f}s",
                    flush=True
                )

    dt = time.time() - t0
    print("\n" + "=" * 80)
    print("📊 TOURNAMENT COMPLETE — FINAL STATISTICAL REPORT")
    print("=" * 80)

    mcts_wins = sum(1 for r in results if r["mcts_won"])
    v152_wins = sum(1 for r in results if r["v152_won"])
    draws = sum(1 for r in results if not r["mcts_won"] and not r["v152_won"])

    # Seat breakdown
    mcts_p0_games = [r for r in results if r["mcts_seat"] == 0]
    mcts_p2_games = [r for r in results if r["mcts_seat"] == 2]

    mcts_p0_wins = sum(1 for r in mcts_p0_games if r["mcts_won"])
    mcts_p2_wins = sum(1 for r in mcts_p2_games if r["mcts_won"])

    v152_p0_wins = sum(1 for r in mcts_p2_games if r["v152_won"])
    v152_p2_wins = sum(1 for r in mcts_p0_games if r["v152_won"])

    p0_total_wins = sum(1 for r in results if r["winner"] == 0)
    p2_total_wins = sum(1 for r in results if r["winner"] == 2)

    avg_moves = np.mean([r["moves"] for r in results])

    p_mcts, low_mcts, high_mcts = wilson_score_interval(mcts_wins, total_games)
    p_v152, low_v152, high_v152 = wilson_score_interval(v152_wins, total_games)

    print(f"Total Games Played:     {total_games:,}")
    print(f"Total Time:             {dt:.2f} seconds ({total_games / dt:.1f} games/sec)")
    print(f"Average Game Length:    {avg_moves:.1f} moves")
    print(f"Draws / Truncations:    {draws}\n")

    print(f"Overall Record:")
    print(f"  • MCTS (N={args.sims}):         {mcts_wins:4d} wins ({p_mcts*100:5.2f}%)  [95% CI: {low_mcts*100:.2f}% – {high_mcts*100:.2f}%]")
    print(f"  • V15.2 GraphTransformer:{v152_wins:4d} wins ({p_v152*100:5.2f}%)  [95% CI: {low_v152*100:.2f}% – {high_v152*100:.2f}%]\n")

    print(f"Seat Breakdown (Testing Seat Bias):")
    print(f"  • Total P0 Wins:        {p0_total_wins} ({p0_total_wins/total_games*100:.1f}%)")
    print(f"  • Total P2 Wins:        {p2_total_wins} ({p2_total_wins/total_games*100:.1f}%)")
    print(f"  • MCTS as P0 (First):   {mcts_p0_wins:4d} / {len(mcts_p0_games)} ({mcts_p0_wins/len(mcts_p0_games)*100:.1f}%)")
    print(f"  • MCTS as P2 (Second):  {mcts_p2_wins:4d} / {len(mcts_p2_games)} ({mcts_p2_wins/len(mcts_p2_games)*100:.1f}%)")
    print(f"  • V15.2 as P0 (First):  {v152_p0_wins:4d} / {len(mcts_p2_games)} ({v152_p0_wins/len(mcts_p2_games)*100:.1f}%)")
    print(f"  • V15.2 as P2 (Second): {v152_p2_wins:4d} / {len(mcts_p0_games)} ({v152_p2_wins/len(mcts_p0_games)*100:.1f}%)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
