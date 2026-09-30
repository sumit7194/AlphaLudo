"""1000-Game Head-to-Head Tournament Benchmark.

Evaluates:
  1. V15_4PW (Winner-distilled with MCTS & Depth2, 9.14M states)
  2. V15_4P_RL (Yesterday's 105k RL model)
  3. ExpertBot (Rule-based tactical bot)
  4. HeuristicBot (Standard baseline heuristic bot)

Uses strict seat rotation across 1,000 games (250 games per seat per bot)
to completely eliminate seat/turn order bias.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

# Add search paths
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for p in (str(_REPO_ROOT / "td_ludo_v15"), str(_REPO_ROOT / "td_ludo"), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import td_ludo_cpp as cpp
from td_ludo.game.heuristic_bot import ExpertBot, HeuristicLudoBot
from td_ludo_v15.game.encoder_4p import encode_frame_4p
from td_ludo_v15.game.cells import (
    NUM_BOARD_CELLS,
    cell_to_index,
    index_to_cell,
    position_to_cell_in_pov,
)
from python.alphaludo.model import V15_4PW_GraphTransformer


BOT_NAMES = [
    "V15_4PW (Winner-Distilled)",
    "V15_4P_RL (105k RL Best)",
    "ExpertBot",
    "HeuristicBot",
]


def select_nn_move(model, state, cp: int, legal, device: torch.device) -> int:
    """Selects move for neural network using policy head."""
    frame = encode_frame_4p(state, pov_player=cp)
    x_t = torch.from_numpy(frame).unsqueeze(0).to(device)

    legal_mask = np.zeros(NUM_BOARD_CELLS, dtype=np.float32)
    for t in legal:
        pos = int(state.player_positions[cp][t])
        c = position_to_cell_in_pov(-1 if pos == -1 else pos, cp, cp)
        legal_mask[cell_to_index(*c)] = 1.0

    mask_t = torch.from_numpy(legal_mask).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x_t, legal_mask=mask_t)
        policy = out[0]

    best_cell_idx = int(policy[0].argmax().item())
    chosen_cell = index_to_cell(best_cell_idx)

    for t in legal:
        pos = int(state.player_positions[cp][t])
        c = position_to_cell_in_pov(-1 if pos == -1 else pos, cp, cp)
        if c == chosen_cell:
            return t

    return legal[0]


def run_tournament(n_games: int = 1000):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print("=" * 65)
    print("🏆 AlphaLudo 4-Player 1,000-Game Tournament")
    print(f"💻 Device: {device} | Games: {n_games}")
    print(f"🤖 Competitors:")
    for i, name in enumerate(BOT_NAMES):
        print(f"   [{i}] {name}")
    print("⚖️ Fair Random Baseline: 25.0% Win Rate (250 wins)")
    print("=" * 65)

    # 1. Load Models
    pw_path = Path("checkpoints/v15_4pw/model_latest.pt")
    rl_path = Path("checkpoints/v15_4p_rl/model_best.pt")

    print(f"📥 Loading V15_4PW from {pw_path}...")
    m_pw = V15_4PW_GraphTransformer().to(device).eval()
    m_pw.load_state_dict(torch.load(pw_path, map_location=device, weights_only=True))

    print(f"📥 Loading V15_4P_RL from {rl_path}...")
    m_rl = V15_4PW_GraphTransformer().to(device).eval()
    m_rl.load_state_dict(
        torch.load(rl_path, map_location=device, weights_only=True), strict=False
    )

    # Statistics
    total_wins = [0] * 4
    seat_wins = [[0] * 4 for _ in range(4)]  # [bot_id][seat]
    seat_games = [[0] * 4 for _ in range(4)]
    game_lengths = []

    t_start = time.time()
    t_last_log = t_start

    print("\n⚔️ Starting 1,000-game match...\n")

    for g in range(n_games):
        # Strict cyclic seat rotation: Bot `i` plays in seat `(i + g) % 4`
        # Thus every 4 games, each bot plays 1 game in each seat.
        seat_to_bot = [
            (0 - g) % 4,
            (1 - g) % 4,
            (2 - g) % 4,
            (3 - g) % 4,
        ]

        for seat in range(4):
            bot_id = seat_to_bot[seat]
            seat_games[bot_id][seat] += 1

        state = cpp.create_initial_state()
        consec_sixes = [0, 0, 0, 0]
        mc = 0

        while not state.is_terminal and mc < 800:
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

            bot_id = seat_to_bot[cp]

            if len(legal) == 1:
                action = legal[0]
            elif bot_id == 0:  # V15_4PW
                action = select_nn_move(m_pw, state, cp, legal, device)
            elif bot_id == 1:  # V15_4P_RL
                action = select_nn_move(m_rl, state, cp, legal, device)
            elif bot_id == 2:  # ExpertBot
                action = ExpertBot(player_id=cp).select_move(state, list(legal))
            else:  # HeuristicBot
                action = HeuristicLudoBot(player_id=cp).select_move(state, list(legal))

            state = cpp.apply_move(state, int(action))
            mc += 1

        game_lengths.append(mc)

        if state.is_terminal:
            winner_seat = int(cpp.get_winner(state))
            winner_bot = seat_to_bot[winner_seat]
            total_wins[winner_bot] += 1
            seat_wins[winner_bot][winner_seat] += 1

        # Periodic telemetry
        completed = g + 1
        if completed % 100 == 0 or completed == n_games:
            dt = time.time() - t_start
            gpm = completed / dt * 60.0
            print(
                f"[{completed:4d}/{n_games}] Elapsed: {dt:.1f}s ({gpm:.1f} GPM) | "
                f"PW: {total_wins[0]} ({total_wins[0]/completed*100:4.1f}%) | "
                f"RL: {total_wins[1]} ({total_wins[1]/completed*100:4.1f}%) | "
                f"Exp: {total_wins[2]} ({total_wins[2]/completed*100:4.1f}%) | "
                f"Heur: {total_wins[3]} ({total_wins[3]/completed*100:4.1f}%)"
            )

    total_time = time.time() - t_start

    print("\n" + "=" * 70)
    print(f"📊 FINAL TOURNAMENT STANDINGS ({n_games} Games)")
    print("=" * 70)
    print(f"{'Bot Name':<30} | {'Wins':>6} | {'Win Rate':>8} | {'Seat 0':>6} | {'Seat 1':>6} | {'Seat 2':>6} | {'Seat 3':>6}")
    print("-" * 30 + "-+-" + "-" * 6 + "-+-" + "-" * 8 + "-+-" + "-" * 6 + "-+-" + "-" * 6 + "-+-" + "-" * 6 + "-+-" + "-" * 6)

    for i in range(4):
        wr = (total_wins[i] / n_games) * 100.0
        s0 = seat_wins[i][0]
        s1 = seat_wins[i][1]
        s2 = seat_wins[i][2]
        s3 = seat_wins[i][3]
        print(
            f"{BOT_NAMES[i]:<30} | {total_wins[i]:>6} | {wr:>7.1f}% | {s0:>6} | {s1:>6} | {s2:>6} | {s3:>6}"
        )

    print("=" * 70)
    print(f"⏱️ Total Tournament Time: {total_time:.2f}s ({n_games / total_time * 60.0:.1f} GPM)")
    print(f"🎲 Average Game Length:    {np.mean(game_lengths):.1f} moves")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=1000)
    args = parser.parse_args()
    run_tournament(n_games=args.games)
