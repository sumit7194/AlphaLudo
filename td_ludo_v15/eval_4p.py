"""Evaluation benchmark for 4-Player AlphaLudo models against bot pool.

Runs N games with seat balancing (model rotates equally through seats 0, 1, 2, 3).
Reports overall win rate, per-seat win rate, and game statistics.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

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
    NUM_BOARD_CELLS,
    cell_to_index,
    index_to_cell,
    position_to_cell_in_pov,
)
from td_ludo_v15.game.encoder_4p import encode_frame_4p
from td_ludo_v15.models.v15_4p import V15_4P_GraphTransformer

_BASE_POS = -1
_HOME_POS = 99


def create_bot(bot_name: str, player_id: int):
    if bot_name == "Expert":
        return ExpertBot(player_id=player_id)
    if bot_name == "Heuristic":
        return HeuristicLudoBot(player_id=player_id)
    if bot_name == "Expectimax":
        return ExpectimaxBot(player_id=player_id)
    if bot_name == "AggressiveExpectimax":
        return AggressiveExpectimaxBot(player_id=player_id)
    return ExpertBot(player_id=player_id)


def evaluate_4p(
    model_path: str,
    n_games: int = 100,
    device_str: str = "auto",
    bot_roster: Optional[List[str]] = None,
):
    if bot_roster is None:
        bot_roster = ["Expert", "Heuristic", "AggressiveExpectimax"]

    if device_str == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"🔎 Loading 4P Model from: {model_path}")
    model = V15_4P_GraphTransformer(d_model=128, n_heads=4, n_layers=4, ffn_dim=256)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"⚔️ Evaluating across {n_games} games on {device}")
    print(f"🤖 Opponent Pool: {', '.join(bot_roster)}")
    print(f"⚖️ Baseline Random Win Rate: 25.0%\n")

    seat_wins = [0, 0, 0, 0]
    seat_games = [0, 0, 0, 0]
    lengths = []
    t0 = time.time()

    with torch.no_grad():
        for g in range(n_games):
            student_seat = g % 4
            seat_games[student_seat] += 1

            bots = {}
            b_idx = 0
            for s in range(4):
                if s != student_seat:
                    bots[s] = create_bot(bot_roster[b_idx % len(bot_roster)], s)
                    b_idx += 1

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

                if cp == student_seat:
                    frame = encode_frame_4p(state, pov_player=cp)
                    x_t = torch.from_numpy(frame).unsqueeze(0).to(device)

                    legal_mask = np.zeros(NUM_BOARD_CELLS, dtype=np.float32)
                    for t in legal:
                        pos = int(state.player_positions[cp][t])
                        c = position_to_cell_in_pov(_BASE_POS if pos == _BASE_POS else pos, cp, cp)
                        legal_mask[cell_to_index(*c)] = 1.0
                    mask_t = torch.from_numpy(legal_mask).unsqueeze(0).to(device)

                    policy, _ = model(x_t, legal_mask=mask_t)
                    best_cell_idx = int(policy[0].argmax().item())
                    chosen_cell = index_to_cell(best_cell_idx)

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

            lengths.append(mc)
            if state.is_terminal and int(cpp.get_winner(state)) == student_seat:
                seat_wins[student_seat] += 1

            if (g + 1) % max(1, n_games // 5) == 0 or (g + 1) == n_games:
                total_w = sum(seat_wins)
                cur_wr = (total_w / (g + 1)) * 100.0
                print(f"[{g+1:4d}/{n_games}] Win Rate: {cur_wr:5.1f}% ({total_w}/{g+1})")

    elapsed = time.time() - t0
    total_wins = sum(seat_wins)
    overall_wr = (total_wins / n_games) * 100.0

    print("\n" + "=" * 55)
    print(f"📊 4-PLAYER EVALUATION RESULTS ({n_games} Games)")
    print("=" * 55)
    print(f"🏆 Overall Student Win Rate : {overall_wr:5.1f}% ({total_wins}/{n_games})")
    print(f"⏱️ Elapsed Time             : {elapsed:.1f}s ({n_games / elapsed * 60:.1f} GPM)")
    print(f"📏 Avg Game Length          : {np.mean(lengths):.1f} moves")
    print("-" * 55)
    print("Seat Breakdown:")
    for s in range(4):
        wr = (seat_wins[s] / max(1, seat_games[s])) * 100.0
        print(f"  Seat {s} (P{s}): {wr:5.1f}% ({seat_wins[s]}/{seat_games[s]})")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/v15_4p_sl/model_best.pt", help="Path to checkpoint")
    parser.add_argument("--games", type=int, default=100, help="Number of games to evaluate")
    parser.add_argument("--device", default="auto", help="Device (auto, mps, cpu)")
    args = parser.parse_args()

    evaluate_4p(args.model, n_games=args.games, device_str=args.device)


if __name__ == "__main__":
    main()
