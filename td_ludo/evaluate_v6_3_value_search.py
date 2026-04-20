"""
1-ply Value-Search Evaluator for V6.3.

For each decision, instead of `argmax π(a|s)`, this evaluator:
  1. Enumerates every legal move
  2. Applies each move to a copy of the state
  3. Runs the value head on the post-move state
  4. Picks the move that maximizes "AI's expected outcome"

Key signs:
  - If the post-move state has current_player == us (bonus turn), value is
    already from our perspective: pick max.
  - If current_player switched to opponent, value is from opponent's
    perspective: pick min (= max(-value)).

Compares against pure policy argmax.

Usage:
  python3 evaluate_v6_3_value_search.py \
      --model checkpoints/ac_v6_3_capture/model_latest.pt \
      --games 200 \
      --bots Expert \
      --mode value_search   # or 'policy' for baseline
"""

import argparse
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import td_ludo_cpp as ludo_cpp
from td_ludo.models.v6_3 import AlphaLudoV63
from src.heuristic_bot import (
    HeuristicLudoBot, AggressiveBot, DefensiveBot, RacingBot, RandomBot,
    get_bot, BOT_REGISTRY,
)
from src.config import MAX_MOVES_PER_GAME


def select_move_policy(model, device, state, legal_moves, consec_sixes):
    """Pure policy argmax (baseline)."""
    if len(legal_moves) == 1:
        return legal_moves[0]
    cp = state.current_player
    state_tensor = ludo_cpp.encode_state_v6_3(state, consec_sixes[cp])
    legal_mask = np.zeros(4, dtype=np.float32)
    for m in legal_moves:
        legal_mask[m] = 1.0
    with torch.no_grad():
        s_t = torch.from_numpy(state_tensor).unsqueeze(0).to(device, dtype=torch.float32)
        m_t = torch.from_numpy(legal_mask).unsqueeze(0).to(device, dtype=torch.float32)
        logits = model.forward_policy_only(s_t, m_t)
        action = int(logits.argmax(dim=1).item())
    return action if action in legal_moves else random.choice(legal_moves)


def select_move_value_search(model, device, state, legal_moves, consec_sixes):
    """1-ply value-search: pick move that maximizes our post-move expected value.

    For each legal move:
      - apply it -> get post-move state
      - encode + run value head
      - if post-move current_player == us, value is OURS -> use as-is
      - if current_player == opponent, value is THEIRS -> negate
    Pick move with highest "our value".
    """
    if len(legal_moves) == 1:
        return legal_moves[0]

    me = state.current_player
    candidates = []
    for move in legal_moves:
        # Apply move on a copy (apply_move returns a new GameState)
        new_state = ludo_cpp.apply_move(state, move)

        if new_state.is_terminal:
            # Did we win immediately?
            winner = ludo_cpp.get_winner(new_state)
            our_value = 1e6 if winner == me else -1e6  # treat win/loss as ±inf
            candidates.append((our_value, move))
            continue

        # Encode post-move state and run value head
        new_cp = new_state.current_player
        # consecutive_sixes for the new state's current player — best-effort:
        # if it's still us and dice was 6, bump our counter; otherwise 0.
        if new_cp == me and state.current_dice_roll == 6:
            new_consec = min(consec_sixes[me] + 1, 2)
        else:
            new_consec = 0

        st_tensor = ludo_cpp.encode_state_v6_3(new_state, int(new_consec))
        # Permissive mask — value doesn't depend on it but model API needs one
        mask = np.ones(4, dtype=np.float32)
        with torch.no_grad():
            s_t = torch.from_numpy(st_tensor).unsqueeze(0).to(device, dtype=torch.float32)
            m_t = torch.from_numpy(mask).unsqueeze(0).to(device, dtype=torch.float32)
            out = model(s_t, m_t)
            v = float(out[1].squeeze().item())

        # Translate to "our value"
        our_value = v if new_cp == me else -v
        candidates.append((our_value, move))

    # Pick best
    best_value, best_move = max(candidates, key=lambda x: x[0])
    return best_move


def is_endgame(state, threshold):
    """Endgame = total tokens already scored across all players >= threshold."""
    return sum(int(state.scores[p]) for p in range(4)) >= threshold


def evaluate(model, device, num_games, mode, bot_types, verbose=True,
             endgame_threshold=4):
    model.eval()

    if mode == 'value_search':
        select_fn = lambda *a: select_move_value_search(*a)
    elif mode == 'hybrid':
        # Use value search only in endgame
        def select_fn(model, device, state, legal_moves, consec):
            if is_endgame(state, endgame_threshold):
                return select_move_value_search(model, device, state, legal_moves, consec)
            return select_move_policy(model, device, state, legal_moves, consec)
    else:  # 'policy'
        select_fn = lambda *a: select_move_policy(*a)
    available_types = bot_types or list(BOT_REGISTRY.keys())

    wins = 0
    total = 0
    game_lengths = []
    per_bot = defaultdict(lambda: {'wins': 0, 'games': 0, 'lengths': []})

    start_time = time.time()
    for game_idx in range(num_games):
        model_player = random.choice([0, 2])
        opp_player = 2 if model_player == 0 else 0
        bot_type = random.choice(available_types)
        opp_bot = get_bot(bot_type, player_id=opp_player)

        state = ludo_cpp.create_initial_state_2p()
        consec = [0, 0, 0, 0]
        move_count = 0

        while not state.is_terminal and move_count < MAX_MOVES_PER_GAME:
            cp = state.current_player
            if not state.active_players[cp]:
                nxt = (cp + 1) % 4
                while not state.active_players[nxt]:
                    nxt = (nxt + 1) % 4
                state.current_player = nxt
                continue

            if state.current_dice_roll == 0:
                state.current_dice_roll = random.randint(1, 6)
                if state.current_dice_roll == 6:
                    consec[cp] += 1
                else:
                    consec[cp] = 0
                if consec[cp] >= 3:
                    nxt = (cp + 1) % 4
                    while not state.active_players[nxt]:
                        nxt = (nxt + 1) % 4
                    state.current_player = nxt
                    state.current_dice_roll = 0
                    consec[cp] = 0
                    continue

            legal = ludo_cpp.get_legal_moves(state)
            if not legal:
                nxt = (cp + 1) % 4
                while not state.active_players[nxt]:
                    nxt = (nxt + 1) % 4
                state.current_player = nxt
                state.current_dice_roll = 0
                continue

            if cp == model_player:
                action = select_fn(model, device, state, legal, consec)
            else:
                action = opp_bot.select_move(state, legal)

            state = ludo_cpp.apply_move(state, action)
            move_count += 1

        winner = ludo_cpp.get_winner(state) if state.is_terminal else -1
        won = (winner == model_player)
        if won:
            wins += 1
        total += 1
        game_lengths.append(move_count)
        per_bot[bot_type]['games'] += 1
        if won:
            per_bot[bot_type]['wins'] += 1
        per_bot[bot_type]['lengths'].append(move_count)

        if verbose and (game_idx + 1) % 25 == 0:
            wr = wins / total * 100
            elapsed = time.time() - start_time
            gpm = (game_idx + 1) / (elapsed / 60) if elapsed > 0 else 0
            print(f"  [{game_idx+1}/{num_games}] WR: {wr:.1f}% ({wins}/{total}) | {gpm:.0f} GPM")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  RESULTS — mode={mode}")
    print(f"{'='*60}")
    print(f"  Overall: {wins}/{total} = {wins/total*100:.1f}% WR")
    print(f"  Avg game length: {np.mean(game_lengths):.0f} moves")
    print(f"  Duration: {elapsed:.1f}s ({total/(elapsed/60):.0f} GPM)")
    print(f"\n  Per-bot:")
    for bt in sorted(per_bot.keys()):
        s = per_bot[bt]
        print(f"    vs {bt:<12s}: {s['wins']}/{s['games']} = {s['wins']/s['games']*100:.1f}%")
    print(f"{'='*60}\n")
    return wins / total if total > 0 else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--games', type=int, default=200)
    parser.add_argument('--bots', type=str, nargs='+', default=['Expert'])
    parser.add_argument('--mode', choices=['policy', 'value_search', 'hybrid'],
                        default='value_search')
    parser.add_argument('--endgame-threshold', type=int, default=4,
                        help='For hybrid mode: # of tokens already scored (across '
                             'all players) before switching to value search. '
                             'Default 4 = mid-late game.')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    model = AlphaLudoV63(num_res_blocks=10, num_channels=128, in_channels=27)
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(sd)
    model.to(device).eval()
    print(f"Loaded {args.model}")
    print(f"Mode: {args.mode}, Games: {args.games}, Bots: {args.bots}"
          + (f", Endgame threshold: {args.endgame_threshold}" if args.mode == 'hybrid' else ''))
    evaluate(model, device, args.games, args.mode, args.bots,
             endgame_threshold=args.endgame_threshold)
