"""
TD-Ludo V9 Evaluator — Slim CNN + Temporal Transformer evaluation.

Same evaluation protocol as V8 (greedy policy against bots), but
uses V9's 14-channel encoder and 80-dim embeddings.
"""

import os
import sys
import random
import time
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import td_ludo_cpp as ludo_cpp
from src.model_v9 import AlphaLudoV9
from src.fast_actor_v62 import TurnHistory
from src.heuristic_bot import get_bot, BOT_REGISTRY
from src.config import MAX_MOVES_PER_GAME


def evaluate_v62_model(model, device, num_games=200, verbose=False,
                      bot_types=None, context_length=16):
    """
    Evaluate V9 model against heuristic bots.
    Uses greedy policy (argmax) with a rolling 24ch grid context window.
    """
    model.eval()
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
        player_bots = {opp_player: get_bot(bot_type, player_id=opp_player)}

        state = ludo_cpp.create_initial_state_2p()
        consecutive_sixes = [0, 0, 0, 0]
        move_count = 0

        history = TurnHistory(context_length, 128)
        last_action = 4

        while not state.is_terminal and move_count < MAX_MOVES_PER_GAME:
            current_player = state.current_player

            if not state.active_players[current_player]:
                next_p = (current_player + 1) % 4
                while not state.active_players[next_p]:
                    next_p = (next_p + 1) % 4
                state.current_player = next_p
                continue

            if state.current_dice_roll == 0:
                state.current_dice_roll = random.randint(1, 6)
                cp = state.current_player
                if state.current_dice_roll == 6:
                    consecutive_sixes[cp] += 1
                else:
                    consecutive_sixes[cp] = 0
                if consecutive_sixes[cp] >= 3:
                    next_p = (cp + 1) % 4
                    while not state.active_players[next_p]:
                        next_p = (next_p + 1) % 4
                    state.current_player = next_p
                    state.current_dice_roll = 0
                    consecutive_sixes[cp] = 0
                    continue

            legal_moves = ludo_cpp.get_legal_moves(state)
            if len(legal_moves) == 0:
                next_p = (state.current_player + 1) % 4
                while not state.active_players[next_p]:
                    next_p = (next_p + 1) % 4
                state.current_player = next_p
                state.current_dice_roll = 0
                continue

            if current_player == model_player:
                if len(legal_moves) == 1:
                    action = legal_moves[0]
                else:
                    # V9: 14-channel encoding
                    grid = ludo_cpp.encode_state_v6(state)
                    history.add_turn(grid, action=last_action, cnn_feature=None)

                    with torch.no_grad():
                        grid_t = torch.from_numpy(grid).unsqueeze(0).to(device, dtype=torch.float32)
                        cnn_feat = model.compute_single_cnn_features(grid_t)
                        history._cnn_features[-1] = cnn_feat.cpu().numpy()[0]

                    cached_cnn, seq_acts, seq_mask = history.get_cached_sequence()

                    legal_mask = np.zeros(4, dtype=np.float32)
                    for m in legal_moves:
                        legal_mask[m] = 1.0

                    with torch.no_grad():
                        t_cached = torch.from_numpy(cached_cnn).unsqueeze(0).to(device, dtype=torch.float32)
                        t_acts = torch.from_numpy(seq_acts).unsqueeze(0).to(device)
                        t_mask = torch.from_numpy(seq_mask).unsqueeze(0).to(device)
                        t_legal = torch.from_numpy(legal_mask).unsqueeze(0).to(device, dtype=torch.float32)

                        policy_logits = model.forward_policy_only_cached(
                            t_cached, t_acts, t_mask, t_legal
                        )
                        action = policy_logits.argmax(dim=1).item()

                    if action not in legal_moves:
                        action = random.choice(legal_moves)

                last_action = action
            else:
                bot = player_bots[current_player]
                action = bot.select_move(state, legal_moves)

            state = ludo_cpp.apply_move(state, action)
            move_count += 1

        winner = ludo_cpp.get_winner(state) if state.is_terminal else -1
        model_won = (winner == model_player)

        if model_won:
            wins += 1
        total += 1
        game_lengths.append(move_count)

        per_bot[bot_type]['games'] += 1
        if model_won:
            per_bot[bot_type]['wins'] += 1
        per_bot[bot_type]['lengths'].append(move_count)

        if verbose and (game_idx + 1) % 50 == 0:
            wr = wins / total * 100
            elapsed = time.time() - start_time
            gpm = (game_idx + 1) / (elapsed / 60) if elapsed > 0 else 0
            print(f"  [{game_idx+1}/{num_games}] Win Rate: {wr:.1f}% ({wins}/{total}) | {gpm:.0f} GPM")

    elapsed = time.time() - start_time
    win_rate = wins / total if total > 0 else 0.0

    results = {
        'win_rate': win_rate,
        'win_rate_percent': round(win_rate * 100, 1),
        'wins': wins,
        'total': total,
        'avg_game_length': round(np.mean(game_lengths), 1) if game_lengths else 0,
        'elapsed': elapsed,
        'per_bot': {},
    }

    for bt, stats in per_bot.items():
        g = stats['games']
        w = stats['wins']
        results['per_bot'][bt] = {
            'games': g,
            'wins': w,
            'win_rate': round(w / g * 100, 1) if g > 0 else 0,
        }

    if verbose:
        print(f"\n{'='*50}")
        print(f"  V9 Evaluation Results ({total} games)")
        print(f"  Overall Win Rate: {results['win_rate_percent']}%")
        print(f"  Avg Game Length: {results['avg_game_length']} moves")
        for bt, s in results['per_bot'].items():
            print(f"  vs {bt}: {s['win_rate']}% ({s['wins']}/{s['games']})")
        print(f"  Time: {elapsed:.1f}s | {total / (elapsed / 60):.0f} GPM")
        print(f"{'='*50}")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate V9 Slim CNN+Transformer Model')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to V9 checkpoint')
    parser.add_argument('--games', type=int, default=500, help='Number of games')
    parser.add_argument('--device', type=str, default=None, help='Device')
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = AlphaLudoV9()

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        model.load_state_dict(sd)

    model.to(device)

    results = evaluate_v62_model(model, device, num_games=args.games, verbose=True)
