"""
TD-Ludo Evaluator — Policy-Based Model Evaluation Against Bots

Runs games with the model using greedy policy selection (argmax π(a|s))
against heuristic bots to measure true skill.

Reports overall win rate + per-opponent-type breakdowns.
"""

import os
import sys
import random
import time
import torch
import torch.nn.functional as F
import argparse
import numpy as np
from collections import defaultdict

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import td_ludo_cpp as ludo_cpp
from src.model import AlphaLudoV5
from src.heuristic_bot import (
    HeuristicLudoBot, AggressiveBot, DefensiveBot, RacingBot, RandomBot,
    get_bot, BOT_REGISTRY,
)
from src.config import MAX_MOVES_PER_GAME


def evaluate_model(model, device, num_games=200, verbose=False,
                   bot_types=None):
    """
    Evaluate model against a mix of heuristic bots.
    
    The model plays as one player (random seat), the other seat
    is filled with a random bot type.
    
    Move selection: argmax π(a|s) — greedy policy (no exploration).
    This is much faster than the old V(s') approach (1 forward pass vs N).
    """
    model.eval()
    available_types = bot_types or list(BOT_REGISTRY.keys())
    
    wins = 0
    total = 0
    game_lengths = []
    
    # Per-bot-type tracking
    per_bot = defaultdict(lambda: {'wins': 0, 'games': 0, 'lengths': []})
    
    start_time = time.time()
    
    for game_idx in range(num_games):
        # 2-player setup: Model at P0 or P2, opponent at the other
        model_player = random.choice([0, 2])
        opp_player = 2 if model_player == 0 else 0
        inactive_players = {1, 3}  # P1 and P3 always inactive
        
        bot_type = random.choice(available_types)
        player_bots = {opp_player: get_bot(bot_type, player_id=opp_player)}
        game_bot_types = {opp_player: bot_type}
        
        # Play game (2-player: P0 vs P2)
        state = ludo_cpp.create_initial_state_2p()
        consecutive_sixes = [0, 0, 0, 0]
        move_count = 0
        
        while not state.is_terminal and move_count < MAX_MOVES_PER_GAME:
            current_player = state.current_player
            
            # Skip inactive players
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
                # Model move — greedy policy (argmax π(a|s))
                if len(legal_moves) == 1:
                    action = legal_moves[0]
                else:
                    # Encode state and create legal mask
                    state_tensor = ludo_cpp.encode_state(state)
                    legal_mask = np.zeros(4, dtype=np.float32)
                    for m in legal_moves:
                        legal_mask[m] = 1.0
                    
                    with torch.no_grad():
                        s_t = torch.from_numpy(state_tensor).unsqueeze(0).to(device, dtype=torch.float32)
                        m_t = torch.from_numpy(legal_mask).unsqueeze(0).to(device, dtype=torch.float32)
                        policy_logits = model.forward_policy_only(s_t, m_t)
                        action = policy_logits.argmax(dim=1).item()
                    
                    # Safety check
                    if action not in legal_moves:
                        action = random.choice(legal_moves)
            else:
                # Bot move
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
        
        # Track per-bot-type stats
        for p, bt in game_bot_types.items():
            per_bot[bt]['games'] += 1
            if model_won:
                per_bot[bt]['wins'] += 1
            per_bot[bt]['lengths'].append(move_count)
        
        if verbose and (game_idx + 1) % 50 == 0:
            wr = wins / total * 100
            elapsed = time.time() - start_time
            gpm = (game_idx + 1) / (elapsed / 60) if elapsed > 0 else 0
            print(f"  [{game_idx+1}/{num_games}] Win Rate: {wr:.1f}% ({wins}/{total}) | {gpm:.0f} GPM")
    
    elapsed = time.time() - start_time
    win_rate = wins / total if total > 0 else 0.0
    
    # Build per-bot breakdown
    bot_breakdown = {}
    for bt, stats in sorted(per_bot.items()):
        bt_wr = stats['wins'] / stats['games'] if stats['games'] > 0 else 0
        bot_breakdown[bt] = {
            'win_rate': round(bt_wr * 100, 1),
            'wins': stats['wins'],
            'games': stats['games'],
            'avg_length': round(float(np.mean(stats['lengths'])), 0) if stats['lengths'] else 0,
        }
    
    return {
        'win_rate': win_rate,
        'wins': wins,
        'total': total,
        'avg_game_length': float(np.mean(game_lengths)) if game_lengths else 0,
        'win_rate_percent': round(win_rate * 100, 1),
        'elapsed_seconds': round(elapsed, 1),
        'games_per_minute': round(total / (elapsed / 60), 1) if elapsed > 0 else 0,
        'per_bot': bot_breakdown,
    }


def main():
    parser = argparse.ArgumentParser(description='TD-Ludo Model Evaluator (Actor-Critic)')
    parser.add_argument('--model', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--games', type=int, default=500, help='Number of evaluation games')
    parser.add_argument('--device', type=str, default='mps', help='Device (cpu/mps)')
    parser.add_argument('--bots', type=str, nargs='+', default=None,
                        help='Bot types to evaluate against (default: all)')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.backends.mps.is_available() else 'cpu')
    print(f"[Evaluator] Device: {device}")
    
    # Load model — AlphaLudoV5
    model = AlphaLudoV5(num_res_blocks=10, num_channels=128)
    model.to(device)
    
    if args.model:
        ckpt = torch.load(args.model, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        print(f"[Evaluator] Loaded model from {args.model}")
    else:
        from src.config import MAIN_CKPT_PATH
        if os.path.exists(MAIN_CKPT_PATH):
            ckpt = torch.load(MAIN_CKPT_PATH, map_location=device, weights_only=False)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            else:
                model.load_state_dict(ckpt)
            print(f"[Evaluator] Loaded model from {MAIN_CKPT_PATH}")
        else:
            print("[Evaluator] No checkpoint found, evaluating random model")
    
    model.eval()
    
    bot_list = args.bots or list(BOT_REGISTRY.keys())
    
    print(f"\n{'='*60}")
    print(f"  TD-Ludo Evaluation (Actor-Critic): {args.games} games")
    print(f"  Opponents: {', '.join(bot_list)}")
    print(f"{'='*60}\n")
    
    results = evaluate_model(model, device, num_games=args.games, verbose=True,
                            bot_types=bot_list)
    
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Overall Win Rate: {results['win_rate_percent']}% ({results['wins']}/{results['total']})")
    print(f"  Avg Game Length: {results['avg_game_length']:.0f} moves")
    print(f"  Duration: {results['elapsed_seconds']:.1f}s ({results['games_per_minute']:.0f} GPM)")
    print(f"  Random Baseline: 50.0% (2-player)")
    
    if results['per_bot']:
        print(f"\n  {'─'*50}")
        print(f"  Per-Opponent Breakdown:")
        print(f"  {'─'*50}")
        for bt, stats in sorted(results['per_bot'].items()):
            print(f"    vs {bt:12s} → {stats['win_rate']:5.1f}% "
                  f"({stats['wins']}/{stats['games']} games, "
                  f"~{stats['avg_length']:.0f} moves)")
    
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
