"""
Pure Model Evaluation Test
--------------------------
Tests the trained v3 model against 3 random heuristic bots.
No MCTS, no training - just pure neural network inference.

Usage:
    python -m src.test_pure_model [--games 100] [--checkpoint path]
"""

import os
import sys
import random
import time
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ludo_cpp
from src.model_v3 import AlphaLudoV3
from src.heuristic_bot import HeuristicLudoBot, AggressiveBot, DefensiveBot, RacingBot
from src.tensor_utils_mastery import state_to_tensor_mastery
from src.config import MAIN_CKPT_PATH


class RandomBot:
    """Simple random move selector."""
    def __init__(self, player_id=None):
        self.player_id = player_id
    
    def select_move(self, state, legal_moves):
        if not legal_moves:
            return -1
        return random.choice(legal_moves)


def get_all_bots():
    """Returns all available bot types."""
    return {
        'Heuristic': HeuristicLudoBot,
        'Aggressive': AggressiveBot,
        'Defensive': DefensiveBot,
        'Racing': RacingBot,
        'Random': RandomBot,
    }


def run_pure_model_game(model, device, model_player_idx, bots):
    """
    Run a single game with the model as one player and bots as others.
    
    Args:
        model: The trained AlphaLudoV3 model
        device: torch device
        model_player_idx: Which player slot (0-3) the model occupies
        bots: Dict mapping player_idx -> (bot_instance, bot_name) for non-model players
        
    Returns:
        winner: Index of winning player
        model_won: True if model won
        total_moves: Number of moves in the game
    """
    state = ludo_cpp.GameState()
    total_moves = 0
    max_moves = 2000  # Safety limit
    consecutive_sixes = [0, 0, 0, 0]
    
    while not state.is_terminal and total_moves < max_moves:
        current_player = state.current_player
        
        # Roll dice if needed
        if state.current_dice_roll == 0:
            state.current_dice_roll = np.random.randint(1, 7)
            if state.current_dice_roll == 6:
                consecutive_sixes[current_player] += 1
            else:
                consecutive_sixes[current_player] = 0
            
            # 3 sixes penalty
            if consecutive_sixes[current_player] >= 3:
                state.current_player = (state.current_player + 1) % 4
                state.current_dice_roll = 0
                consecutive_sixes[current_player] = 0
                continue
        
        # Get legal moves
        legal_moves = ludo_cpp.get_legal_moves(state)
        
        if len(legal_moves) == 0:
            # No valid moves, pass turn
            state.current_player = (state.current_player + 1) % 4
            state.current_dice_roll = 0
            continue
        
        # Select action
        if current_player == model_player_idx:
            # Model's turn - use pure policy (no MCTS)
            state_tensor = state_to_tensor_mastery(state)
            input_tensor = state_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
            
            with torch.no_grad():
                policy, _ = model.forward_policy_value(input_tensor)
            
            # Mask illegal moves
            probs = policy[0].cpu().numpy()
            masked_probs = np.zeros(4)
            for m in legal_moves:
                masked_probs[m] = probs[m]
            
            if masked_probs.sum() > 0:
                masked_probs /= masked_probs.sum()
                action = np.argmax(masked_probs)  # Greedy (best move)
            else:
                action = random.choice(legal_moves)
        else:
            # Bot's turn
            bot, _ = bots[current_player]
            action = bot.select_move(state, legal_moves)
        
        # Apply move
        state = ludo_cpp.apply_move(state, action)
        total_moves += 1
        
        # Check if game ended (after move, not from terminal check)
        if state.scores[current_player] == 4:
            break
    
    # Determine winner
    winner = -1
    for p in range(4):
        if state.scores[p] == 4:
            winner = p
            break
    
    # If no winner after max moves, pick highest scorer
    if winner == -1:
        winner = np.argmax(state.scores)
    
    model_won = (winner == model_player_idx)
    return winner, model_won, total_moves


def run_test_league(model, device, num_games=100, seed=None):
    """
    Run a test league: model vs 3 random bots for num_games games.
    
    Returns detailed stats.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    all_bot_types = ['Heuristic', 'Aggressive', 'Defensive', 'Racing', 'Random']
    bot_classes = get_all_bots()
    
    results = {
        'model_wins': 0,
        'model_losses': 0,
        'total_games': 0,
        'wins_by_position': {0: 0, 1: 0, 2: 0, 3: 0},
        'games_by_position': {0: 0, 1: 0, 2: 0, 3: 0},
        'wins_vs_bot': {},
        'games_vs_bot': {},
        'total_moves': [],
        'game_details': []
    }
    
    print(f"\n{'='*60}")
    print(f"  PURE MODEL EVALUATION - {num_games} Games")
    print(f"{'='*60}")
    print(f"Model: {MAIN_CKPT_PATH}")
    print(f"Mode: Pure Policy (Greedy, No MCTS)")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    for game_idx in range(num_games):
        # Random position for model
        model_player_idx = random.randint(0, 3)
        
        # Pick 3 random bots for opponents
        opponent_slots = [i for i in range(4) if i != model_player_idx]
        opponent_types = random.sample(all_bot_types, 3)
        
        bots = {}
        game_opponents = []
        for slot, bot_type in zip(opponent_slots, opponent_types):
            bot_instance = bot_classes[bot_type](player_id=slot)
            bots[slot] = (bot_instance, bot_type)
            game_opponents.append(bot_type)
            
            # Track opponent types
            if bot_type not in results['wins_vs_bot']:
                results['wins_vs_bot'][bot_type] = 0
                results['games_vs_bot'][bot_type] = 0
            results['games_vs_bot'][bot_type] += 1
        
        # Run game
        winner, model_won, total_moves = run_pure_model_game(
            model, device, model_player_idx, bots
        )
        
        # Update stats
        results['total_games'] += 1
        results['games_by_position'][model_player_idx] += 1
        results['total_moves'].append(total_moves)
        
        if model_won:
            results['model_wins'] += 1
            results['wins_by_position'][model_player_idx] += 1
            for bot_type in game_opponents:
                results['wins_vs_bot'][bot_type] += 1
        else:
            results['model_losses'] += 1
        
        # Store game detail
        results['game_details'].append({
            'game_id': game_idx + 1,
            'model_position': model_player_idx,
            'opponents': game_opponents,
            'winner': winner,
            'model_won': model_won,
            'moves': total_moves
        })
        
        # Progress update
        if (game_idx + 1) % 10 == 0:
            win_rate = results['model_wins'] / results['total_games'] * 100
            elapsed = time.time() - start_time
            games_per_sec = results['total_games'] / elapsed
            print(f"  Game {game_idx + 1:3d}/{num_games}: "
                  f"Win Rate = {win_rate:.1f}% | "
                  f"Speed = {games_per_sec:.1f} games/s")
    
    elapsed = time.time() - start_time
    
    # Calculate final stats
    win_rate = results['model_wins'] / results['total_games'] * 100
    avg_moves = np.mean(results['total_moves'])
    
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  Total Games:        {results['total_games']}")
    print(f"  Model Wins:         {results['model_wins']}")
    print(f"  Model Losses:       {results['model_losses']}")
    print(f"  Win Rate:           {win_rate:.1f}%")
    print(f"  Expected (Random):  25.0%")
    print(f"  Avg Moves/Game:     {avg_moves:.1f}")
    print(f"  Total Time:         {elapsed:.1f}s")
    print(f"  Speed:              {results['total_games']/elapsed:.1f} games/s")
    print(f"\n  --- Win Rate by Position ---")
    for pos in range(4):
        games = results['games_by_position'][pos]
        wins = results['wins_by_position'][pos]
        rate = (wins / games * 100) if games > 0 else 0
        print(f"  Position {pos}: {wins}/{games} ({rate:.1f}%)")
    
    print(f"\n  --- Win Rate vs Bot Type ---")
    for bot_type in sorted(results['games_vs_bot'].keys()):
        games = results['games_vs_bot'][bot_type]
        wins = results['wins_vs_bot'][bot_type]
        rate = (wins / games * 100) if games > 0 else 0
        print(f"  vs {bot_type:12s}: {wins}/{games} ({rate:.1f}%)")
    
    print(f"{'='*60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Pure Model Evaluation Test')
    parser.add_argument('--games', type=int, default=100, 
                        help='Number of games to play (default: 100)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (default: uses config)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    ckpt_path = args.checkpoint or MAIN_CKPT_PATH
    
    model = AlphaLudoV3(num_res_blocks=10, num_channels=128)
    model.to(device)
    model.eval()
    
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"Loaded checkpoint: {ckpt_path}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Using random weights!")
    else:
        print(f"No checkpoint found at {ckpt_path}")
        print("Using random weights!")
    
    # Run test
    results = run_test_league(model, device, num_games=args.games, seed=args.seed)
    
    # Verdict
    win_rate = results['model_wins'] / results['total_games'] * 100
    if win_rate > 35:
        print("✅ VERDICT: Model is significantly better than random!")
    elif win_rate > 28:
        print("⚠️  VERDICT: Model shows slight improvement over random.")
    elif win_rate > 22:
        print("🔶 VERDICT: Model is performing at random level.")
    else:
        print("❌ VERDICT: Model is performing worse than random!")
    
    return results


if __name__ == "__main__":
    main()
