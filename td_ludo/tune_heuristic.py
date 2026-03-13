import sys
import os
import random
import multiprocessing
import itertools
from collections import defaultdict
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import td_ludo_cpp as ludo_cpp
from src.heuristic_bot import HeuristicLudoBot, W_WIN_GAME, W_FINISH_TOKEN, W_EXIT_BASE
from src.config import MAX_MOVES_PER_GAME

# A configurable bot that accepts custom weights
class ConfigurableBot(HeuristicLudoBot):
    def __init__(self, player_id, weights):
        super().__init__(player_id)
        self.w_cut = weights.get('cut', 6000.0)
        self.w_danger = weights.get('danger', -2000.0)
        self.w_progress = weights.get('progress', 10.0)
        self.w_safe = weights.get('safe', 800.0)
        self.w_stack = weights.get('stack', 150.0)
        
    def _evaluate(self, prev, curr, token_idx, player, w_cut, w_danger):
        # Override the base weights with our optimized ones
        return super()._evaluate(prev, curr, token_idx, player, self.w_cut, self.w_danger)
        
def play_game(bot_configs):
    """
    Play 1 game with the given 4 bot configs.
    bot_configs: list of 4 weight dicts
    Returns the index of the winning bot (0-3), or -1 for draw.
    """
    bots = [ConfigurableBot(i, bot_configs[i]) for i in range(4)]
    state = ludo_cpp.create_initial_state()
    move_count = 0
    consecutive_sixes = [0, 0, 0, 0]
    
    while not state.is_terminal and move_count < MAX_MOVES_PER_GAME:
        current_p = state.current_player
        
        if not state.active_players[current_p]:
            state.current_player = (current_p + 1) % 4
            continue
            
        if state.current_dice_roll == 0:
            roll = random.randint(1, 6)
            state.current_dice_roll = roll
            if roll == 6:
                consecutive_sixes[current_p] += 1
            else:
                consecutive_sixes[current_p] = 0
            
            if consecutive_sixes[current_p] >= 3:
                state.current_dice_roll = 0
                consecutive_sixes[current_p] = 0
                state.current_player = (current_p + 1) % 4
                continue
                
        legal_moves = ludo_cpp.get_legal_moves(state)
        if not legal_moves:
            state.current_dice_roll = 0
            state.current_player = (current_p + 1) % 4
            continue
            
        action = bots[current_p].select_move(state, legal_moves)
        if action not in legal_moves:
            action = random.choice(legal_moves)
            
        state = ludo_cpp.apply_move(state, action)
        move_count += 1
        
    return ludo_cpp.get_winner(state)

def evaluate_candidate(candidate_weights, baseline_weights, num_games=200):
    wins = 0
    draws = 0
    total = num_games
    
    for _ in range(num_games):
        # 2 candidates vs 2 baselines
        configs = [candidate_weights, baseline_weights, candidate_weights, baseline_weights]
        random.shuffle(configs)
        
        candidate_indices = [i for i, c in enumerate(configs) if c == candidate_weights]
        
        winner = play_game(configs)
        if winner in candidate_indices:
            wins += 1
        elif winner == -1:
            draws += 1
            
    win_rate = wins / total
    return candidate_weights, win_rate

if __name__ == "__main__":
    baseline = {
        'cut': 6000.0,
        'danger': -2000.0,
        'progress': 10.0,
        'safe': 800.0,
        'stack': 150.0
    }
    
    # Generate 50 random variations of the heuristic
    print("Generating 50 random heuristic profiles for genetic optimization...")
    candidates = []
    for _ in range(50):
        c = {
            'cut': random.uniform(2000, 15000),
            'danger': random.uniform(-8000, -500),
            'progress': random.uniform(1, 50),
            'safe': random.uniform(200, 3000),
            'stack': random.uniform(50, 1000)
        }
        candidates.append(c)
        
    print(f"Staging tournament of {len(candidates)} heuristic bots vs Baseline...")
    
    import concurrent.futures
    best_wr = 0
    best_weights = None
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(evaluate_candidate, c, baseline) for c in candidates]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(candidates)):
            weights, wr = future.result()
            if wr > best_wr:
                best_wr = wr
                best_weights = weights
                
    print("\n" + "="*50)
    print(f"OPTIMIZATION COMPLETE - Best Candidate Win Rate: {best_wr*100:.1f}%")
    print("New Optimal Weights:")
    for k, v in best_weights.items():
        print(f"W_{k.upper()} = {v:.1f}")
    print("="*50)
