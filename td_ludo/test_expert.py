import sys
import os
from src.heuristic_bot import BOT_REGISTRY
from benchmark_models import run_evaluation

try:
    from train import _player, _game_db
except ImportError:
    pass

import random
import td_ludo_cpp as ludo_cpp
from src.config import MAX_MOVES_PER_GAME
from collections import defaultdict

def play_match(bot1_type, bot2_type, num_games=1000):
    wins = 0
    total = num_games
    
    bot1_class = BOT_REGISTRY[bot1_type]
    bot2_class = BOT_REGISTRY[bot2_type]
    
    for _ in range(num_games):
        b1_idx = random.choice([0, 2])
        b2_idx = 2 if b1_idx == 0 else 0
        
        bots = {
            b1_idx: bot1_class(b1_idx),
            b2_idx: bot2_class(b2_idx)
        }
        
        state = ludo_cpp.create_initial_state_2p()
        move_count = 0
        consecutive_sixes = [0, 0, 0, 0]
        
        while not state.is_terminal and move_count < MAX_MOVES_PER_GAME:
            current_p = state.current_player
            
            if not state.active_players[current_p]:
                next_p = (current_p + 1) % 4
                while not state.active_players[next_p]:
                    next_p = (next_p + 1) % 4
                state.current_player = next_p
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
                    while not state.active_players[state.current_player]:
                        state.current_player = (state.current_player + 1) % 4
                    continue
            
            legal_moves = ludo_cpp.get_legal_moves(state)
            if not legal_moves:
                state.current_dice_roll = 0
                state.current_player = (current_p + 1) % 4
                while not state.active_players[state.current_player]:
                    state.current_player = (state.current_player + 1) % 4
                continue
            
            action = bots[current_p].select_move(state, legal_moves)
            if action not in legal_moves:
                action = random.choice(legal_moves)
                
            state = ludo_cpp.apply_move(state, action)
            move_count += 1
            
        winner = ludo_cpp.get_winner(state)
        if winner == b1_idx:
            wins += 1
            
    print(f"{bot1_type} vs {bot2_type}: {bot1_type} won {wins}/{total} ({(wins/total)*100:.1f}%)")

if __name__ == "__main__":
    play_match('Expert', 'Heuristic', 500)
    play_match('Expert', 'Random', 500)
