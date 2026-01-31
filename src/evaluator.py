"""
Evaluator for AlphaLudo.

Evaluates model performance against previous versions and baseline bots.
"""

import numpy as np
import torch
import ludo_cpp
from mcts import MCTS, get_action_probs


class GreedyBot:
    """
    Simple baseline bot that:
    1. Cuts if possible.
    2. Otherwise moves the token closest to home.
    """
    
    def select_action(self, state):
        """
        Select an action using greedy strategy.
        
        Args:
            state: GameState with dice rolled.
            
        Returns:
            Token index to move.
        """
        legal_moves = ludo_cpp.get_legal_moves(state)
        
        if len(legal_moves) == 0:
            return None
        
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        player = state.current_player
        roll = state.current_dice_roll
        
        # Check for cut opportunities
        for move in legal_moves:
            next_state = ludo_cpp.apply_move(state, move)
            # Check if we cut someone (their token went to base)
            for other_p in range(4):
                if other_p == player:
                    continue
                for t in range(4):
                    old_pos = state.player_positions[other_p][t]
                    new_pos = next_state.player_positions[other_p][t]
                    if old_pos != -1 and old_pos != 99 and new_pos == -1:
                        # We cut them!
                        return move
        
        # No cut possible, move token closest to home
        best_move = legal_moves[0]
        best_progress = -1
        
        for move in legal_moves:
            pos = state.player_positions[player][move]
            if pos == -1:
                # Coming out of base
                progress = 0
            else:
                progress = pos  # Higher position = closer to home
            
            if progress > best_progress:
                best_progress = progress
                best_move = move
        
        return best_move


def play_match(player1_fn, player2_fn, player3_fn, player4_fn):
    """
    Play a single 4-player game.
    
    Args:
        player1_fn to player4_fn: Functions that take state and return action.
        
    Returns:
        Winner index (0-3) or -1 for draw.
    """
    player_fns = [player1_fn, player2_fn, player3_fn, player4_fn]
    state = ludo_cpp.create_initial_state()
    max_moves = 1000
    move_count = 0
    
    while not state.is_terminal and move_count < max_moves:
        # Roll dice
        state.current_dice_roll = np.random.randint(1, 7)
        
        legal_moves = ludo_cpp.get_legal_moves(state)
        if len(legal_moves) == 0:
            state.current_player = (state.current_player + 1) % 4
            state.current_dice_roll = 0
            continue
        
        # Get action from current player
        player_fn = player_fns[state.current_player]
        action = player_fn(state)
        
        if action is None or action not in legal_moves:
            action = np.random.choice(legal_moves)
        
        state = ludo_cpp.apply_move(state, action)
        move_count += 1
    
    return ludo_cpp.get_winner(state)


def evaluate_model_vs_greedy(model, num_games=100, mcts_simulations=50):
    """
    Evaluate model against 3 Greedy Bots.
    
    Args:
        model: Neural network model.
        num_games: Number of games to play.
        mcts_simulations: MCTS simulations per move.
        
    Returns:
        Win rate of the model (as player 0).
    """
    greedy = GreedyBot()
    wins = 0
    
    def model_player(state):
        mcts = MCTS(model, num_simulations=mcts_simulations)
        probs = get_action_probs(mcts, state, temperature=0)
        return int(np.argmax(probs))
    
    def greedy_player(state):
        return greedy.select_action(state)
    
    for game_idx in range(num_games):
        winner = play_match(model_player, greedy_player, greedy_player, greedy_player)
        if winner == 0:
            wins += 1
        if (game_idx + 1) % 10 == 0:
            print(f"Eval game {game_idx + 1}/{num_games}: Model wins = {wins}")
    
    win_rate = wins / num_games
    return win_rate


def evaluate_model_vs_model(model_new, model_old, num_games=100, mcts_simulations=50):
    """
    Evaluate new model against old model.
    
    Args:
        model_new: New neural network model.
        model_old: Previous neural network model.
        num_games: Number of games to play.
        mcts_simulations: MCTS simulations per move.
        
    Returns:
        Win rate of the new model.
    """
    def make_player(model):
        def player_fn(state):
            mcts = MCTS(model, num_simulations=mcts_simulations)
            probs = get_action_probs(mcts, state, temperature=0)
            return int(np.argmax(probs))
        return player_fn
    
    new_player = make_player(model_new)
    old_player = make_player(model_old)
    
    wins_new = 0
    
    for game_idx in range(num_games):
        # Rotate positions each game for fairness
        if game_idx % 4 == 0:
            players = [new_player, old_player, old_player, old_player]
            new_pos = 0
        elif game_idx % 4 == 1:
            players = [old_player, new_player, old_player, old_player]
            new_pos = 1
        elif game_idx % 4 == 2:
            players = [old_player, old_player, new_player, old_player]
            new_pos = 2
        else:
            players = [old_player, old_player, old_player, new_player]
            new_pos = 3
        
        winner = play_match(*players)
        if winner == new_pos:
            wins_new += 1
    
    win_rate = wins_new / num_games
    return win_rate
