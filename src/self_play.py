"""
Self-Play module for AlphaLudo.

Generates training data by playing games against itself using MCTS + Neural Network.
"""

import numpy as np
import torch
import ludo_cpp
from mcts import MCTS, get_action_probs
from tensor_utils import state_to_tensor

# Optional visualization
try:
    from visualizer import visualizer
    HAS_VISUALIZER = True
except ImportError:
    HAS_VISUALIZER = False


class SelfPlayWorker:
    """
    Plays games against itself to generate training data.
    """
    
    def __init__(self, model, mcts_simulations=100, temperature_threshold=30, visualize=False):
        """
        Args:
            model: Neural network model.
            mcts_simulations: Number of MCTS simulations per move.
            temperature_threshold: Use temperature=1.0 for first N moves, then 0.
            visualize: If True, broadcast states to visualizer.
        """
        self.model = model
        self.mcts_simulations = mcts_simulations
        self.temperature_threshold = temperature_threshold
        self.visualize = visualize and HAS_VISUALIZER

    def play_game(self):
        """
        Play one complete game.
        
        Returns:
            List of (state_tensor, policy_vector, current_player) tuples.
            The value_target will be filled in after we know the winner.
        """
        game_history = []
        state = ludo_cpp.create_initial_state()
        move_count = 0
        max_moves = 1000  # Prevent infinite games
        
        while not state.is_terminal and move_count < max_moves:
            # Roll dice
            dice_roll = np.random.randint(1, 7)
            state.current_dice_roll = dice_roll
            
            # Broadcast state for visualization
            if self.visualize:
                visualizer.broadcast_state(state)
            
            # Get legal moves
            legal_moves = ludo_cpp.get_legal_moves(state)
            
            if len(legal_moves) == 0:
                # No legal moves, pass turn
                state.current_player = (state.current_player + 1) % 4
                state.current_dice_roll = 0
                continue
            
            # Create MCTS and search
            mcts = MCTS(self.model, num_simulations=self.mcts_simulations)
            
            # Determine temperature
            temperature = 1.0 if move_count < self.temperature_threshold else 0.0
            
            # Get action probabilities from MCTS
            action_probs = get_action_probs(mcts, state, temperature=temperature)
            
            # Record state and policy
            state_tensor = state_to_tensor(state)
            game_history.append({
                'state': state_tensor,
                'policy': action_probs,
                'player': state.current_player
            })
            
            # Select action
            if temperature == 0:
                # Deterministic: pick best (should already be handled by get_action_probs)
                action = int(np.argmax(action_probs))
            else:
                # Stochastic: sample from distribution
                action = np.random.choice(4, p=action_probs)
            
            # Ensure action is legal
            if action not in legal_moves:
                # Fallback to random legal move
                action = np.random.choice(legal_moves)
            
            # Broadcast move for visualization (with logic to update dice stats)
            if self.visualize:
                visualizer.broadcast_move(state.current_player, action, dice_roll)
            
            # Apply move
            state = ludo_cpp.apply_move(state, action)
            move_count += 1
            
        # Broadcast final state (so visualizer sees the winner/score=4)
        if self.visualize:
            visualizer.broadcast_state(state)
        
        # Fill in value target based on winner
        
        # Determine winner
        winner = ludo_cpp.get_winner(state)
        
        # Assign value targets based on winner
        training_examples = []
        for record in game_history:
            if winner == -1:
                # Draw (shouldn't happen in Ludo, but handle gracefully)
                value_target = 0.0
            elif winner == record['player']:
                value_target = 1.0
            else:
                value_target = -1.0
            
            training_examples.append((
                record['state'],
                torch.tensor(record['policy'], dtype=torch.float32),
                torch.tensor([value_target], dtype=torch.float32)
            ))
        
        return training_examples


def generate_self_play_data(model, num_games=10, mcts_simulations=50):
    """
    Generate training data from multiple self-play games.
    
    Args:
        model: Neural network model.
        num_games: Number of games to play.
        mcts_simulations: MCTS simulations per move.
        
    Returns:
        List of (state, policy, value) training examples.
    """
    worker = SelfPlayWorker(model, mcts_simulations=mcts_simulations)
    all_examples = []
    
    for game_idx in range(num_games):
        examples = worker.play_game()
        all_examples.extend(examples)
        print(f"Game {game_idx + 1}/{num_games}: Generated {len(examples)} examples")
    
    return all_examples
