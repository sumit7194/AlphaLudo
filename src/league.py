import numpy as np
import torch
import ludo_cpp
from self_play import SelfPlayWorker
from mcts import MCTS, get_action_probs
from tensor_utils import state_to_tensor
try:
    from src.visualizer import visualizer
except ImportError:
    visualizer = None

class LeagueWorker(SelfPlayWorker):
    def __init__(self, main_model, specialist_pool, probabilities, mcts_simulations=100, visualize=False):
        """
        League Worker for diverse training.
        
        Args:
            main_model: The model being trained (Baseline).
            specialist_pool: Dict of {name: model}.
            probabilities: Dict of {name: prob} to select opponents.
                           Must include 'Main' (Baseline).
        """
        super().__init__(main_model, mcts_simulations=mcts_simulations, visualize=visualize)
        self.specialist_pool = specialist_pool
        self.probabilities = probabilities
        
        # Ensure Main is in pool for lookup
        self.full_pool = specialist_pool.copy()
        self.full_pool['Main'] = main_model
        
        print(f"LeagueWorker initialized. Visualizer enabled: {self.visualize}. Visualizer instance: {visualizer}")

    def select_opponents(self):
        """
        Select 4 players based on probabilities.
        P0 is ALWAYS Main.
        P1, P2, P3 are rolled independently.
        """
        players = ['Main']
        options = list(self.probabilities.keys())
        probs = list(self.probabilities.values())
        
        # Select P1, P2, P3
        opponents = np.random.choice(options, size=3, p=probs)
        players.extend(opponents)
        return players

    def play_game(self, temperature=1.0):
        """
        Play a league game.
        """
        # 1. Setup Matchup
        identities = self.select_opponents()
        identities = [str(x) for x in identities]
        models = [self.full_pool[id] for id in identities]
        
        # Broadcast Identity
        if self.visualize and visualizer:
            visualizer.broadcast_identities(identities)

        game_history = []
        state = ludo_cpp.create_initial_state()
        move_count = 0
        max_moves = 1000
        
        while not state.is_terminal and move_count < max_moves:
            dice_roll = np.random.randint(1, 7)
            state.current_dice_roll = dice_roll
            
            # Broadcast state (showing identities would be nice here)
            if self.visualize and visualizer:
                visualizer.broadcast_state(state) # Standard broadcast
            
            legal_moves = ludo_cpp.get_legal_moves(state)
            if len(legal_moves) == 0:
                state.current_player = (state.current_player + 1) % 4
                state.current_dice_roll = 0
                continue
            
            # Select Model for Current Player
            current_player = state.current_player
            current_id = identities[current_player]
            current_model = models[current_player]
            
            # MCTS
            mcts = MCTS(current_model, num_simulations=self.mcts_simulations)
            action_probs = get_action_probs(mcts, state, temperature=temperature)
            
            # Record Data logic:
            # We ONLY learn from 'Main' players?
            # Or if P2 is Main, we learn from P2.
            # If P2 is Aggressive, do we learn from it?
            # User said: "we will use there moves only to do further training" (referring to baseline instances).
            # So: Record if current_id == 'Main'.
            
            if current_id == 'Main':
                state_tensor = state_to_tensor(state)
                game_history.append({
                    'state': state_tensor,
                    'policy': action_probs,
                    'player': current_player
                })
            
            # Select Action
            if temperature == 0:
                action = int(np.argmax(action_probs))
            else:
                action = np.random.choice(4, p=action_probs)
                
            if action not in legal_moves:
                action = np.random.choice(legal_moves)
            
            if self.visualize and visualizer:
                 visualizer.broadcast_move(state.current_player, action, dice_roll)

            state = ludo_cpp.apply_move(state, action)
            move_count += 1
            
        # End Game
        if self.visualize and visualizer:
            visualizer.broadcast_state(state)
            
        winner = ludo_cpp.get_winner(state)
        
        # Assign Rewards
        training_examples = []
        for record in game_history:
            p = record['player']
            # Only records for 'Main' players exist in history
            
            if winner == -1:
                value_target = 0.0
            elif winner == p:
                value_target = 1.0
            else:
                value_target = -1.0
                
            training_examples.append((
                record['state'],
                torch.tensor(record['policy'], dtype=torch.float32),
                torch.tensor(value_target, dtype=torch.float32)
            ))
            
        return training_examples, winner, identities
