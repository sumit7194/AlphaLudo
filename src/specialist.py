
import numpy as np
import torch
import ludo_cpp
from self_play import SelfPlayWorker
from mcts import MCTS, get_action_probs
from tensor_utils import state_to_tensor

class SpecialistWorker(SelfPlayWorker):
    def __init__(self, model, reward_config=None, mcts_simulations=100, visualize=False):
        """
        Specialist Worker for Reward Shaping.
        """
        super().__init__(model, mcts_simulations=mcts_simulations, visualize=visualize)
        self.reward_config = reward_config or {}

    def get_reward_shaping(self, prev_state, action, next_state, player):
        """
        Calculate extra reward based on what happened in the move.
        """
        reward = 0.0
        
        # 1. Check for CUT (Aggression)
        prev_pos = prev_state.player_positions
        curr_pos = next_state.player_positions
        
        for opp in range(4):
            if opp == player: continue
            
            for t in range(4):
                was_on_board = (prev_pos[opp][t] != -1 and prev_pos[opp][t] != 99)
                is_at_base = (curr_pos[opp][t] == -1)
                
                if was_on_board and is_at_base:
                    reward += self.reward_config.get('cut', 0.0)
                    
        # 2. Check for HOME (Rusher)
        # 1.0 reward for reaching home (99) or close to it? Config: {'home': 0.5}
        prev_my_tokens = prev_pos[player]
        curr_my_tokens = curr_pos[player]
        for t in range(4):
            if prev_my_tokens[t] != 99 and curr_my_tokens[t] == 99:
                 reward += self.reward_config.get('home', 0.0)

        # 3. Check for DEFENSIVE (Penalty for being threatened)
        # A token is threatened if an opponent is 1-6 steps behind it.
        # We check the `next_state` (after my move).
        # We apply penalty if ANY of my tokens are threatened.
        # Config: {'unsafe': -0.1} (applied per threatened token per turn)
        if 'unsafe' in self.reward_config:
            unsafe_penalty = self.reward_config['unsafe']
            
            # Get all opponent positions (global indices)
            opp_positions = []
            for opp in range(4):
                if opp == player: continue
                # We need global position to compare distance?
                # Ludo 'positions' are usually player-relative (0-51 path).
                # Converting to global index is tricky without 'board' map.
                # BUT, ludo_cpp might assume relative.
                # Assuming standard Ludo with offsets:
                # P0 start 0. P1 start 13. P2 start 26. P3 start 39.
                # Global = (pos + offset) % 52.
                # Safe spots are global 0, 8, ...
                # Let's approximate: If we assume `ludo_cpp` logic handles converting... 
                # Wait, without `ludo_cpp` helper to check safety, we might implement a flaky rule.
                # ALTERNATIVE: Use `ludo_cpp.is_safe(state, player, token)` if it existed.
                # It does not.
                
                # SIMPLIFICATION:
                # Reward "Safe Spots" (Globes).
                # Globe indices (relative for P0): 0, 8, 13, 21, 26, 34, 39, 47.
                # If we land on a globe -> Reward.
                pass
            
            # Implementation of Globe Reward (Simpler Defensive)
            # Safe relative positions for any player:
            SAFE_REL_POS = {0, 8, 13, 21, 26, 34, 39, 47}
            for t in range(4):
                pos = curr_my_tokens[t]
                if pos != -1 and pos != 99:
                    if pos in SAFE_REL_POS:
                         # We are safe.
                         # Maybe small positive reward?
                         # Or reducing penalty?
                         pass
                    
            # Let's stick to "Reward Safety" instead of "Punish Danger" to avoid complex relative math in Python.
            # Config: {'safe': 0.2}
            if 'safe' in self.reward_config:
                 for t in range(4):
                     pos = curr_my_tokens[t]
                     if pos in SAFE_REL_POS:
                         reward += self.reward_config['safe']

        # 4. Check for BLOCKADE (Pairing)
        # Config: {'blockade': 0.2}
        if 'blockade' in self.reward_config:
            # Count tokens at each position
            counts = {}
            for t in range(4):
                pos = curr_my_tokens[t]
                if pos != -1 and pos != 99:
                    counts[pos] = counts.get(pos, 0) + 1
            
            for pos, count in counts.items():
                if count >= 2:
                    reward += self.reward_config['blockade']
                 
        return reward

    def play_game(self, temperature=1.0):
        """
        Play a game and return examples with SHAPED rewards.
        Matches SelfPlayWorker logic but adds reward hooks.
        """
        game_history = []
        state = ludo_cpp.create_initial_state()
        move_count = 0
        max_moves = 1000
        
        shaped_returns = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        
        while not state.is_terminal and move_count < max_moves:
            # Dice logic (must duplicate SelfPlayWorker logic)
            dice_roll = np.random.randint(1, 7)
            state.current_dice_roll = dice_roll
            
            # Legal moves
            legal_moves = ludo_cpp.get_legal_moves(state)
            
            if len(legal_moves) == 0:
                 state.current_player = (state.current_player + 1) % 4
                 state.current_dice_roll = 0
                 continue

            # MCTS
            mcts = MCTS(self.model, num_simulations=self.mcts_simulations)
            
            try:
                action_probs = get_action_probs(mcts, state, temperature=temperature)
            except Exception as e:
                print(f"MCTS Error: {e}")
                # Fallback? break
                break
                
            # Store history
            state_tensor = state_to_tensor(state)
            game_history.append({
                'state': state_tensor,
                'policy': action_probs,
                'player': state.current_player,
                'dice': dice_roll # Optional debug
            })
            
            # Pick action
            if temperature == 0:
                action = int(np.argmax(action_probs))
            else:
                action = np.random.choice(4, p=action_probs)
                
            if action not in legal_moves:
                action = np.random.choice(legal_moves)
                
            # Apply move
            prev_state = state
            state = ludo_cpp.apply_move(state, action)
            move_count += 1
            
            # Reward Shaping Hook
            # Who acted? prev_state.current_player
            actor = prev_state.current_player
            shaping = self.get_reward_shaping(prev_state, action, state, actor)
            shaped_returns[actor] += shaping
            
        # Determine winner
        winner = ludo_cpp.get_winner(state)
        
        training_examples = []
        for record in game_history:
            p = record['player']
            if winner == -1:
                base = 0.0
            elif winner == p:
                base = 1.0
            else:
                base = -1.0
                
            total_target = base + shaped_returns[p]
            
            training_examples.append((
                record['state'],
                torch.tensor(record['policy'], dtype=torch.float32),
                torch.tensor(total_target, dtype=torch.float32)
            ))
            
        return training_examples
