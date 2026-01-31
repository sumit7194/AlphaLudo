
"""
Stochastic MCTS for Ludo - Mastery Version.

Updates:
- Uses `tensor_utils_mastery` for 12-channel input.
- Handles Spatial Policy Head (15x15 logits) -> Maps to Token Moves (0-3).
"""

import math
import copy
import numpy as np
import torch
import ludo_cpp
from tensor_utils_mastery import state_to_tensor_mastery, get_board_coords, BOARD_SIZE

# Hyperparameters
CPUCT = 1.0  # Exploration constant

class MCTSNode:
    """
    Represents a node in the MCTS tree.
    """
    def __init__(self, state, parent=None, prior=0.0, action=None, dice_roll=None):
        self.state = state
        self.parent = parent
        self.prior = prior  # P(s, a) from NN
        self.action = action  # Token index (0-3)
        self.dice_roll = dice_roll 
        
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        
        self.is_chance = (state.current_dice_roll == 0) and (not state.is_terminal)
        self.is_terminal = state.is_terminal

    def is_leaf(self):
        return len(self.children) == 0

    def get_mean_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def copy_state(state):
    """Create a deep copy of a GameState."""
    new_state = ludo_cpp.GameState()
    new_state.current_player = state.current_player
    new_state.current_dice_roll = state.current_dice_roll
    new_state.is_terminal = state.is_terminal
    new_state.player_positions[:] = state.player_positions[:]
    new_state.scores[:] = state.scores[:]
    new_state.board[:] = state.board[:]
    return new_state


class MCTSMastery:
    """
    MCTS adapted for Mastery Architecture (Spatial Policy).
    """
    
    def __init__(self, model, num_simulations=100, device=None):
        self.model = model
        self.num_simulations = num_simulations
        self.root_player = None 
        self.device = device or next(model.parameters()).device

    def get_search_stats(self, root_state):
        """
        Run MCTS and return rich statistics for visualization.
        """
        assert root_state.current_dice_roll > 0
        self.root_player = root_state.current_player
        root = MCTSNode(root_state)
        
        # --- Run Simulations ---
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # Selection
            while not node.is_leaf() and not node.is_terminal:
                if node.is_chance:
                    roll = np.random.randint(1, 7)
                    if roll in node.children:
                        node = node.children[roll]
                    else:
                        break # Standard fallback
                else:
                    node = self._select_child(node)
                search_path.append(node)
            
            # Expand
            value = self._expand_and_evaluate(node)
            
            # Backprop
            self._backpropagate(search_path, value)

        # --- Extract Stats ---
        stats = {
            'root_player': self.root_player,
            'children': [],
            'pv': []
        }
        
        # 1. Child Stats (Root level)
        total_visits = root.visit_count
        positions = root.state.player_positions[self.root_player] # Get current positions
        
        # 1. Child Stats (Root level)
        total_visits = root.visit_count
        positions = root.state.player_positions[self.root_player]
        
        for action, child in root.children.items():
            child_visits = child.visit_count
            child_q = child.get_mean_value()
            if child.state.current_player != self.root_player:
                child_q = -child_q 
            
            # Identify source position
            pos_from = positions[action]
            
            # --- Extract PV for this specific move ---
            # We trace this child's best path
            pv_path = []
            curr = child
            # Record the initial move (Root -> Child)
            # Wait, the child node represents the state AFTER the move.
            # So the first step of the path is (RootPos -> ChildPos).
            # We should include this so the path starts from the board.
            
            # Step 0: The candidate move itself
            # But the 'child' state has new positions.
            # pos_from = positions[action] (Root state)
            # pos_to = child.state...
            
            # Actually, let's trace FUTURE steps from this child.
            # The UI already draws the arrow for the candidate move itself (Root->Child).
            # So we only need the *continuation* (PV vs Opponent).
            
            for _ in range(5): # Depth 5 continuation
                if curr.is_leaf() or curr.is_terminal:
                    break
                
                # Pick best sub-child
                best_sub_action = -1
                best_sub_child = None
                max_sub_v = -1
                
                if curr.is_chance:
                    if len(curr.children) > 0:
                        # Chance Node: For visualization, we must pick ONE outcome to show a path.
                        # We pick the first available roll outcome to preserve continuity.
                        first_roll = list(curr.children.keys())[0]
                        curr = curr.children[first_roll]
                        continue
                    else:
                        break
                else:
                    for a, c in curr.children.items():
                        if c.visit_count > max_sub_v:
                            max_sub_v = c.visit_count
                            
                            best_sub_child = c
                            best_sub_action = a
                    
                    if best_sub_child:
                        p = curr.state.current_player
                        p_pos_from = curr.state.player_positions[p][best_sub_action]
                        p_pos_to = best_sub_child.state.player_positions[p][best_sub_action]
                        
                        pv_path.append({
                            'player': p,
                            'token': best_sub_action,
                            'pos_from': int(p_pos_from),
                            'pos_to': int(p_pos_to)
                        })
                        curr = best_sub_child
                    else:
                         break
            
            stats['children'].append({
                'action': action,
                'pos_from': int(pos_from),
                'visits': child_visits,
                'q_value': child_q,
                'prior': child.prior,
                'visit_pct': child_visits / total_visits if total_visits > 0 else 0,
                'pv': pv_path # Attach PV to the specific option
            })
            
        return stats

    # ... (keep helper methods same) ...

class MCTSVisualizer(MCTSMastery):
    """
    Subclass that extends functionality for visualization.
    Inherits core logic from MCTSMastery but adds get_search_stats.
    """
    pass

    def _select_child(self, node):
        best_score = -float('inf')
        best_child = None
        total_visits = node.visit_count
        
        for action, child in node.children.items():
            q_value = child.get_mean_value()
            if child.state.current_player != self.root_player:
                q_value = -q_value
            
            u_value = CPUCT * child.prior * math.sqrt(total_visits) / (1 + child.visit_count)
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child

    def _expand_and_evaluate(self, node):
        if node.is_terminal:
            winner = ludo_cpp.get_winner(node.state)
            if winner == self.root_player:
                return 1.0
            elif winner != -1:
                return -1.0
            else:
                return 0.0
        
        if node.is_chance:
            values = []
            for roll in range(1, 7):
                if roll not in node.children:
                    new_state = copy_state(node.state)
                    new_state.current_dice_roll = roll
                    child = MCTSNode(new_state, parent=node, dice_roll=roll)
                    node.children[roll] = child
                    child_value = self._evaluate_state(child.state)
                    values.append(child_value)
                else:
                    values.append(node.children[roll].get_mean_value())
            return sum(values) / 6.0
        
        else:
            # ===== DECISION NODE (Modified for Spatial Policy) =====
            tensor = state_to_tensor_mastery(node.state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                spatial_logits, value = self.model(tensor)
            
            value = value.item()
            legal_moves = ludo_cpp.get_legal_moves(node.state) # List of token indices 0-3
            
            if len(legal_moves) == 0:
                return value

            # --- Spatial Masking Logic ---
            # 1. We have spatial_logits (1, 225)
            # 2. We identify the square index for each legal token move
            
            logits_for_tokens = []
            current_p = node.state.current_player
            positions = node.state.player_positions[current_p]
            
            for token_idx in legal_moves:
                pos = positions[token_idx]
                r, c = get_board_coords(current_p, pos, token_idx)
                # Flatten index (0-224)
                flat_idx = r * BOARD_SIZE + c
                
                # Extract logit
                logit = spatial_logits[0, flat_idx].item()
                logits_for_tokens.append(logit)
            
            # 3. Softmax locally over the legal token moves
            logits_tensor = torch.tensor(logits_for_tokens, dtype=torch.float32)
            probs_tensor = torch.softmax(logits_tensor, dim=0)
            token_probs = probs_tensor.numpy().tolist()
            
            # Create children
            for i, move in enumerate(legal_moves):
                next_state = ludo_cpp.apply_move(node.state, move)
                child = MCTSNode(next_state, parent=node, prior=token_probs[i], action=move)
                node.children[move] = child
            
            if node.state.current_player != self.root_player:
                value = -value
            
            return value

    def _evaluate_state(self, state):
        tensor = state_to_tensor_mastery(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, value = self.model(tensor)
        
        v = value.item()
        if state.current_player != self.root_player:
            v = -v
        return v

    def _backpropagate(self, search_path, value):
        for node in reversed(search_path):
            node.visit_count += 1
            if node.state.current_player == self.root_player:
                node.value_sum += value
            else:
                node.value_sum += -value

def get_action_probs_mastery(mcts, state, temperature=1.0):
    visit_counts = mcts.search(state)
    
    if temperature == 0:
        best_action = max(visit_counts, key=visit_counts.get)
        probs = [0.0] * 4
        probs[best_action] = 1.0
        return probs
    
    visits = np.array([visit_counts.get(i, 0) for i in range(4)])
    if visits.sum() == 0:
        return [0.25] * 4
    
    visits = visits ** (1.0 / temperature)
    probs = visits / visits.sum()
    return probs.tolist()
