"""
Stochastic MCTS for Ludo.

Key Design:
- Decision Nodes: Use PUCT to select which token to move.
- Chance Nodes: Fully expand all 6 dice outcomes. Value = average of children.
- Neural Network provides Policy (which token to move) and Value (win probability).
"""

import math
import copy
import numpy as np
import torch
import ludo_cpp
from tensor_utils import state_to_tensor

# Hyperparameters
# CPUCT now passed in MCTS.__init__

class MCTSNode:
    """
    Represents a node in the MCTS tree.
    Can be either a Decision Node (player picks a move) or a Chance Node (dice roll).
    """
    def __init__(self, state, parent=None, prior=0.0, action=None, dice_roll=None):
        self.state = state
        self.parent = parent
        self.prior = prior  # P(s, a) from NN (only for Decision Node children)
        self.action = action  # Token index that led here (from Decision Node)
        self.dice_roll = dice_roll  # Dice value that led here (from Chance Node)
        
        self.children = {}  # {action: MCTSNode} or {roll: MCTSNode}
        self.visit_count = 0
        self.value_sum = 0.0
        
        # Determine node type
        # Chance Node: Dice not yet rolled (current_dice_roll == 0) and game not over
        self.is_chance = (state.current_dice_roll == 0) and (not state.is_terminal)
        self.is_terminal = state.is_terminal

    def is_leaf(self):
        """A node is a leaf if it has no children (not yet expanded)."""
        return len(self.children) == 0

    def get_mean_value(self):
        """Returns Q(s, a) = W / N."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def copy_state(state):
    """Create a deep copy of a GameState."""
    new_state = ludo_cpp.GameState()
    new_state.current_player = state.current_player
    new_state.current_dice_roll = state.current_dice_roll
    new_state.is_terminal = state.is_terminal
    # Copy numpy arrays
    new_state.player_positions[:] = state.player_positions[:]
    new_state.scores[:] = state.scores[:]
    new_state.board[:] = state.board[:]
    return new_state


class MCTS:
    """
    Monte Carlo Tree Search with Stochastic (Chance) Nodes.
    
    Features:
    - Full Expansion: Chance Nodes expand all 6 dice outcomes.
    - PUCT Selection: Decision Nodes use UCB with neural network priors.
    - Value Head: Neural network evaluates leaf nodes (no random rollouts).
    """
    
    def __init__(self, model, num_simulations=100, device=None, cpuct=1.0):
        self.model = model
        self.num_simulations = num_simulations
        self.root_player = None  # Track whose perspective we're evaluating from
        # Determine device from model
        self.device = device or next(model.parameters()).device
        self.cpuct = cpuct

    def search(self, root_state):
        """
        Run MCTS simulations and return visit counts for each action.
        
        Args:
            root_state: GameState with dice already rolled (Decision Node).
            
        Returns:
            dict: {action: visit_count} for each legal action.
        """
        assert root_state.current_dice_roll > 0, "Root must have dice rolled (Decision Node)"
        
        self.root_player = root_state.current_player
        root = MCTSNode(root_state)

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # ===== SELECTION =====
            while not node.is_leaf() and not node.is_terminal:
                if node.is_chance:
                    # Chance Node: Randomly sample a dice roll for traversal
                    # (All children already exist due to full expansion)
                    roll = np.random.randint(1, 7)
                    node = node.children[roll]
                else:
                    # Decision Node: Use PUCT to select best action
                    node = self._select_child(node)
                search_path.append(node)
            
            # ===== EXPANSION & EVALUATION =====
            value = self._expand_and_evaluate(node)
            
            # ===== BACKPROPAGATION =====
            self._backpropagate(search_path, value)

        # Return visit counts for root's children (the legal moves)
        return {action: child.visit_count for action, child in root.children.items()}

    def _select_child(self, node):
        """
        Select the best child of a Decision Node using PUCT formula.
        
        PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
        """
        best_score = -float('inf')
        best_child = None
        
        total_visits = node.visit_count
        
        for action, child in node.children.items():
            q_value = child.get_mean_value()
            
            # Flip value if child is opponent's turn
            # (Value head returns probability of current_player winning)
            if child.state.current_player != self.root_player:
                q_value = -q_value
            
            # Exploration bonus
            u_value = self.cpuct * child.prior * math.sqrt(total_visits) / (1 + child.visit_count)
            
            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child

    def _expand_and_evaluate(self, node):
        """
        Expand a leaf node and return its value.
        
        - Decision Node: Use NN to get policy and value. Create children for each legal move.
        - Chance Node: Create all 6 dice children. Return average of their values.
        - Terminal Node: Return +1 (win), -1 (loss), or 0 (draw).
        """
        if node.is_terminal:
            # Game over - return result from root player's perspective
            winner = ludo_cpp.get_winner(node.state)
            if winner == self.root_player:
                return 1.0
            elif winner != -1:
                return -1.0
            else:
                return 0.0
        
        if node.is_chance:
            # ===== CHANCE NODE: Full Expansion =====
            # Create all 6 children (one for each dice outcome)
            values = []
            for roll in range(1, 7):
                if roll not in node.children:
                    # Create new state with this dice roll
                    new_state = copy_state(node.state)
                    new_state.current_dice_roll = roll
                    child = MCTSNode(new_state, parent=node, dice_roll=roll)
                    node.children[roll] = child
                    
                    # Evaluate this child with NN
                    child_value = self._evaluate_state(child.state)
                    values.append(child_value)
                else:
                    # Child already exists, use its current value
                    values.append(node.children[roll].get_mean_value())
            
            # Return expected value (average across all dice outcomes)
            return sum(values) / 6.0
        
        else:
            # ===== DECISION NODE: Expand with NN Policy =====
            # Get NN predictions
            tensor = state_to_tensor(node.state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                policy_logits, value = self.model(tensor)
            
            value = value.item()
            
            # Get legal moves
            legal_moves = ludo_cpp.get_legal_moves(node.state)
            
            if len(legal_moves) == 0:
                # No legal moves - pass turn (shouldn't happen often)
                return value
            
            # Mask and normalize policy
            policy_probs = torch.softmax(policy_logits, dim=1).detach().cpu().numpy()[0]
            
            # Extract probabilities for legal moves only
            legal_probs = [policy_probs[m] for m in legal_moves]
            prob_sum = sum(legal_probs)
            
            if prob_sum > 0:
                legal_probs = [p / prob_sum for p in legal_probs]
            else:
                legal_probs = [1.0 / len(legal_moves)] * len(legal_moves)
            
            # Create children for each legal move
            for i, move in enumerate(legal_moves):
                next_state = ludo_cpp.apply_move(node.state, move)
                child = MCTSNode(next_state, parent=node, prior=legal_probs[i], action=move)
                node.children[move] = child
            
            # Adjust value to root player's perspective
            if node.state.current_player != self.root_player:
                value = -value
            
            return value

    def _evaluate_state(self, state):
        """Evaluate a state using the neural network's value head."""
        tensor = state_to_tensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, value = self.model(tensor)
        
        v = value.item()
        
        # Adjust to root player's perspective
        if state.current_player != self.root_player:
            v = -v
        
        return v

    def _backpropagate(self, search_path, value):
        """
        Update statistics for all nodes in the search path.
        
        For proper perspective handling:
        - Value is from root player's perspective.
        - We flip the sign when crossing player boundaries.
        """
        for node in reversed(search_path):
            node.visit_count += 1
            
            # Store value from the perspective of the node's player
            if node.state.current_player == self.root_player:
                node.value_sum += value
            else:
                node.value_sum += -value


def get_action_probs(mcts, state, temperature=1.0):
    """
    Run MCTS and return action probabilities.
    
    Args:
        mcts: MCTS instance.
        state: GameState with dice rolled.
        temperature: Controls exploration (higher = more random).
        
    Returns:
        list: Probabilities for each token index [0, 1, 2, 3].
    """
    visit_counts = mcts.search(state)
    
    if temperature == 0:
        # Deterministic: pick most visited
        best_action = max(visit_counts, key=visit_counts.get)
        probs = [0.0, 0.0, 0.0, 0.0]
        probs[best_action] = 1.0
        return probs
    
    # Apply temperature
    visits = np.array([visit_counts.get(i, 0) for i in range(4)])
    
    if visits.sum() == 0:
        return [0.25, 0.25, 0.25, 0.25]  # Uniform if no visits
    
    visits = visits ** (1.0 / temperature)
    probs = visits / visits.sum()
    
    return probs.tolist()
