
import math
import copy
import numpy as np
import torch
import ludo_cpp
from src.tensor_utils_mastery import state_to_tensor_mastery, get_board_coords, BOARD_SIZE

# Hyperparameters
CPUCT = 1.0

class MCTSNode:
    """Same node structure as MCTS Mastery, repeated here for dependency clarity."""
    def __init__(self, state, parent=None, prior=0.0, action=None, dice_roll=None):
        self.state = state
        self.parent = parent
        self.prior = prior
        self.action = action
        self.dice_roll = dice_roll 
        
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        
        # Determine if chance node
        # In this game, chance happens at start of turn (dice roll).
        # But our state usually *has* a dice roll already provided by environment loop.
        # Wait, inside MCTS we need to simulate chance?
        # Standard Ludo MCTS:
        # Root has Dice Roll X.
        # Child (Post-Action) has Dice Roll 0 (waiting).
        # So next node is Chance Node.
        self.is_chance = (state.current_dice_roll == 0) and (not state.is_terminal)
        self.is_terminal = state.is_terminal

    def is_leaf(self):
        return len(self.children) == 0

    def get_mean_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

def copy_state(state):
    """Deep copy."""
    new_state = ludo_cpp.GameState()
    new_state.current_player = state.current_player
    new_state.current_dice_roll = state.current_dice_roll
    new_state.is_terminal = state.is_terminal
    new_state.player_positions[:] = state.player_positions[:]
    new_state.scores[:] = state.scores[:]
    new_state.board[:] = state.board[:]
    return new_state

class VectorMCTSMastery:
    """
    Batched MCTS for multiple parallel games.
    """
    def __init__(self, model, num_simulations=50, device=None):
        self.model = model
        self.num_simulations = num_simulations
        self.device = device or next(model.parameters()).device
        
    def search_batch(self, roots):
        """
        Run MCTS for a batch of root nodes.
        roots: List of MCTSNode objects (one per game).
        """
        batch_size = len(roots)
        root_players = [r.state.current_player for r in roots] # Player whose turn it is at root
        
        # Pre-allocate random rolls for chance selection efficiency? 
        # Or just call randint inside loop (cheap).

        for _ in range(self.num_simulations):
            # --- 1. SELECTION ---
            leaf_nodes = []
            search_paths = []
            expanded_rolls = [] # For chance nodes: which roll did we expand?
            
            for i in range(batch_size):
                node = roots[i]
                path = [node]
                
                # Traverse tree
                while not node.is_leaf() and not node.is_terminal:
                    if node.is_chance:
                        # Chance Node: Pick a random child
                        # If child exists, descend.
                        # If child missing, we treat this as "Leaf to Expand"
                        roll = np.random.randint(1, 7)
                        if roll in node.children:
                            node = node.children[roll]
                            path.append(node)
                        else:
                            # Child missing -> Stop here. 
                            # We will expand this specific outcome (roll)
                            # We store 'roll' to know which child to create.
                            # We treat 'node' (the Chance Node) as the leaf representative.
                            expanded_rolls.append(roll) 
                            break 
                    else:
                        # Decision Node: Select best child (PUCT)
                        node = self._select_child(node, root_players[i])
                        path.append(node)
                
                # Check consistency if we broke out due to chance missing
                if len(expanded_rolls) <= i: 
                    # If we didn't add to expanded_rolls...
                    if node.is_chance:
                        # We ended at a Chance Node Leaf (e.g. Root)
                        # We must pick a roll to expand
                        roll = np.random.randint(1, 7)
                        expanded_rolls.append(roll)
                    else:
                        # Regular leaf decision node or terminal
                        expanded_rolls.append(None)
                
                leaf_nodes.append(node)
                search_paths.append(path)
            
            # --- 2. EXPANSION & EVALUATION (BATCHED) ---
            
            states_to_eval = []
            indices_to_eval = [] # (original_batch_index, type='decision'|'chance', roll)
            
            # Identify what needs evaluation
            for i, node in enumerate(leaf_nodes):
                if node.is_terminal:
                    continue # Will handle value in step 3
                
                if node.is_chance:
                    # We need to expand a specific Chance Child
                    roll = expanded_rolls[i]
                    # Create new state for evaluation
                    new_state = copy_state(node.state)
                    new_state.current_dice_roll = roll
                    # This new state is what the NN sees
                    state_tensor = state_to_tensor_mastery(new_state)
                    states_to_eval.append(state_tensor)
                    indices_to_eval.append((i, 'chance', roll))
                else:
                    # Regular Expansion
                    # Evaluate the node.state directly
                    state_tensor = state_to_tensor_mastery(node.state)
                    states_to_eval.append(state_tensor)
                    indices_to_eval.append((i, 'decision', None))
            
            values_map = {} # batch_idx -> value (from root player perspective)
            
            if states_to_eval:
                state_batch = torch.stack(states_to_eval).to(self.device)
                with torch.no_grad():
                    spatial_logits, predicted_values = self.model(state_batch)
                
                # Process Neural Net Outputs
                for j, (orig_idx, mtype, roll) in enumerate(indices_to_eval):
                    node = leaf_nodes[orig_idx]
                    val = predicted_values[j].item()
                    
                    if mtype == 'chance':
                         # Create the Chance Child
                        new_state = copy_state(node.state)
                        new_state.current_dice_roll = roll
                        # Create child node
                        # Chance child is a Decision Node (usually, unless terminal)
                        child = MCTSNode(new_state, parent=node, dice_roll=roll)
                        node.children[roll] = child
                        
                        # We use the NN Value directly for backprop
                        # NN evaluates state relative to current_player of new_state
                        # But wait, chance node parent logic?
                        # If Chance Node Player is P0.
                        # Child state Player is still P0 (dice roll happened).
                        # NN returns Val for P0.
                        # We store this.
                        values_map[orig_idx] = val
                        
                        # Note: We usually don't expand policy for chance child immediately?
                        # Standard AlphaZero expands logic node and uses P to select child.
                        # Chance: We just evaluated Value. We didn't use Policy.
                        # The newly created child is now a leaf. Next sim will expand it.
                        
                    elif mtype == 'decision':
                        # Regular Expansion
                        s_logits = spatial_logits[j]
                        self._expand_decision_node(node, val, s_logits, root_players[orig_idx])
                        
                        # Value from NN is for node.state.current_player
                        values_map[orig_idx] = val

            # --- 3. BACKPROPAGATION ---
            for i in range(batch_size):
                path = search_paths[i]
                node = leaf_nodes[i]
                
                # Determine value to backprop
                if node.is_terminal:
                    val = self._evaluate_terminal(node, root_players[i])
                else:
                    # NN Value
                    if i in values_map:
                        val = values_map[i]
                        
                        if node.is_chance: # Special case
                             # Chance child state has same player as parent? Usually yes.
                             # If expanded_rolls[i] is None here, it means logic failed.
                             roll = expanded_rolls[i]
                             if roll is not None and roll in node.children:
                                 leaf_p = node.children[roll].state.current_player
                                 if leaf_p != root_players[i]:
                                     val = -val
                             else:
                                 # Should not happen if logic is correct
                                 pass
                        else:
                             if node.state.current_player != root_players[i]:
                                 val = -val
                    else:
                         val = 0.0 # Should not happen unless empty?
                
                # Backprop loop
                for n in reversed(path):
                    n.visit_count += 1
                    if n.state.current_player == root_players[i]:
                        n.value_sum += val
                    else:
                        n.value_sum += -val


    def _select_child(self, node, root_player):
        best_score = -float('inf')
        best_child = None
        total_visits = node.visit_count
        
        for action, child in node.children.items():
            q_value = child.get_mean_value()
            if child.state.current_player != root_player:
                q_value = -q_value
            
            u_value = CPUCT * child.prior * math.sqrt(total_visits) / (1 + child.visit_count)
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child

    def _evaluate_terminal(self, node, root_player):
        winner = ludo_cpp.get_winner(node.state)
        if winner == root_player:
            return 1.0
        elif winner != -1:
            return -1.0
        return 0.0


    def _expand_decision_node(self, node, value, spatial_logits, root_player):
        # Flatten logits, Apply Mask, Softmax -> Priors
        legal_moves = ludo_cpp.get_legal_moves(node.state)
        if not legal_moves:
             return
             
        current_p = node.state.current_player
        positions = node.state.player_positions[current_p]
        
        logits_for_tokens = []
        for token_idx in legal_moves:
            pos = positions[token_idx]
            r, c = get_board_coords(current_p, pos, token_idx)
            flat_idx = r * BOARD_SIZE + c
            logits_for_tokens.append(spatial_logits[flat_idx].item())
            
        probs_tensor = torch.softmax(torch.tensor(logits_for_tokens), dim=0)
        token_probs = probs_tensor.tolist()
        
        for i, move in enumerate(legal_moves):
            next_state = ludo_cpp.apply_move(node.state, move)
            child = MCTSNode(next_state, parent=node, prior=token_probs[i], action=move)
            node.children[move] = child

def get_action_probs_vector(mcts, roots, temperature=1.0):
    """
    Get action probabilities for a batch of roots after search.
    Returns: List of [prob_0, prob_1, prob_2, prob_3]
    """
    results = []
    for root in roots:
        visit_counts = {action: child.visit_count for action, child in root.children.items()}
        
        visits = np.array([visit_counts.get(i, 0) for i in range(4)])
        if visits.sum() == 0:
            probs = [0.25] * 4
        elif temperature == 0:
            best_action = np.argmax(visits)
            probs = [0.0] * 4
            probs[best_action] = 1.0
        else:
            visits = visits ** (1.0 / temperature)
            # Safe divide
            if visits.sum() == 0:
                 probs = [0.25] * 4
            else:
                 probs = visits / visits.sum()
                 probs = probs.tolist()
        results.append(probs)
    return results

