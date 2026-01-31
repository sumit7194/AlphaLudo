import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import ludo_cpp
from mcts import MCTS, MCTSNode, get_action_probs
from model import AlphaLudoNet

def test_mcts_node_types():
    """Test that Decision and Chance nodes are correctly identified."""
    # Decision Node (dice rolled)
    state = ludo_cpp.create_initial_state()
    state.current_dice_roll = 6
    node = MCTSNode(state)
    assert not node.is_chance, "Node with dice roll should be Decision Node"
    
    # Chance Node (dice not rolled)
    state2 = ludo_cpp.create_initial_state()
    state2.current_dice_roll = 0
    node2 = MCTSNode(state2)
    assert node2.is_chance, "Node without dice roll should be Chance Node"
    
    print("✓ Node type test passed.")

def test_mcts_search():
    """Test that MCTS search runs and produces valid results."""
    model = AlphaLudoNet()
    mcts = MCTS(model, num_simulations=50)
    
    state = ludo_cpp.create_initial_state()
    state.current_dice_roll = 6  # Allows leaving base
    
    visit_counts = mcts.search(state)
    
    # All 4 tokens should be legal with a 6 from base
    assert len(visit_counts) == 4, f"Expected 4 legal moves, got {len(visit_counts)}"
    
    # Total visits should be close to num_simulations (may differ by 1 due to root handling)
    total_visits = sum(visit_counts.values())
    assert 48 <= total_visits <= 50, f"Expected ~50 visits, got {total_visits}"
    
    print(f"✓ MCTS search test passed. Visit distribution: {visit_counts}")

def test_chance_node_full_expansion():
    """Test that Chance Nodes expand all 6 dice outcomes."""
    model = AlphaLudoNet()
    mcts = MCTS(model, num_simulations=10)
    
    state = ludo_cpp.create_initial_state()
    state.current_dice_roll = 6
    
    # Apply a move to get to a Chance Node
    next_state = ludo_cpp.apply_move(state, 0)
    assert next_state.current_dice_roll == 0, "After move, dice should be 0"
    
    # Create a chance node
    chance_node = MCTSNode(next_state)
    assert chance_node.is_chance, "Should be a Chance Node"
    
    # Expand it via MCTS internal method
    mcts.root_player = state.current_player
    mcts._expand_and_evaluate(chance_node)
    
    # Check all 6 children exist
    assert len(chance_node.children) == 6, f"Expected 6 dice children, got {len(chance_node.children)}"
    
    for roll in range(1, 7):
        assert roll in chance_node.children, f"Missing child for dice roll {roll}"
    
    print("✓ Chance Node full expansion test passed.")

def test_action_probs():
    """Test that action probabilities are valid."""
    model = AlphaLudoNet()
    mcts = MCTS(model, num_simulations=20)
    
    state = ludo_cpp.create_initial_state()
    state.current_dice_roll = 6
    
    probs = get_action_probs(mcts, state, temperature=1.0)
    
    assert len(probs) == 4, "Should return 4 probabilities"
    assert abs(sum(probs) - 1.0) < 0.01, "Probabilities should sum to 1"
    
    print(f"✓ Action probs test passed. Probs: {probs}")

if __name__ == "__main__":
    test_mcts_node_types()
    test_mcts_search()
    test_chance_node_full_expansion()
    test_action_probs()
    print("\n🎉 All MCTS tests passed!")
