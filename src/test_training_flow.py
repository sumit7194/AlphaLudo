"""
End-to-End Training Flow Test
==============================
Tests the complete training data flow:
1. Self-play game generation
2. Training example creation (states, policies, values)
3. Training step execution
4. Model gradient flow

This verifies the entire pipeline from game to gradient.
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ludo_cpp
from src.model_v3 import AlphaLudoV3
from src.tensor_utils_mastery import state_to_tensor_mastery
from src.train_v3 import TrainerV3

PASSED = []
FAILED = []

def test_passed(name):
    print(f"  ✅ {name}")
    PASSED.append(name)

def test_failed(name, reason):
    print(f"  ❌ {name}: {reason}")
    FAILED.append((name, reason))


def test_game_to_examples():
    """Test that a complete game produces valid training examples."""
    print("\n--- Test: Game → Training Examples ---")
    
    # Play a quick game
    state = ludo_cpp.create_initial_state()
    history = []
    max_moves = 100
    
    for _ in range(max_moves):
        if state.is_terminal:
            break
        
        # Roll dice
        state.current_dice_roll = np.random.randint(1, 7)
        legal = ludo_cpp.get_legal_moves(state)
        
        if not legal:
            # Re-roll (no moves)
            continue
        
        # Random policy
        policy = np.zeros(4)
        for m in legal:
            policy[m] = 1.0 / len(legal)
        
        # Record
        tensor = state_to_tensor_mastery(state)
        history.append({
            'state': tensor,
            'policy': policy,
            'player': state.current_player
        })
        
        # Apply random move
        action = np.random.choice(legal)
        state = ludo_cpp.apply_move(state, action)
    
    # Check we got some history
    if len(history) < 5:
        test_failed("Game length", f"Only {len(history)} moves recorded")
        return False
    
    test_passed(f"Game length ({len(history)} moves)")
    
    # Check tensor shapes
    for i, h in enumerate(history[:3]):
        if h['state'].shape != (21, 15, 15):
            test_failed("Tensor shape", f"Move {i}: {h['state'].shape}")
            return False
        if h['policy'].shape != (4,):
            test_failed("Policy shape", f"Move {i}: {h['policy'].shape}")
            return False
    
    test_passed("Tensor and policy shapes correct")
    
    # Assign values based on winner
    winner = ludo_cpp.get_winner(state)
    
    examples = []
    for h in history:
        p = h['player']
        if winner == -1:
            val = 0.0
        elif winner == p:
            val = 1.0
        else:
            val = -1.0
        
        examples.append({
            'state': h['state'],
            'policy': h['policy'],
            'value': val
        })
    
    # Verify value distribution
    values = [e['value'] for e in examples]
    if winner != -1 and not any(v == 1.0 for v in values):
        test_failed("Value targets", "No +1.0 values for winner")
        return False
    
    test_passed("Value targets assigned correctly")
    
    return True


def test_training_step():
    """Test that a training step produces valid gradients."""
    print("\n--- Test: Training Step → Gradients ---")
    
    model = AlphaLudoV3()
    model.train()
    
    # Create mock batch
    batch_size = 4
    states = torch.randn(batch_size, 21, 15, 15)
    policies = torch.softmax(torch.randn(batch_size, 4), dim=1)
    values = torch.randn(batch_size, 1).clamp(-1, 1)
    legal_masks = torch.ones(batch_size, 4)  # All moves legal
    
    # Forward pass
    policy_out, value_out, aux_out = model(states, legal_masks)
    
    # Check outputs
    if policy_out.shape != (batch_size, 4):
        test_failed("Policy output shape", str(policy_out.shape))
        return False
    
    if value_out.shape != (batch_size, 1):
        test_failed("Value output shape", str(value_out.shape))
        return False
    
    test_passed("Forward pass produces correct shapes")
    
    # Compute loss
    policy_loss = -torch.sum(policies * torch.log(policy_out + 1e-8)) / batch_size
    value_loss = torch.nn.functional.mse_loss(value_out, values)
    total_loss = policy_loss + value_loss
    
    if torch.isnan(total_loss):
        test_failed("Loss computation", "NaN loss")
        return False
    
    test_passed(f"Loss computed (P: {policy_loss.item():.4f}, V: {value_loss.item():.4f})")
    
    # Backward pass
    total_loss.backward()
    
    # Check gradients exist
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    if not has_grad:
        test_failed("Gradient flow", "No gradients computed")
        return False
    
    test_passed("Gradients computed and flow through model")
    
    return True


def test_legal_mask_enforcement():
    """Test that illegal moves get zero probability."""
    print("\n--- Test: Legal Mask Enforcement ---")
    
    model = AlphaLudoV3()
    model.eval()
    
    # Create state where only move 0 and 2 are legal
    states = torch.randn(1, 21, 15, 15)
    legal_mask = torch.tensor([[1, 0, 1, 0]], dtype=torch.float32)
    
    with torch.no_grad():
        policy, _, _ = model(states, legal_mask)
    
    # Check that masked moves have ~0 probability
    if policy[0, 1].item() > 1e-6:
        test_failed("Legal mask move 1", f"Should be 0, got {policy[0, 1].item()}")
        return False
    
    if policy[0, 3].item() > 1e-6:
        test_failed("Legal mask move 3", f"Should be 0, got {policy[0, 3].item()}")
        return False
    
    # Check legal moves have probability
    if policy[0, 0].item() < 0.1:
        test_failed("Legal mask move 0", f"Should be >0.1, got {policy[0, 0].item()}")
        return False
    
    test_passed("Illegal moves get zero probability")
    
    # Check probabilities sum to 1
    prob_sum = policy.sum().item()
    if abs(prob_sum - 1.0) > 0.01:
        test_failed("Probability sum", f"Should be 1.0, got {prob_sum}")
        return False
    
    test_passed("Probabilities sum to 1.0")
    
    return True


def test_value_range():
    """Test that value predictions are in [-1, 1] range."""
    print("\n--- Test: Value Range ---")
    
    model = AlphaLudoV3()
    model.eval()
    
    # Test with random inputs
    for _ in range(10):
        states = torch.randn(8, 21, 15, 15)
        with torch.no_grad():
            _, values, _ = model(states)
        
        min_v = values.min().item()
        max_v = values.max().item()
        
        if min_v < -1.0 - 1e-6 or max_v > 1.0 + 1e-6:
            test_failed("Value range", f"Values outside [-1, 1]: min={min_v}, max={max_v}")
            return False
    
    test_passed("All value predictions in [-1, 1] range")
    return True


def test_determinism():
    """Test that same input produces same output."""
    print("\n--- Test: Model Determinism ---")
    
    model = AlphaLudoV3()
    model.eval()
    
    torch.manual_seed(42)
    states = torch.randn(2, 21, 15, 15)
    
    with torch.no_grad():
        p1, v1, _ = model(states)
        p2, v2, _ = model(states)
    
    if not torch.allclose(p1, p2):
        test_failed("Policy determinism", "Different outputs for same input")
        return False
    
    if not torch.allclose(v1, v2):
        test_failed("Value determinism", "Different outputs for same input")
        return False
    
    test_passed("Same input produces same output")
    return True


def main():
    print("="*70)
    print("  END-TO-END TRAINING FLOW TEST")
    print("="*70)
    
    test_game_to_examples()
    test_training_step()
    test_legal_mask_enforcement()
    test_value_range()
    test_determinism()
    
    print("\n" + "="*70)
    print(f"  RESULTS: {len(PASSED)} PASSED, {len(FAILED)} FAILED")
    print("="*70)
    
    if FAILED:
        print("\nFailed Tests:")
        for name, reason in FAILED:
            print(f"  - {name}: {reason}")
        return 1
    else:
        print("\n✅ ALL TRAINING FLOW TESTS PASSED!")
        return 0


if __name__ == "__main__":
    exit(main())
