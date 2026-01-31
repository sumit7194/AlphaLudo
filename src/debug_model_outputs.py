"""
Debug Model Outputs
-------------------
Inspects raw model outputs for sample game states to understand what's happening.
"""

import os
import sys
import random
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ludo_cpp
from src.model_v3 import AlphaLudoV3
from src.tensor_utils_mastery import state_to_tensor_mastery
from src.config import MAIN_CKPT_PATH


def debug_model_outputs():
    """Debug model outputs for various game states."""
    
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = AlphaLudoV3(num_res_blocks=10, num_channels=128)
    model.to(device)
    model.eval()
    
    if os.path.exists(MAIN_CKPT_PATH):
        ckpt = torch.load(MAIN_CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded checkpoint: {MAIN_CKPT_PATH}")
    else:
        print("No checkpoint found! Using random weights.")
    
    print("\n" + "="*70)
    print("  DEBUGGING MODEL OUTPUTS")
    print("="*70)
    
    random.seed(42)
    np.random.seed(42)
    
    # Test 1: Fresh game start with dice roll 6
    print("\n--- TEST 1: Fresh game, dice=6 ---")
    state = ludo_cpp.GameState()
    state.current_dice_roll = 6
    legal_moves = ludo_cpp.get_legal_moves(state)
    print(f"Current player: {state.current_player}")
    print(f"Dice roll: {state.current_dice_roll}")
    print(f"Legal moves: {legal_moves}")
    print(f"Player positions: {list(state.player_positions[0])}")
    
    state_tensor = state_to_tensor_mastery(state)
    input_tensor = state_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
    
    with torch.no_grad():
        policy, value = model.forward_policy_value(input_tensor)
    
    policy_np = policy[0].cpu().numpy()
    value_np = value[0].cpu().item()
    
    print(f"\nRaw Policy Output: {policy_np}")
    print(f"Policy Sum: {policy_np.sum():.4f}")
    print(f"Policy Max: {policy_np.max():.4f} at index {policy_np.argmax()}")
    print(f"Policy Min: {policy_np.min():.4f}")
    print(f"Value Output: {value_np:.4f}")
    
    # Masked policy
    masked = np.zeros(4)
    for m in legal_moves:
        masked[m] = policy_np[m]
    if masked.sum() > 0:
        masked /= masked.sum()
    print(f"Masked & Normalized: {masked}")
    print(f"Chosen action (greedy): {masked.argmax()}")
    
    # Test 2: Mid-game with multiple tokens out
    print("\n--- TEST 2: Mid-game scenario ---")
    state2 = ludo_cpp.GameState()
    # Manually set up a mid-game scenario
    state2.player_positions[0][0] = 10  # Token 0 at position 10
    state2.player_positions[0][1] = 20  # Token 1 at position 20
    state2.player_positions[0][2] = -1  # Token 2 in base
    state2.player_positions[0][3] = -1  # Token 3 in base
    state2.current_dice_roll = 4
    legal_moves2 = ludo_cpp.get_legal_moves(state2)
    
    print(f"Player positions: {list(state2.player_positions[0])}")
    print(f"Dice roll: {state2.current_dice_roll}")
    print(f"Legal moves: {legal_moves2}")
    
    state_tensor2 = state_to_tensor_mastery(state2)
    input_tensor2 = state_tensor2.unsqueeze(0).to(device=device, dtype=torch.float32)
    
    with torch.no_grad():
        policy2, value2 = model.forward_policy_value(input_tensor2)
    
    policy_np2 = policy2[0].cpu().numpy()
    value_np2 = value2[0].cpu().item()
    
    print(f"\nRaw Policy Output: {policy_np2}")
    print(f"Policy Sum: {policy_np2.sum():.4f}")
    print(f"Value Output: {value_np2:.4f}")
    
    # Masked policy
    masked2 = np.zeros(4)
    for m in legal_moves2:
        masked2[m] = policy_np2[m]
    if masked2.sum() > 0:
        masked2 /= masked2.sum()
    print(f"Masked & Normalized: {masked2}")
    
    # Test 3: Check entropy of outputs
    print("\n--- TEST 3: Entropy Analysis (10 random states) ---")
    entropies = []
    for i in range(10):
        state = ludo_cpp.GameState()
        # Random moves to get different states
        for _ in range(random.randint(5, 20)):
            state.current_dice_roll = random.randint(1, 6)
            moves = ludo_cpp.get_legal_moves(state)
            if moves:
                state = ludo_cpp.apply_move(state, random.choice(moves))
            else:
                state.current_player = (state.current_player + 1) % 4
            if state.is_terminal:
                break
        
        if state.is_terminal:
            continue
            
        state.current_dice_roll = random.randint(1, 6)
        state_tensor = state_to_tensor_mastery(state)
        input_tensor = state_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
        
        with torch.no_grad():
            policy, value = model.forward_policy_value(input_tensor)
        
        policy_np = policy[0].cpu().numpy()
        
        # Calculate entropy
        entropy = -np.sum(policy_np * np.log(policy_np + 1e-10))
        entropies.append(entropy)
        
        print(f"  State {i+1}: Policy={policy_np}, Entropy={entropy:.3f}, Value={value[0].item():.3f}")
    
    if entropies:
        avg_entropy = np.mean(entropies)
        print(f"\nAverage Entropy: {avg_entropy:.3f}")
        print(f"Max Possible Entropy (uniform over 4): {np.log(4):.3f}")
        print(f"Entropy Ratio: {avg_entropy / np.log(4) * 100:.1f}%")
    
    # Test 4: Check if model differentiates between players
    print("\n--- TEST 4: Same board, different current player ---")
    state = ludo_cpp.GameState()
    state.player_positions[0][0] = 10
    state.current_dice_roll = 3
    
    for player in range(4):
        state.current_player = player
        state_tensor = state_to_tensor_mastery(state)
        input_tensor = state_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
        
        with torch.no_grad():
            policy, value = model.forward_policy_value(input_tensor)
        
        print(f"  Player {player}: Policy={policy[0].cpu().numpy()}, Value={value[0].item():.3f}")
    
    print("\n" + "="*70)
    print("  DIAGNOSIS")
    print("="*70)
    
    if entropies:
        avg_entropy = np.mean(entropies)
        if avg_entropy > 1.3:  # Close to uniform entropy of ~1.39
            print("⚠️  ISSUE: Policies are near-uniform (high entropy).")
            print("   The model is not learning to differentiate between actions.")
        elif avg_entropy < 0.5:
            print("✅ Policies show strong preferences (low entropy).")
        else:
            print("🔶 Policies show moderate differentiation.")


if __name__ == "__main__":
    debug_model_outputs()
