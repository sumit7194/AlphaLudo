"""
Prove Ambiguity in v3 Input
---------------------------
Demonstrates that the current v3 input tensor cannot distinguish between
different token assignments, making learning impossible for an index-based policy.
"""

import os
import sys
import numpy as np
import torch
import ludo_cpp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tensor_utils_mastery import state_to_tensor_mastery

def prove_ambiguity():
    print("="*60)
    print("  PROOF OF INPUT AMBIGUITY")
    print("="*60)
    
    # State A: Token 0 at 10, Token 1 at 20
    state_a = ludo_cpp.GameState()
    state_a.player_positions[0][0] = 10
    state_a.player_positions[0][1] = 20
    state_a.player_positions[0][2] = -1
    state_a.player_positions[0][3] = -1
    
    # State B: Token 0 at 20, Token 1 at 10 (Swapped)
    state_b = ludo_cpp.GameState()
    state_b.player_positions[0][0] = 20
    state_b.player_positions[0][1] = 10
    state_b.player_positions[0][2] = -1
    state_b.player_positions[0][3] = -1
    
    print("\n--- State A ---")
    print(f"Token 0: Pos 10")
    print(f"Token 1: Pos 20")
    
    print("\n--- State B (Swapped) ---")
    print(f"Token 0: Pos 20")
    print(f"Token 1: Pos 10")
    
    # Convert to Tensors
    tensor_a = state_to_tensor_mastery(state_a)
    tensor_b = state_to_tensor_mastery(state_b)
    
    # Compare Channel 0 (My Pieces)
    # Channel 0 is index 0 in tensor (since P0 is current)
    ch0_a = tensor_a[0].numpy()
    ch0_b = tensor_b[0].numpy()
    
    print("\n--- Comparing Input Tensors (Channel 0: My Tokens) ---")
    
    if np.array_equal(ch0_a, ch0_b):
        print("❌ CRITICAL FAIL: Input tensors are IDENTICAL!")
        print("   The network receives the EXACT SAME input for both states.")
    else:
        print("✅ Success: Inputs are different.")
        
    # Analysis
    print("\n--- Why this kills learning ---")
    print("In State A, the correct move (e.g. escaping danger at 10) might be Token 0.")
    print("In State B, the correct move (escaping danger at 10) is Token 1.")
    print("\nNetwork sees Input X.")
    print("Trainer says: 'For Input X, output Action 0'.")
    print("Trainer also says: 'For Input X, output Action 1'.")
    print("Result: Network learns the average (uniform distribution) or collapses.")
    print("This explains the 1% win rate.")

if __name__ == "__main__":
    prove_ambiguity()
