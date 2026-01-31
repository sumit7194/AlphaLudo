
import sys
import os
import numpy as np
import torch # Mocking depends

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

try:
    import ludo_cpp
except ImportError:
    print("Error: Could not import ludo_cpp.")
    sys.exit(1)

def repro_reset():
    print("--- Repro Game Reset Logic ---")
    
    # 1. Setup State: P3 about to win
    state = ludo_cpp.GameState()
    
    # Manually setup P3 to win
    # P3 has 3 tokens home (scores[3] = 3)
    # P3 has 1 token at 55 (End of track)
    # Needs 1 to win (Home is 56)
    
    # We can't inject state easily into C++ object if attributes aren't writable lists...
    # Bindings show `scores` and `player_positions` are properties copyable from numpy.
    # Let's try to verify if we can set them.
    # Actually, we can just use `create_initial_state` and simulate? No, 300 turns.
    # We will assume we can set terminal flag or mock the transition.
    
    # Actually, let's just RUN logic on a Terminated State created manually?
    # Or rely on the code snippet logic directly.
    
    # Creating a dummy state and assuming it IS terminal for the loop check.
    
    print("1. Simulating End of Game (Terminated State)...")
    idx = 0
    states = {}
    states[idx] = ludo_cpp.create_initial_state()
    state = states[idx]
    
    # Mock Termination
    state.is_terminal = True
    # We don't care about internal board state for this test, just the Reset Logic flow.
    
    # 2. Logic Block from vector_league.py
    completed_episodes = 0
    target_episodes = 10
    
    print("\n2. Entering Reset Logic Block...")
    
    # Mimic vector_league.py lines 272+
    is_terminal = state.is_terminal # True
    
    if is_terminal:
        winner = 3 # P3 Won
        print(f"   [Logic] Game Finished. Winner: {winner}")
        
        completed_episodes += 1
        
        if completed_episodes < target_episodes:
            print("   [Logic] Resetting Game...")
            states[idx] = ludo_cpp.create_initial_state()
            state = states[idx] # THIS IS THE FIX I ADDED.
            print(f"   [Logic] State Replaced. New Player: {state.current_player} (Should be 0)")
            
            # Reset other buffers
            # histories...
            # shaped_returns...
            
            # Visualizer Broadcast (Mock)
            print("   [Logic] Broadcasting New State to Visualizer")

    # 3. Next Loop Logic (Lines 313+)
    print("\n3. Entering Next Loop Logic...")
    
    # Current Dice Roll Logic
    print(f"   [Logic] Current Dice Before: {state.current_dice_roll}")
    if state.current_dice_roll == 0:
        state.current_dice_roll = 3 # Force non-6 to test skip
        print(f"   [Logic] Rolled Dice: {state.current_dice_roll}")
    
    legal_moves = ludo_cpp.get_legal_moves(state)
    print(f"   [Logic] Legal Moves: {legal_moves}")
    
    if len(legal_moves) == 0:
        print("   [Logic] NO LEGAL MOVES -> Executing Skip Logic")
        # Broadcast Skip
        print(f"   [Logic] Broadcasting Move: Player {state.current_player} -> Action -1 (Skip) with Dice {state.current_dice_roll}")
        
        # Advance Turn
        state.current_player = (state.current_player + 1) % 4
        state.current_dice_roll = 0
        print(f"   [Logic] Turn Advanced to P{state.current_player}")
        
    else:
        print("   [Logic] Moves Available. Agent Move...")

    print("\n--- End Repro ---")

if __name__ == "__main__":
    repro_reset()
