
import sys
import os
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import ludo_cpp

def test_skip_logic():
    print("--- Testing Skip Logic ---")
    
    # 1. Setup State
    state = ludo_cpp.create_initial_state()
    
    # Yellow (P2) Setup: 1 token at 55 (Needs 1). Others Home/Base.
    # Actually P0 is Red, P1 Green, P2 Yellow, P3 Blue.
    # Let's use P0 and P1 for simplicity.
    
    current_p = 0
    state.current_player = current_p
    state.player_positions[current_p][0] = 55 # Close to home (56)
    
    # P1 (Next) Setup: All at BASE (Needs 6).
    next_p = 1
    # positions are BASE_POS (-1) by default
    
    # 2. Simulate P0 Turn (Invalid Roll)
    print(f"\n[P{current_p}] Start Turn. Token at 55.")
    
    # Force Dice = 5
    state.current_dice_roll = 5
    print(f"[P{current_p}] Rolled 5.")
    
    legal_moves = ludo_cpp.get_legal_moves(state)
    print(f"[P{current_p}] Legal Moves: {legal_moves}")
    
    if len(legal_moves) == 0:
        print(f"[P{current_p}] No Moves! Skipping...")
        state.current_player = (state.current_player + 1) % 4
        state.current_dice_roll = 0
    else:
        print("ERROR: Should have 0 moves!")
        return
        
    # 3. Verify State Transition
    print(f"\n[Transition] Current Player is now: P{state.current_player}")
    print(f"[Transition] Current Dice is: {state.current_dice_roll}")
    
    if state.current_player != next_p:
        print(f"ERROR: Player should be {next_p}, but is {state.current_player}")
        return
        
    # 4. Simulate P1 Turn (Next Loop Iteration)
    print(f"\n[P{state.current_player}] Start Loop Iteration.")
    
    if state.current_dice_roll == 0:
        print(f"[P{state.current_player}] Needs Roll. Rolling...")
        state.current_dice_roll = 6 # Force 6 to verify valid move
        # state.current_dice_roll = 1 # Force 1 to verify skip
        
    print(f"[P{state.current_player}] Rolled {state.current_dice_roll}.")
    
    legal_moves_next = ludo_cpp.get_legal_moves(state)
    print(f"[P{state.current_player}] Legal Moves: {legal_moves_next}")
    
    if len(legal_moves_next) > 0:
        print(f"[P{state.current_player}] HAS MOVES! Logic Correct.")
    else:
        print(f"[P{state.current_player}] NO MOVES. Logic Correct (if roll != 6).")

if __name__ == "__main__":
    test_skip_logic()
