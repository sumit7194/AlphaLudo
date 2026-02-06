import sys
import os
sys.path.append(os.getcwd())
import ludo_cpp
import numpy as np

def test_t2_at_2():
    print("Setting up 'T2 at 2' Scenario...")
    state = ludo_cpp.create_initial_state()
    
    new_pos = state.player_positions
    
    # Red: T0, T1 Home(99). T2 at 2. T3 Base(-1).
    new_pos[0][0] = 99
    new_pos[0][1] = 99
    new_pos[0][2] = 2
    new_pos[0][3] = -1
    
    # Populate opponents to ensure no blockade at 8 (target for 2+6)
    # Target = 8.
    # Safe square 8.
    
    state.player_positions = new_pos
    state.current_player = 0
    state.current_dice_roll = 6
    
    print(f"\nScenario: Red T2 at {state.player_positions[0][2]}, T3 in Base.")
    print(f"Roll: {state.current_dice_roll}")
    
    legal = ludo_cpp.get_legal_moves(state)
    print(f"Legal Moves: {legal}")
    
    # Expected: 2 (Move T2 -> 8) AND 3 (Spawn T3)
    if 2 in legal and 3 in legal:
        print("SUCCESS: Both moves found. (Normal behavior)")
    elif not legal:
        print("!!! REPRODUCED: Pass on 6 with T2@2/T3@Base!")
    else:
        print(f"PARTIAL: Found {legal}.")

if __name__ == "__main__":
    test_t2_at_2()
