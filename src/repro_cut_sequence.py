import sys
import os
sys.path.append(os.getcwd())
import ludo_cpp
import numpy as np

def test_cut_sequence():
    print("Setting up 'Cut at 4' Scenario...")
    state = ludo_cpp.create_initial_state()
    
    # Setup P0 (Red) T0 at 0
    # Setup P1 (Green) T0 at 4 (Absolute 4)
    # Green starts at 13.
    # Relative pos for Green to be at Abs 4?
    # Abs = (Rel + 13) % 52
    # 4 = (Rel + 13) % 52
    # Rel = -9 -> 43.
    # So Green T0 at 43.
    
    # Verify mapping
    # 43 + 13 = 56. 56 % 52 = 4. Correct.
    
    new_pos = state.player_positions
    
    # Red T0 at 0
    new_pos[0][0] = 0
    new_pos[0][1] = -1 # Base
    new_pos[0][2] = -1
    new_pos[0][3] = -1
    
    # Green T0 at 43 (Abs 4)
    new_pos[1][0] = 43
    new_pos[1][1] = -1
    new_pos[1][2] = -1
    new_pos[1][3] = -1
    
    state.player_positions = new_pos
    state.current_player = 0 # Red
    
    # 1. Red Rolls 4
    state.current_dice_roll = 4
    print(f"\nStep 1: Red Rolled 4. T0 at {state.player_positions[0][0]}")
    
    legal = ludo_cpp.get_legal_moves(state)
    print(f"Legal Moves: {legal}")
    
    # Apply Move 0 (T0)
    move_idx = 0
    print(f"Applying Move {move_idx}...")
    state = ludo_cpp.apply_move(state, move_idx)
    
    # Check Result
    print(f"Red Pos: {state.player_positions[0]}")
    print(f"Green Pos: {state.player_positions[1]}")
    
    if state.player_positions[0][0] == 4 and state.player_positions[1][0] == -1:
        print("SUCCESS: Cut executed. Green T0 sent to Base.")
    else:
        print("FAIL: Cut did not happen as expected.")
    
    # Check Turn
    print(f"Next Player: {state.current_player} (Expected 0 for Bonus)")
    
    if state.current_player != 0:
        print("FAIL: No Bonus Turn given!")
        return
    
    # 2. Red Rolls 6
    state.current_dice_roll = 6
    print(f"\nStep 2: Red Rolled 6. T0 at {state.player_positions[0][0]}")
    
    legal_2 = ludo_cpp.get_legal_moves(state)
    print(f"Legal Moves: {legal_2}")
    
    if not legal_2:
        print("!!! REPRODUCED: Pass on 6 after Cut!")
    else:
        print(f"Moves Found: {legal_2}. (Normal behavior)")

if __name__ == "__main__":
    test_cut_sequence()
