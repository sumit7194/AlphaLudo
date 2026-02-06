import sys
import os
sys.path.append(os.getcwd())
import ludo_cpp
import numpy as np

def test_numpy_int():
    print("Testing NumPy int64 assignment to C++ state...")
    state = ludo_cpp.create_initial_state()
    
    # 1. Setup Base State
    new_pos = state.player_positions
    new_pos[0][0] = 99
    new_pos[0][1] = 99
    new_pos[0][2] = -1
    new_pos[0][3] = -1
    state.player_positions = new_pos
    state.current_player = 0
    
    # 2. Assign np.int64(6)
    roll_np = np.int64(6)
    print(f"Assigning {type(roll_np)}: {roll_np}")
    
    try:
        state.current_dice_roll = roll_np
    except Exception as e:
        print(f"Assignment Attempt Failed: {e}")
        # Even if it fails, let's see what happens
    
    # Check what C++ sees (by reading it back)
    read_back = state.current_dice_roll
    print(f"Read back from state: {type(read_back)} {read_back}")
    
    if read_back != 6:
        print(f"CRITICAL: Value corrupted! Expected 6, got {read_back}")
    
    legal = ludo_cpp.get_legal_moves(state)
    print(f"Legal Moves: {legal}")
    
    if not legal:
         print("FAIL: NumPy assignment caused No Moves!")
    else:
         print("SUCCESS: NumPy assignment worked.")

if __name__ == "__main__":
    test_numpy_int()
