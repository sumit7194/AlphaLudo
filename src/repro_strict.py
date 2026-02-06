import sys
import os
sys.path.append(os.getcwd())
import ludo_cpp
import numpy as np

def test_strict_repro():
    print("Creating strict state from logs...")
    state = ludo_cpp.create_initial_state()
    
    # 1. P0 (Red) State
    # T0: Home (99)
    # T1: Home (99)
    # T2: Base (-1)
    # T3: Base (-1)
    
    # P1 (Green)
    # T0: 55
    # T1: 24
    # T2: 21
    # T3: 15
    
    # P2 (Yellow)
    # T0: 51
    # T1: Home
    # T2: Home
    # T3: Home
    
    # P3 (Blue) - Just filling plausible values
    # T0: 3
    # T1: Home
    # T2: 22
    # T3: Base (-1)
    
    new_pos = state.player_positions
    
    # Red
    new_pos[0][0] = 99
    new_pos[0][1] = 99
    new_pos[0][2] = -1
    new_pos[0][3] = -1
    
    # Green
    new_pos[1][0] = 55
    new_pos[1][1] = 24
    new_pos[1][2] = 21
    new_pos[1][3] = 15
    
    # Yellow
    new_pos[2][0] = 51
    new_pos[2][1] = 99
    new_pos[2][2] = 99
    new_pos[2][3] = 99
    
    # Blue
    new_pos[3][0] = 3
    new_pos[3][1] = 99
    new_pos[3][2] = 22
    new_pos[3][3] = -1
    
    state.player_positions = new_pos
    
    # Set Scores (approx)
    state.scores[0] = 2
    state.scores[1] = 0
    state.scores[2] = 3
    state.scores[3] = 1
    
    # 2. Trigger Roll 6
    state.current_player = 0
    state.current_dice_roll = 6
    
    print(f"\n--- REPRO ATTEMPT ---")
    print(f"P0 Pos: {state.player_positions[0]}")
    print(f"Roll: {state.current_dice_roll}")
    
    legal = ludo_cpp.get_legal_moves(state)
    print(f"Legal Moves: {legal}")
    
    if not legal:
        print("!!! REPRODUCED: No moves found on 6! !!!")
    else:
        print(f"Not Reproduced: Found {len(legal)} moves.")

if __name__ == "__main__":
    test_strict_repro()
