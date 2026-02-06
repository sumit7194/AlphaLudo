import sys
import os
sys.path.append(os.getcwd())
import ludo_cpp
import numpy as np

def test_mixed_state_six():
    print("Creating mixed state (2 Home, 2 Base)...")
    state = ludo_cpp.create_initial_state()
    
    # Manually hack the state if bindings allow, or simulate moves.
    # Bindings usually expose read-write properties for arrays if configured well.
    # Let's try direct assignment.
    
    # P0 State: T0=Home(99), T1=Home(99), T2=Base(-1), T3=Base(-1)
    # T0
    # In C++ bindings, `player_positions` is often a list of lists or numpy array.
    # Let's check type in previous logs. It printed as [-1 -1 -1 -1].
    # So it supports assignment?
    
    # We might need to iterate valid moves to get there, but that's hard.
    # Let's hope pybind11 allows list/array assignment.
    
    try:
        # Construct the full positions array
        # 4 players, 4 tokens each
        new_pos = state.player_positions
        # P0
        new_pos[0][0] = 99 # Home
        new_pos[0][1] = 99 # Home
        new_pos[0][2] = -1 # Base
        new_pos[0][3] = -1 # Base
        
        # P1-P3 Base
        for p in range(1, 4):
            for t in range(4):
                new_pos[p][t] = -1
        
        state.player_positions = new_pos
        print("State modification successful.")
    except Exception as e:
        print(f"Direct assignment failed: {e}")
        return

    # Set Turn P0, Roll 6
    state.current_player = 0
    state.current_dice_roll = 6
    
    print(f"Player {state.current_player} Rolled {state.current_dice_roll}")
    print(f"Positions P0: {state.player_positions[0]}")
    
    legal = ludo_cpp.get_legal_moves(state)
    print(f"Legal Moves: {legal}")
    
    if not legal:
        print("FAIL: No legal moves on 6 with Mixed State (Home/Base)!")
    else:
        print(f"SUCCESS: Found {len(legal)} moves: {legal}")
        # Expectation: T2 and T3 should be able to spawn (Moves 2, 3)

if __name__ == "__main__":
    test_mixed_state_six()
