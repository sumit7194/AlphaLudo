
import sys
import os
sys.path.append(os.path.abspath('src'))

try:
    import ludo_cpp
except ImportError:
    print("Failed to import ludo_cpp. Ensure it is compiled and in path.")
    sys.exit(1)

def test_mutation():
    print("Creating initial state...")
    state = ludo_cpp.create_initial_state()
    
    # Verify Initial State
    # P0 Token 0 should be -1 (Base)
    print(f"Initial P0 T0 Pos: {state.player_positions[0][0]}")
    
    # Setup for Move
    state.current_player = 0
    state.current_dice_roll = 6
    action = 0 # Move Token 0
    
    print(f"Applying Action {action} (Dice 6)...")
    new_state = ludo_cpp.apply_move(state, action)
    
    if new_state is None:
        print("FAILURE: apply_move returned None.")
        return

    # Check Mutation on New State
    new_pos = new_state.player_positions[0][0]
    print(f"New P0 T0 Pos (in new_state): {new_pos}")
    print(f"Old P0 T0 Pos (in old state): {state.player_positions[0][0]}")
    
    if new_pos != -1:
        print("SUCCESS: New state has updated position.")
    else:
        print("FAILURE: New state did not change.")

if __name__ == "__main__":
    test_mutation()
