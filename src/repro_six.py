import sys
import os
sys.path.append(os.getcwd())
import ludo_cpp
import numpy as np

def test_six_spawn():
    print("Creating state...")
    state = ludo_cpp.create_initial_state()
    
    # Empty board mostly, but ensure we have tokens in base
    # By default new state has all in base
    
    # P0 Turn
    state.current_player = 0
    state.current_dice_roll = 6
    
    print(f"Player {state.current_player} Rolled {state.current_dice_roll}")
    print(f"Positions: {state.player_positions[0]}")
    
    legal = ludo_cpp.get_legal_moves(state)
    print(f"Legal Moves: {legal}")
    
    if not legal:
        print("FAIL: No legal moves on 6 from base!")
    else:
        print("SUCCESS: Found moves on 6.")

    # Test Case 2: Some tokens out, some in base
    print("\nTest Case 2: One token out, roll 6")
    # Move T0 out (manually hack position if possible, or play move)
    # Applying move requires generating one. 
    # Let's just assume we can apply a move if legal.
    if legal:
        # Expected: moving a token out of base sends it to Pos 0 (-1 -> 0)
        # So move index corresponds to token index usually?
        move = legal[0]
        state = ludo_cpp.apply_move(state, move)
        print(f"Applied move {move}. Pos: {state.player_positions[0]}")
        
        # Now roll 6 again
        state.current_player = 0
        state.current_dice_roll = 6
        legal_2 = ludo_cpp.get_legal_moves(state)
        print(f"Roll 6 Again. Pos: {state.player_positions[0]}")
        print(f"Legal Moves: {legal_2}")
        
        if not legal_2:
             print("FAIL: Cannot spawn second token on 6?")
        elif len(legal_2) < 2:
             print(f"WARN: Found {len(legal_2)} moves (Expected >1: move T0 or spawn T1)")

if __name__ == "__main__":
    test_six_spawn()
