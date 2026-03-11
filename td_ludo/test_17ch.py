import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import td_ludo_cpp
import random
import numpy as np

def run_test():
    print("Initializing 2-Player Ludo Vector Environment...")
    env = td_ludo_cpp.VectorGameState(batch_size=1, two_player_mode=True)
    
    # We will step randomly until we get a valid dice roll (1-6)
    # Actually, a new game starts with P0 needing a roll. 
    # Let's forcefully set the dice roll or just step until we get one.
    
    game = env.get_game(0)
    print(f"Initial Dice Roll: {game.current_dice_roll}")
    
    # Force a dice roll to guarantee testing
    # It might be 0 initially.
    test_roll = 4
    game.current_dice_roll = test_roll
    print(f"Forcing Dice Roll to: {test_roll}")
    
    # Get the state tensor
    state_tensor = env.get_state_tensor()
    print(f"State Tensor Shape: {state_tensor.shape}")
    
    assert state_tensor.shape == (1, 17, 15, 15), "ERROR: Shape is not 17 channels!"
    
    # Check the dice roll channels (11 to 16)
    matrix = state_tensor[0]
    
    for i in range(11, 17):
        channel = matrix[i]
        roll_represented = i - 11 + 1
        avg_val = np.mean(channel)
        print(f"Channel {i} (Roll={roll_represented}): Mean Value = {avg_val:.2f}")
        
        if roll_represented == test_roll:
            assert avg_val == 1.0, f"ERROR: Expected Channel {i} to be fully 1.0!"
        else:
            assert avg_val == 0.0, f"ERROR: Expected Channel {i} to be completely 0.0!"
            
    print("SUCCESS: 17-Channel Encoding and One-Hot Dice Roll perfectly verified!")

if __name__ == "__main__":
    run_test()
