import torch
import numpy as np

# Board Constants
BOARD_SIZE = 15
NUM_PLAYERS = 4
NUM_TOKENS = 4
HOME_POS = 99
BASE_POS = -1

# Coordinate Lookup Tables (Mapped from C++ implementation)
# P0 (Bottom Left) specific paths
PATH_COORDS_P0 = [
    (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), # 0-4
    (5, 6), (4, 6), (3, 6), (2, 6), (1, 6), (0, 6), # 5-10
    (0, 7), (0, 8), # 11-12
    (1, 8), (2, 8), (3, 8), (4, 8), (5, 8), # 13-17
    (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), (6, 14), # 18-23
    (7, 14), (8, 14), # 24-25
    (8, 13), (8, 12), (8, 11), (8, 10), (8, 9), # 26-30
    (9, 8), (10, 8), (11, 8), (12, 8), (13, 8), (14, 8), # 31-36
    (14, 7), (14, 6), # 37-38
    (13, 6), (12, 6), (11, 6), (10, 6), (9, 6), # 39-43
    (8, 5), (8, 4), (8, 3), (8, 2), (8, 1), (8, 0), # 44-49
    (7, 0) # 50 (End of main track)
]

HOME_RUN_P0 = [
    (7, 1), (7, 2), (7, 3), (7, 4), (7, 5)
]

HOME_COORD_P0 = (7, 6)

BASE_COORDS = [
    [(2, 2), (2, 3), (3, 2), (3, 3)],     # P0 (Top Left) - Red
    [(2, 11), (2, 12), (3, 11), (3, 12)], # P1 (Top Right) - Green
    [(11, 11), (11, 12), (12, 11), (12, 12)], # P2 (Bottom Right) - Yellow
    [(11, 2), (11, 3), (12, 2), (12, 3)] # P3 (Bottom Left) - Blue
]

SAFE_INDICES = {0, 8, 13, 21, 26, 34, 39, 47}

def get_board_coords(player, pos, token_idx=0):
    """Maps a player's token position to (row, col) on the 15x15 board."""
    if pos == BASE_POS:
        # Map to specific base spot based on token index
        return BASE_COORDS[player][token_idx]
    
    local_r, local_c = 0, 0
    
    if pos == HOME_POS:
        local_r, local_c = HOME_COORD_P0
    elif pos > 50: # Home run (indices 51-55 in my logic, but verify C++ logic)
        # In C++, I used:
        # > 50 is home run. 52-56 is typical indices for implementation plan.
        # Let's align with C++: "0-50 (Path), 51-55 (Home Run), 99 (Home)"
        # Wait, C++ said "0-50 (Path)". 
        # PATH_COORDS_P0 has 51 entries (0-50).
        # So 0-50 is Main Path.
        # 51-55 is Home Run.
        
        idx = pos - 51
        if 0 <= idx < 5:
            local_r, local_c = HOME_RUN_P0[idx]
        else:
            local_r, local_c = HOME_COORD_P0
    elif 0 <= pos < 51:
        local_r, local_c = PATH_COORDS_P0[pos]
    else:
        # Should not happen
        return -1, -1

    # Rotate based on player
    # P0: 0, P1: 90, P2: 180, P3: 270
    r, c = local_r, local_c
    for _ in range(player):
        # Rotate 90 deg clockwise around (7, 7)
        # (r, c) -> (c, 14-r)
        r, c = c, 14 - r
        
    return r, c

def get_safe_mask():
    """Generates the constant safe zone mask."""
    mask = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    # Safe indices on path for all players
    for p in range(NUM_PLAYERS):
        for idx in SAFE_INDICES:
            # Map P0 safe index to global board relative to Player P
            # We can use get_board_coords treating it as a path pos for Player P
            r, c = get_board_coords(p, idx)
            mask[r, c] = 1.0
            
    # Bases are also safe? Technically yes, you can't be cut in base. 
    # But usually "safe zone" refers to stars on the track.
    # Plan says "Channel 4: Safe zones (constant)".
    return mask

def get_home_path_mask():
    """Generates the constant home path mask."""
    mask = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    # Mark home run squares
    for p in range(NUM_PLAYERS):
        for i in range(5):
             r, c = get_board_coords(p, 52 + i)
             mask[r, c] = 1.0
        # Home itself?
        r, c = get_board_coords(p, HOME_POS)
        mask[r, c] = 1.0
    return mask

# Precompute constant masks
SAFE_MASK = torch.from_numpy(get_safe_mask())
HOME_PATH_MASK = torch.from_numpy(get_home_path_mask())

def state_to_tensor(state):
    """
    Converts GameState to (8, 15, 15) tensor.
    Channels:
    0-3: Player masks
    4: Safe zones
    5: Home paths
    6: Dice roll
    7: Turn indicator
    """
    # Create empty tensor
    # Using numpy for easier indexing then converting
    tensor_np = np.zeros((8, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    
    # 0-3: Player Masks
    # We need to access state.player_positions.
    # The C++ binding exposes it as a numpy array.
    
    positions = state.player_positions # Shape (4, 4)
    
    for p in range(NUM_PLAYERS):
        for t in range(NUM_TOKENS):
            pos = positions[p][t]
            r, c = get_board_coords(p, pos, t)
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                tensor_np[p, r, c] = 1.0
                
    # 4: Safe Zones
    tensor_np[4] = SAFE_MASK.numpy()
    
    # 5: Home Paths
    tensor_np[5] = HOME_PATH_MASK.numpy()
    
    # 6: Dice Roll
    # Normalize 1-6 to 1/6 - 1.0
    roll_val = state.current_dice_roll / 6.0
    tensor_np[6].fill(roll_val)
    
    # 7: Turn Indicator
    # "filled with 0, 0.25, 0.5, or 1.0" -> I'll use 0.0, 0.33, 0.66, 1.0 mapping or 0.25 steps
    # Player 0 -> 0.0, P1 -> 0.25, P2 -> 0.5, P3 -> 0.75? Or 0.25, 0.5, 0.75, 1.0?
    # Let's do (current_player + 1) * 0.25
    turn_val = (state.current_player + 1) * 0.25
    tensor_np[7].fill(turn_val)
    
    return torch.from_numpy(tensor_np)
