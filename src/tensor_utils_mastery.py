
import torch
import numpy as np
import ludo_cpp
from src.tensor_utils import get_board_coords, SAFE_MASK, HOME_PATH_MASK, NUM_PLAYERS, NUM_TOKENS, BOARD_SIZE, BASE_POS

def get_home_run_masks():
    """
    Generates 4 separate masks for the Home Run paths of P0, P1, P2, P3.
    Returns: numpy array of shape (4, 15, 15)
    """
    masks = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for p in range(NUM_PLAYERS):
        for i in range(5):
             # FIXED: Indices 51-55 (5 squares). Previously started at 52 (Off-By-One).
             r, c = get_board_coords(p, 51 + i)
             if r >= 0 and c >= 0:
                masks[p, r, c] = 1.0
    return masks

HOME_RUN_MASKS = torch.from_numpy(get_home_run_masks())

def state_to_tensor_mastery(state):
    """
    Converts GameState to (18, 15, 15) single spatial tensor.
    
    Channels:
    0-3: Pieces (Density / 4.0) [Me, Next, Team, Prev]
    4-7: Home Paths (Binary) [Me, Next, Team, Prev]
    8: Safe Zones (Binary)
    9-14: Dice One-Hot (Roll 1-6)
    15: Score Diff (Broadcast)
    16: Locked State (Broadcast)
    17: Race Progress (Broadcast)
    """
    current_p = state.current_player
    positions = state.player_positions
    scores = state.scores
    
    # Initialize 21-channel tensor
    final_tensor = np.zeros((21, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    
    # --- CHANNELS 0-3: My Tokens (Distinct Identity) ---
    # Ch 0: My Token 0
    # Ch 1: My Token 1
    # Ch 2: My Token 2
    # Ch 3: My Token 3
    for t in range(4):
        pos = positions[current_p][t]
        if pos == BASE_POS:
            r, c = get_board_coords(current_p, pos, t)
        else:
            r, c = get_board_coords(current_p, pos, 0)
        
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
             final_tensor[t, r, c] = 1.0

    # --- CHANNELS 4-6: Opponent Pieces (Density) ---
    # Ch 4: Next Player
    # Ch 5: Team Player
    # Ch 6: Prev Player
    for p_offset in range(1, 4):
        p = (current_p + p_offset) % 4
        target_ch = 3 + p_offset # 4, 5, 6
        
        for t in range(4):
            pos = positions[p][t]
            if pos == BASE_POS:
                r, c = get_board_coords(p, pos, t)
            else:
                r, c = get_board_coords(p, pos, 0)
            
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                final_tensor[target_ch, r, c] += 0.25

    # --- Channel 7: Safe Zones ---
    SAFE_INDICES = [0, 8, 13, 21, 26, 34, 39, 47]
    for p in range(NUM_PLAYERS):
        for s in SAFE_INDICES:
             r, c = get_board_coords(p, s, 0)
             if r >= 0:
                 final_tensor[7, r, c] = 0.5

    # --- Channels 8-11: Home Paths (Binary) ---
    # We use precomputed masks. Masks are (4, 15, 15) global.
    # We need to map them relative to current player.
    # Ch 8: My Home Path
    # Ch 9: Next Home Path
    # Ch 10: Team Home Path
    # Ch 11: Prev Home Path
    # We can reconstruct them dynamically for safety using get_board_coords
    for p_offset in range(4):
        target_ch = 8 + p_offset
        p = (current_p + p_offset) % 4
        for i in range(5):
             r, c = get_board_coords(p, 51 + i, 0)
             if r >= 0:
                 final_tensor[target_ch, r, c] = 1.0

    # --- Apply Rotation to Spatial Channels (0-11) ---
    k = current_p
    if k > 0:
        final_tensor[:12] = np.rot90(final_tensor[:12], k=k, axes=(1, 2))

    # --- CHANNELS 12-17: Dice One-Hot ---
    roll = state.current_dice_roll
    if 1 <= roll <= 6:
        dice_ch = 12 + (roll - 1)
        final_tensor[dice_ch, :, :] = 1.0

    # --- BROADCAST STATS (Channels 18-20) ---
    
    # 18: Score Diff
    my_score = scores[current_p]
    max_opp = 0
    for p in range(4):
        if p != current_p:
            max_opp = max(max_opp, scores[p])
    score_val = (my_score - max_opp) / 4.0
    final_tensor[18, :, :] = score_val

    # 19: My Locked
    total_locked = 0
    for t in range(4):
        if positions[current_p][t] == BASE_POS:
            total_locked += 1
    final_tensor[19, :, :] = total_locked / 4.0

    # 20: Opp Locked
    total_opp_locked = 0
    for p in range(4):
        if p == current_p: continue
        for t in range(4):
            if positions[p][t] == BASE_POS:
                total_opp_locked += 1
    final_tensor[20, :, :] = total_opp_locked / 12.0
    
    return torch.from_numpy(final_tensor.copy())


