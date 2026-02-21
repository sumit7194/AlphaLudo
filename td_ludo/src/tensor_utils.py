"""
TD-Ludo Tensor Utilities

Converts game states to neural network input tensors.
Self-contained — all constants and utilities are inlined here.
"""

import torch
import numpy as np
import td_ludo_cpp as ludo_cpp

# =============================================================================
# Board Constants
# =============================================================================
BOARD_SIZE = 15
NUM_PLAYERS = 4
NUM_TOKENS = 4
HOME_POS = 99
BASE_POS = -1

# =============================================================================
# Coordinate Lookup Tables (Mapped from C++ implementation)
# =============================================================================
# P0-centric path coordinates (51 squares: indices 0-50)
PATH_COORDS_P0 = [
    (6, 1), (6, 2), (6, 3), (6, 4), (6, 5),           # 0-4
    (5, 6), (4, 6), (3, 6), (2, 6), (1, 6), (0, 6),   # 5-10
    (0, 7), (0, 8),                                     # 11-12
    (1, 8), (2, 8), (3, 8), (4, 8), (5, 8),             # 13-17
    (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), (6, 14),# 18-23
    (7, 14), (8, 14),                                    # 24-25
    (8, 13), (8, 12), (8, 11), (8, 10), (8, 9),         # 26-30
    (9, 8), (10, 8), (11, 8), (12, 8), (13, 8), (14, 8),# 31-36
    (14, 7), (14, 6),                                    # 37-38
    (13, 6), (12, 6), (11, 6), (10, 6), (9, 6),         # 39-43
    (8, 5), (8, 4), (8, 3), (8, 2), (8, 1), (8, 0),     # 44-49
    (7, 0)                                               # 50
]

HOME_RUN_P0 = [
    (7, 1), (7, 2), (7, 3), (7, 4), (7, 5)
]

HOME_COORD_P0 = (7, 6)

BASE_COORDS = [
    [(2, 2), (2, 3), (3, 2), (3, 3)],          # P0 (Top Left) - Red
    [(2, 11), (2, 12), (3, 11), (3, 12)],       # P1 (Top Right) - Green
    [(11, 11), (11, 12), (12, 11), (12, 12)],   # P2 (Bottom Right) - Yellow
    [(11, 2), (11, 3), (12, 2), (12, 3)]        # P3 (Bottom Left) - Blue
]

SAFE_INDICES = {0, 8, 13, 21, 26, 34, 39, 47}


# =============================================================================
# Coordinate Mapping
# =============================================================================
def get_board_coords(player, pos, token_idx=0):
    """Maps a player's token position to (row, col) on the 15x15 board."""
    if pos == BASE_POS:
        return BASE_COORDS[player][token_idx]
    
    local_r, local_c = 0, 0
    
    if pos == HOME_POS:
        local_r, local_c = HOME_COORD_P0
    elif pos > 50:
        idx = pos - 51
        if 0 <= idx < 5:
            local_r, local_c = HOME_RUN_P0[idx]
        else:
            local_r, local_c = HOME_COORD_P0
    elif 0 <= pos < 51:
        local_r, local_c = PATH_COORDS_P0[pos]
    else:
        return -1, -1

    # Rotate based on player (90° clockwise per player around center (7,7))
    r, c = local_r, local_c
    for _ in range(player):
        r, c = c, 14 - r
        
    return r, c


# =============================================================================
# Precomputed Masks
# =============================================================================
def get_safe_mask():
    """Generates the constant safe zone mask."""
    mask = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for p in range(NUM_PLAYERS):
        for idx in SAFE_INDICES:
            r, c = get_board_coords(p, idx)
            mask[r, c] = 1.0
    return mask

def get_home_path_mask():
    """Generates the constant home path mask."""
    mask = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for p in range(NUM_PLAYERS):
        for i in range(5):
            r, c = get_board_coords(p, 52 + i)
            mask[r, c] = 1.0
        r, c = get_board_coords(p, HOME_POS)
        mask[r, c] = 1.0
    return mask

def get_home_run_masks():
    """Generates 4 separate masks for the Home Run paths of P0-P3."""
    masks = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for p in range(NUM_PLAYERS):
        for i in range(5):
            r, c = get_board_coords(p, 51 + i)
            if r >= 0 and c >= 0:
                masks[p, r, c] = 1.0
    return masks

SAFE_MASK = torch.from_numpy(get_safe_mask())
HOME_PATH_MASK = torch.from_numpy(get_home_path_mask())
HOME_RUN_MASKS = torch.from_numpy(get_home_run_masks())


# =============================================================================
# State to Tensor Conversion (11-channel, cleaned 2P afterstate)
# =============================================================================
def state_to_tensor_mastery(state):
    """
    Converts GameState to (11, 15, 15) single spatial tensor.
    
    Channels:
    0-3:   My Tokens (Distinct Identity, one channel per token)
    4:     Opponent Pieces (Density, 0.25 per token, skip inactive)
    5:     Safe Zones (Binary)
    6:     My Home Path (Binary)
    7:     Opponent Home Path (Binary, skip inactive)
    8:     Score Diff (Broadcast)
    9:     My Locked (Broadcast)
    10:    Opp Locked (Broadcast, skip inactive)
    """
    current_p = state.current_player
    positions = state.player_positions
    scores = state.scores
    active_players = state.active_players
    
    final_tensor = np.zeros((11, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    
    # --- CHANNELS 0-3: My Tokens (Distinct Identity) ---
    for t in range(4):
        pos = positions[current_p][t]
        if pos == BASE_POS:
            r, c = get_board_coords(current_p, pos, t)
        else:
            r, c = get_board_coords(current_p, pos, 0)
        
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            final_tensor[t, r, c] = 1.0

    # --- CHANNEL 4: Opponent Pieces (Single Density Channel) ---
    for p_offset in range(1, 4):
        p = (current_p + p_offset) % 4
        if not active_players[p]:
            continue
        for t in range(4):
            pos = positions[p][t]
            if pos == BASE_POS:
                r, c = get_board_coords(p, pos, t)
            else:
                r, c = get_board_coords(p, pos, 0)
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                final_tensor[4, r, c] += 0.25

    # --- CHANNEL 5: Safe Zones ---
    for p in range(4):
        for idx in SAFE_INDICES:
            r, c = get_board_coords(p, idx)
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                final_tensor[5, r, c] = 0.5

    # --- CHANNEL 6: My Home Path ---
    for i in range(5):
        r, c = get_board_coords(current_p, 51 + i, 0)
        if r >= 0:
            final_tensor[6, r, c] = 1.0

    # --- CHANNEL 7: Opponent Home Path (skip inactive) ---
    for p_offset in range(1, 4):
        p = (current_p + p_offset) % 4
        if not active_players[p]:
            continue
        for i in range(5):
            r, c = get_board_coords(p, 51 + i, 0)
            if r >= 0:
                final_tensor[7, r, c] = 1.0

    # --- Apply Rotation to Spatial Channels (0-7) ---
    k = current_p
    if k > 0:
        final_tensor[:8] = np.rot90(final_tensor[:8], k=k, axes=(1, 2))

    # --- BROADCAST STATS (Channels 8-10) ---
    
    # 8: Score Diff
    my_score = scores[current_p]
    max_opp = 0
    for p in range(4):
        if p != current_p:
            max_opp = max(max_opp, scores[p])
    score_val = (my_score - max_opp) / 4.0
    final_tensor[8, :, :] = score_val

    # 9: My Locked
    total_locked = 0
    for t in range(4):
        if positions[current_p][t] == BASE_POS:
            total_locked += 1
    final_tensor[9, :, :] = total_locked / 4.0

    # 10: Opp Locked (skip inactive)
    total_opp_locked = 0
    active_opp_tokens = 0
    for p in range(4):
        if p == current_p:
            continue
        if not active_players[p]:
            continue
        active_opp_tokens += 4
        for t in range(4):
            if positions[p][t] == BASE_POS:
                total_opp_locked += 1
    opp_locked_val = total_opp_locked / active_opp_tokens if active_opp_tokens > 0 else 0.0
    final_tensor[10, :, :] = opp_locked_val
    
    return final_tensor
