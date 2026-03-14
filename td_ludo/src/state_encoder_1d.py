"""
V7 1D State Encoder — Converts GameState to compact 1D vector.

Replaces the 15×15×17 spatial tensor (tensor_utils.py / C++ encode_state)
with a lightweight 1D representation designed for transformer input.

1D State Vector (per turn):
  - 8 token positions: 4 self + 4 opponent (integers 0-58)
  - 3 global scalars: opp_locked_frac, my_locked_frac, score_diff
  - 6 dice one-hot: current dice roll
  - 1 historical action: last action taken (0-3 = token, 4 = pass/none)

Position encoding (0-58):
  0  = locked in base
  1-52 = main shared track (player-relative)
  53-57 = home stretch
  58 = home (scored)

Note: Positions are player-relative. The C++ GameState stores positions
as player-relative integers already (base=-1, path 0-50, home run 51-55, home=99).
We remap: base(-1)→0, path(0-50)→(1-51), home_run(51-55)→(53-57), home(99)→58.
"""

import numpy as np

# Position remapping from C++ GameState conventions
# C++: base=-1, path=0-50, home_run=51-55, home=99
# V7:  base=0,  path=1-51, skip 52, home_stretch=53-57, home=58

NUM_POSITION_CLASSES = 59  # 0-58 inclusive (for nn.Embedding)
NUM_ACTION_CLASSES = 5     # 0-3 = token move, 4 = pass/none
DICE_DIM = 6               # one-hot for dice 1-6
GLOBAL_DIM = 3             # opp_locked_frac, my_locked_frac, score_diff
NUM_TOKENS = 4
NUM_TOKEN_POSITIONS = 8    # 4 self + 4 opponent

# Total continuous features per turn (global scalars + dice one-hot)
CONTINUOUS_DIM = GLOBAL_DIM + DICE_DIM  # 9


def _remap_position(cpp_pos):
    """
    Remap C++ position to V7 1D position integer.

    C++ conventions:
      -1 = base (locked)
      0-50 = main path (51 squares)
      51-55 = home stretch (5 squares)
      99 = home (scored)

    V7 conventions:
      0 = base (locked)
      1-51 = main path
      53-57 = home stretch
      58 = home (scored)
    """
    if cpp_pos == -1:
        return 0       # base
    elif 0 <= cpp_pos <= 50:
        return cpp_pos + 1  # path: 0-50 → 1-51
    elif 51 <= cpp_pos <= 55:
        return cpp_pos + 2  # home stretch: 51-55 → 53-57
    elif cpp_pos == 99:
        return 58      # home
    else:
        return 0       # fallback to base


def encode_state_1d(game_state):
    """
    Convert a C++ GameState to a V7 1D state representation.

    Args:
        game_state: td_ludo_cpp GameState object

    Returns:
        token_positions: np.array of shape (8,) dtype int64
            [my_tok0, my_tok1, my_tok2, my_tok3, opp_tok0, opp_tok1, opp_tok2, opp_tok3]
        continuous: np.array of shape (9,) dtype float32
            [opp_locked_frac, my_locked_frac, score_diff, dice_onehot_1..6]
    """
    cp = game_state.current_player
    positions = game_state.player_positions
    scores = game_state.scores
    active_players = game_state.active_players
    dice = game_state.current_dice_roll

    # --- Token positions (8 integers) ---
    token_positions = np.zeros(8, dtype=np.int64)

    # My 4 tokens (indices 0-3)
    for t in range(4):
        token_positions[t] = _remap_position(positions[cp][t])

    # Opponent tokens (indices 4-7)
    # In 2-player mode, opponent is at seat (cp + 2) % 4
    # In 4-player mode, we use the first active opponent's tokens
    opp_found = False
    for p_offset in range(1, 4):
        p = (cp + p_offset) % 4
        if active_players[p]:
            for t in range(4):
                token_positions[4 + t] = _remap_position(positions[p][t])
            opp_found = True
            break

    if not opp_found:
        # No active opponent (shouldn't happen in practice)
        token_positions[4:] = 0

    # --- Global context (3 floats) ---
    # My locked count
    my_locked = sum(1 for t in range(4) if positions[cp][t] == -1)
    my_locked_frac = my_locked / 4.0

    # Opponent locked count
    total_opp_locked = 0
    active_opp_tokens = 0
    for p in range(4):
        if p == cp or not active_players[p]:
            continue
        active_opp_tokens += 4
        for t in range(4):
            if positions[p][t] == -1:
                total_opp_locked += 1
    opp_locked_frac = total_opp_locked / active_opp_tokens if active_opp_tokens > 0 else 0.0

    # Score difference (normalized)
    my_score = scores[cp]
    max_opp_score = 0
    for p in range(4):
        if p != cp:
            max_opp_score = max(max_opp_score, scores[p])
    score_diff = (my_score - max_opp_score) / 4.0

    # --- Dice one-hot (6 floats) ---
    dice_onehot = np.zeros(6, dtype=np.float32)
    if 1 <= dice <= 6:
        dice_onehot[dice - 1] = 1.0

    # --- Combine continuous features ---
    continuous = np.zeros(CONTINUOUS_DIM, dtype=np.float32)
    continuous[0] = opp_locked_frac
    continuous[1] = my_locked_frac
    continuous[2] = score_diff
    continuous[3:9] = dice_onehot

    return token_positions, continuous


def make_empty_state_1d():
    """
    Create a zero-padded 1D state (used for context window padding).

    Returns:
        token_positions: np.zeros(8, dtype=int64) — all tokens at base (position 0)
        continuous: np.zeros(9, dtype=float32) — all zeros
    """
    return np.zeros(8, dtype=np.int64), np.zeros(CONTINUOUS_DIM, dtype=np.float32)
