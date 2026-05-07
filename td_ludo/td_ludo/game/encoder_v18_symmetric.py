"""V18 encoder — token-permutation-symmetric input for V13.5.

Motivation
----------
V13.x encoders use 4 separate channels for the 4 own tokens (and 4 more for
opp). But Ludo's rules treat the 4 tokens as fully interchangeable
(permutation-symmetric). The asymmetric encoding forces the model to:
  (a) waste capacity learning a symmetry it shouldn't need to learn;
  (b) cope with state-distribution drift caused by the model's own
      "preferred token-id ordering" patterns at training time (e.g.
      after capture-and-return events).

V18 collapses the 4-per-side token planes into a single per-side count
plane, plus broadcast scalar planes for at-home counts. All other
information channels (dice, statics) are unchanged from V17.

Layout (13 channels, 15×15 each):
    ch0       own_token_count_per_cell    (sum of V14 ch0..3)
    ch1       opp_token_count_per_cell    (sum of V14 ch4..7)
    ch2       own_at_home_count           broadcast scalar / 4 (so values are 0.0, 0.25, 0.5, 0.75, 1.0)
    ch3       opp_at_home_count           broadcast scalar / 4
    ch4..9    dice 6-one-hot              (V14 ch8..13, unchanged)
    ch10..12  V11 statics                 (safe / my-home-path / opp-home-path)

Properties:
  - PERMUTATION-INVARIANT under any permutation of the 4 own (or 4 opp)
    token IDs. Verified by unit tests.
  - Stacking is preserved (count > 1 cell encodes a stack).
  - Home-counts are exposed as scalar planes for easy "n at home" reasoning.

The companion module `td_ludo/game/rank_mapping.py` provides the
canonical-rank helpers needed for symmetric *output*.
"""
from __future__ import annotations

import numpy as np
import td_ludo_cpp as ludo_cpp


V14_CHANNELS = 14
V18_CHANNELS = 13

# Each token-ID has a dedicated home cell in V14_minimal's per-token planes:
#   own (canonical POV): token 0→(2,2), 1→(2,3), 2→(3,2), 3→(3,3)
#   opp                : token 0→(12,12), 1→(12,11), 2→(11,12), 3→(11,11)
# Summing ch0..3 (own) without zeroing these would leak token-ID info via
# *which* home cells are occupied (e.g. token 1 at home → cell (2,3) is hot,
# token 2 at home → cell (3,2) is hot). Token-ID symmetry requires us to
# zero out the entire home corner after summing — home occupancy is already
# captured by the broadcast scalar plane (ch2 for own, ch3 for opp).
_OWN_HOME_CELLS = [(2, 2), (2, 3), (3, 2), (3, 3)]
_OPP_HOME_CELLS = [(11, 11), (11, 12), (12, 11), (12, 12)]


def _zero_home_cells(plane: np.ndarray, cells) -> None:
    for (r, c) in cells:
        plane[r, c] = 0.0


def _build_static_cache() -> np.ndarray:
    """Extract the 3 static channels from V11 encoder (safe, my-home, opp-home).

    These are CONSTANT in the post-fix canonical 2P encoder, so we cache
    them once at import time and reuse. (Same trick as encoder_v17.)
    """
    g = ludo_cpp.create_initial_state_2p()
    g.current_player = 0
    g.current_dice_roll = 6
    enc = np.asarray(ludo_cpp.encode_state_v11(g), dtype=np.float32)
    return enc[5:8].copy()  # (3, 15, 15)


_STATIC_CACHE: np.ndarray = _build_static_cache()


def _count_at_home(player_positions_row: np.ndarray) -> int:
    """player_positions_row is (4,) int array; -1 = at home, 0+ = on board."""
    return int((player_positions_row == -1).sum())


def encode_state_v18_symmetric(state) -> np.ndarray:
    """Token-symmetric V18 encoding.

    Returns
    -------
    out : np.ndarray of shape (13, 15, 15), dtype float32
    """
    v14 = np.asarray(ludo_cpp.encode_state_v14_minimal(state), dtype=np.float32)
    # ch0..3: own tokens (one channel per token-id) — collapse by sum
    own_count = v14[0:4].sum(axis=0).copy()  # (15, 15) writeable
    # ch4..7: opp tokens — collapse by sum
    opp_count = v14[4:8].sum(axis=0).copy()
    # Zero out home cells: see _OWN_HOME_CELLS / _OPP_HOME_CELLS comment.
    # Home occupancy is captured by the broadcast scalars (ch2/ch3 below).
    _zero_home_cells(own_count, _OWN_HOME_CELLS)
    _zero_home_cells(opp_count, _OPP_HOME_CELLS)
    # ch8..13: dice 6-one-hot (unchanged)
    dice = v14[8:14]  # (6, 15, 15)

    # at-home counts (read directly from state, normalized to 0..1)
    pp = state.player_positions  # (4, 4) int
    cp = int(state.current_player)
    opp = (cp + 2) % 4  # 2P: opponent is the player 2 slots away
    own_home = _count_at_home(pp[cp]) / 4.0
    opp_home = _count_at_home(pp[opp]) / 4.0
    own_home_plane = np.full((15, 15), own_home, dtype=np.float32)
    opp_home_plane = np.full((15, 15), opp_home, dtype=np.float32)

    out = np.empty((V18_CHANNELS, 15, 15), dtype=np.float32)
    out[0] = own_count
    out[1] = opp_count
    out[2] = own_home_plane
    out[3] = opp_home_plane
    out[4:10] = dice
    out[10:13] = _STATIC_CACHE
    return out


# Compatibility alias
encode_state_v18 = encode_state_v18_symmetric


__all__ = ["encode_state_v18_symmetric", "encode_state_v18", "V18_CHANNELS"]
