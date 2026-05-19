"""Per-token / per-player progress score for reward shaping + aux head training.

Maps each token's position to a scalar in [0, 1]:
    -1 (HOME)  → 0.0
    0..50 (board path) → ((pos+1)/56)^EXPONENT, gradually rising
    51..55 (home column) → continues the same curve, capping near 1.0
    99 (SCORED) → 1.0

EXPONENT > 1 makes later-stage progress weighted more heavily than early-
stage (matches the user's intuition: a token at pos 50 is "almost there",
worth way more than a token at pos 25 even though they're both half-way
in linear terms). Default exponent 2.5 gives a moderate curve:

    pos=0   → 0.0000
    pos=10  → 0.0124
    pos=25  → 0.131
    pos=40  → 0.504
    pos=50  → 0.797
    pos=55  → 1.000
    SCORED  → 1.000

Per-player total: T_p = Σ_{i in 0..3} S(pos_i). For 2-player games,
T_p ranges over [0.0, 4.0] (all 4 home → all 4 scored).

Used by:
  - Reward shaping in trainer_v10 (per-step shaping bonus from ΔS)
  - V135Symmetric / V135ProductionAdapter aux head (predict per-token S)
  - Game-state telemetry / debugging
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


# Position semantics
HOME_POS = -1
SCORED_POS = 99
BOARD_PATH_LEN = 51       # 0..50
HOME_COLUMN_LEN = 5        # 51..55
TOTAL_PROGRESS_STEPS = BOARD_PATH_LEN + HOME_COLUMN_LEN  # 56

# Exponent for the non-linear curve. >1 makes late progress more valuable.
DEFAULT_EXPONENT = 2.5


def progress_score(pos: int, exponent: float = DEFAULT_EXPONENT) -> float:
    """Scalar progress score for one token's position. Range [0, 1]."""
    if pos == HOME_POS:
        return 0.0
    if pos == SCORED_POS:
        return 1.0
    # Both board (0..50) and home column (51..55) use the same curve;
    # the home column naturally pushes toward 1.0 because (pos+1)/56 → 1.
    if 0 <= pos <= 55:
        return float(((pos + 1) / float(TOTAL_PROGRESS_STEPS)) ** exponent)
    # Defensive — unknown position
    return 0.0


def progress_scores_for_player(
    player_positions_row: Sequence[int],
    exponent: float = DEFAULT_EXPONENT,
) -> np.ndarray:
    """Per-token (length-4) array of S(pos) for one player's tokens."""
    arr = np.asarray(player_positions_row).flatten()
    out = np.zeros(4, dtype=np.float32)
    for i in range(4):
        out[i] = progress_score(int(arr[i]), exponent)
    return out


def total_progress_for_player(
    player_positions_row: Sequence[int],
    exponent: float = DEFAULT_EXPONENT,
) -> float:
    """Sum of per-token S(pos). Range [0, 4]."""
    return float(progress_scores_for_player(player_positions_row, exponent).sum())


def total_progress_for_state(state, player_id: int,
                              exponent: float = DEFAULT_EXPONENT) -> float:
    """Convenience: pull the player's row from a GameState and compute T_p."""
    row = state.player_positions[player_id]
    return total_progress_for_player(row, exponent)


__all__ = [
    "HOME_POS", "SCORED_POS", "TOTAL_PROGRESS_STEPS",
    "DEFAULT_EXPONENT",
    "progress_score",
    "progress_scores_for_player",
    "total_progress_for_player",
    "total_progress_for_state",
]
