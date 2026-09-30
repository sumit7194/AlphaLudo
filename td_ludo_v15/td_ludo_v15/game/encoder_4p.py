"""V15 4-Player per-cell quintuplet encoder.

Produces `(15, 15, 5)` per-frame arrays in current mover's POV (P0).
Quintuplet slot semantics:
    slot 0 = my token count (0..4 if I can be here, -1 if I cannot)
    slot 1 = opp_next token count (P1, moves next in cycle: (cp + 1) % 4)
    slot 2 = opp_opp token count (P2, opposite seat: (cp + 2) % 4)
    slot 3 = opp_prev token count (P3, moved before me: (cp + 3) % 4)
    slot 4 = safety/playability flag from my POV:
        1: cell is on my route AND safe
        0: cell is on my route AND unsafe
       -1: cell is NOT on my route

Home base spread-fill (Option B):
    When N tokens of player i are in base, the first N cells of base i
    get slot i = 1, and remaining cells get slot i = 0. Other players get -1.

Special-cell overrides:
    4 Dice Corners:
        (0, 0)   P0 (me):   (-1, -1, -1, -1, dice) if P0's turn, else -1
        (0, 14)  P1 (next): (-1, -1, -1, -1, dice) if P1's turn, else -1
        (14, 14) P2 (opp):  (-1, -1, -1, -1, dice) if P2's turn, else -1
        (14, 0)  P3 (prev): (-1, -1, -1, -1, dice) if P3's turn, else -1
    4 Scored Centers:
        (7, 6) P0: (score_0, -1, -1, -1, -1)
        (6, 7) P1: (-1, score_1, -1, -1, -1)
        (7, 8) P2: (-1, -1, score_2, -1, -1)
        (8, 7) P3: (-1, -1, -1, score_3, -1)
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

import td_ludo_v15_cpp as _cpp
from .cells import (
    BOARD_SIZE,
    DICE_CELLS_4P,
    HOME_BASE_CELLS_4P,
    HOME_STRETCH_CELLS_4P,
    SCORED_CELLS_4P,
    position_to_cell_in_pov,
)

# Safe path-position indices on the 52-cell loop (canonical, same in any POV).
_SAFE_POSITIONS = frozenset({0, 8, 13, 21, 26, 34, 39, 47})


def encode_frame_4p(state, pov_player: Optional[int] = None) -> np.ndarray:
    """Encode a single 4-player GameState into a (15, 15, 5) int8 array in
    `pov_player`'s POV.

    If `pov_player` is None, uses `state.current_player`. The encoder
    rotates the board so that pov_player appears at the canonical P0 spawn (6, 1).
    """
    if pov_player is None:
        pov_player = int(state.current_player)

    # 4 players in relative turn order starting from pov_player
    rel_players = [(pov_player + i) % 4 for i in range(4)]

    # Initialize all cells with -1 sentinel
    frame = np.full((BOARD_SIZE, BOARD_SIZE, 5), -1, dtype=np.int8)

    # ── 1. Mark shared main path cells (52 cells of the outer loop in P0 POV) ──
    for path_pos in range(51):
        r, c = _cpp.position_to_cell(path_pos, 0)
        is_safe = path_pos in _SAFE_POSITIONS
        frame[r, c, 0] = 0
        frame[r, c, 1] = 0
        frame[r, c, 2] = 0
        frame[r, c, 3] = 0
        frame[r, c, 4] = 1 if is_safe else 0

    # Cell (6, 0) completes the 52-cell outer loop (opponents pass through here, P0 turns into stretch)
    frame[6, 0, 0] = -1  # P0 never visits (turns into home stretch at (7, 0))
    frame[6, 0, 1] = 0   # P1 can visit
    frame[6, 0, 2] = 0   # P2 can visit
    frame[6, 0, 3] = 0   # P3 can visit
    frame[6, 0, 4] = -1  # not on P0's route

    # ── 2. Mark home bases for all 4 players ──────────────────────────────
    for rel_idx in range(4):
        for (r, c) in HOME_BASE_CELLS_4P[rel_idx]:
            # Initial empty base: player rel_idx gets 0, others -1
            # safety is 1 only for P0's own base
            base_tuple = [-1, -1, -1, -1, 1 if rel_idx == 0 else -1]
            base_tuple[rel_idx] = 0
            frame[r, c] = tuple(base_tuple)

    # ── 3. Mark home stretches for all 4 players ──────────────────────────
    for rel_idx in range(4):
        for (r, c) in HOME_STRETCH_CELLS_4P[rel_idx]:
            stretch_tuple = [-1, -1, -1, -1, 1 if rel_idx == 0 else -1]
            stretch_tuple[rel_idx] = 0
            frame[r, c] = tuple(stretch_tuple)

    # ── 4. Count tokens and apply spread-fill ─────────────────────────────
    for rel_idx in range(4):
        p_actual = rel_players[rel_idx]
        positions = list(state.player_positions[p_actual])

        # Home base spread-fill
        at_base = sum(1 for pos in positions if int(pos) == _cpp.BASE_POS)
        for cell_idx, (r, c) in enumerate(HOME_BASE_CELLS_4P[rel_idx]):
            frame[r, c, rel_idx] = 1 if cell_idx < at_base else 0

        # Non-base, non-home tokens: sum into board cell in pov_player's POV
        for pos in positions:
            pos = int(pos)
            if pos == _cpp.HOME_POS or pos == _cpp.BASE_POS:
                continue
            r, c = position_to_cell_in_pov(pos, p_actual, pov_player)
            cur = int(frame[r, c, rel_idx])
            if cur < 0:
                cur = 0
            frame[r, c, rel_idx] = cur + 1

    # ── 5. Special-cell overrides: Dice corners & Scored centers ──────────
    cp = int(state.current_player)
    dice = int(state.current_dice_roll)

    # 4 Dice Corners
    for rel_idx in range(4):
        p_actual = rel_players[rel_idx]
        r, c = DICE_CELLS_4P[rel_idx]
        if cp == p_actual and dice > 0:
            frame[r, c] = (-1, -1, -1, -1, dice)
        else:
            frame[r, c] = (-1, -1, -1, -1, -1)

    # 4 Scored Centers
    for rel_idx in range(4):
        p_actual = rel_players[rel_idx]
        score_val = int(state.scores[p_actual])
        r, c = SCORED_CELLS_4P[rel_idx]
        scored_tuple = [-1, -1, -1, -1, -1]
        scored_tuple[rel_idx] = score_val
        frame[r, c] = tuple(scored_tuple)

    return frame
