"""Cell ↔ index utilities and special-cell constants for V15.

The V15 graph has 225 board cells (15×15) indexed as `idx = row * 15 + col`.
A CLS readout node is the 226th node (idx=225).

Special cells per the V15 design (see ../V15_DESIGN_PLAN.md):
    MD = (0, 0)    — current player's dice this frame
    OD = (14, 14)  — opponent's dice this frame
    MS = (7, 6)    — current player's scored-token count
    OS = (7, 8)    — opponent's scored-token count

Home-counter cells per player POV (canonical = P0 view; encoder always
operates in current-player POV after rotation):
    My home base counter:  (2, 2)
    Opp home base counter: (11, 11)   ← in P0's POV; opp is P2 in 2P mode
    My home stretch (5 cells):  (7, 1)..(7, 5)
    Opp home stretch (5 cells): (7, 13)..(7, 9)  in P0's POV
    Center (HOME):  (7, 6)  ← same cell as MS by coincidence; MS slot is
                              the scored counter, "HOME" is the conceptual
                              endpoint. In encoding, MS encodes
                              `(scored_count, -1, -1)` so it's distinguishable
                              from any path cell by sign pattern.

All path-position-to-cell conversions go through `td_ludo_v15_cpp.position_to_cell`.
This module only provides cell-side utilities + invariants.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

import td_ludo_v15_cpp as _cpp

# ─── Grid dimensions ──────────────────────────────────────────────────────
BOARD_SIZE: int = _cpp.BOARD_SIZE  # 15
NUM_BOARD_CELLS: int = BOARD_SIZE * BOARD_SIZE  # 225
CLS_INDEX: int = NUM_BOARD_CELLS  # 225 (the readout node)
NUM_NODES: int = NUM_BOARD_CELLS + 1  # 226 (board + CLS)

# ─── Cell ↔ linear index (row-major) ──────────────────────────────────────
def cell_to_index(row: int, col: int) -> int:
    """(row, col) → 0..224. Raises if out of bounds."""
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise ValueError(f"cell ({row}, {col}) out of bounds")
    return row * BOARD_SIZE + col


def index_to_cell(idx: int) -> Tuple[int, int]:
    """0..224 → (row, col). Raises if out of bounds."""
    if not (0 <= idx < NUM_BOARD_CELLS):
        raise ValueError(f"index {idx} out of board cell range [0, {NUM_BOARD_CELLS})")
    return divmod(idx, BOARD_SIZE)


# Precomputed lookup tables (small, built once at import).
CELL_TO_INDEX: np.ndarray = np.arange(NUM_BOARD_CELLS, dtype=np.int32).reshape(
    BOARD_SIZE, BOARD_SIZE
)
INDEX_TO_CELL: np.ndarray = np.array(
    [[i // BOARD_SIZE, i % BOARD_SIZE] for i in range(NUM_BOARD_CELLS)],
    dtype=np.int32,
)

# ─── Special-cell constants (encoder + graph use these) ───────────────────
MD_CELL: Tuple[int, int] = (0, 0)     # My Dice
OD_CELL: Tuple[int, int] = (14, 14)   # Opp Dice
MS_CELL: Tuple[int, int] = (7, 6)     # My Scored count (also the HOME center)
OS_CELL: Tuple[int, int] = (7, 8)     # Opp Scored count

SPECIAL_CELLS = (MD_CELL, OD_CELL, MS_CELL, OS_CELL)


def is_special_cell(row: int, col: int) -> bool:
    """True iff (row, col) is one of the 4 global-state special cells."""
    return (row, col) in SPECIAL_CELLS


# ─── Home base cells (4 cells per player, P0-canonical positions) ─────────
# In P0's POV, P0's home base is the top-left 2×2. Under V15's design
# Option B (spread-fill rule, see V15_DESIGN_PLAN.md):
# When N tokens are at home base, the first N cells in canonical order
# encode (1, -1, 1); the remaining 4-N cells encode (0, -1, 1).
# This preserves the visual stack while staying token-id-symmetric.
HOME_BASE_CELLS_P0 = ((2, 2), (2, 3), (3, 2), (3, 3))
HOME_BASE_COUNTER: Tuple[int, int] = (2, 2)  # legacy name; first cell in spread order

# Opp home base in current-player POV. Canonical CCW-rotation order from
# P0's POV → P2's perspective: (12, 12) → (12, 11) → (11, 12) → (11, 11).
# Same spread-fill applied symmetrically.
OPP_HOME_BASE_CELLS = ((12, 12), (12, 11), (11, 12), (11, 11))
OPP_HOME_BASE_COUNTER: Tuple[int, int] = (12, 12)  # legacy name; first cell in spread order

# ─── Home stretch cells ───────────────────────────────────────────────────
# In P0's POV: my stretch runs (7, 1) → (7, 5), then HOME at (7, 6).
HOME_STRETCH_CELLS_P0 = ((7, 1), (7, 2), (7, 3), (7, 4), (7, 5))
HOME_CENTER: Tuple[int, int] = (7, 6)  # same physical cell as MS

# Opp home stretch in P0's POV (P2's stretch after rotation): (7, 13) → (7, 9).
OPP_HOME_STRETCH_CELLS = ((7, 13), (7, 12), (7, 11), (7, 10), (7, 9))


# ─── Rotations + path-position-to-cell ────────────────────────────────────
def rotate_cell_ccw(row: int, col: int, k: int) -> Tuple[int, int]:
    """Rotate (row, col) k times 90° CCW around the board center (7, 7).

    One CCW rotation: `(r, c) -> (14 - c, r)`.
    """
    for _ in range(k % 4):
        row, col = 14 - col, row
    return row, col


def rotate_cell_cw(row: int, col: int, k: int) -> Tuple[int, int]:
    """Rotate (row, col) k times 90° CW around the board center (7, 7).

    One CW rotation: `(r, c) -> (c, 14 - r)`.
    """
    for _ in range(k % 4):
        row, col = col, 14 - row
    return row, col


def position_to_cell(pos: int, player: int = 0) -> Tuple[int, int]:
    """Returns the ACTUAL board cell where `player`'s token at `pos` lives.

    This wraps the cpp helper. For player=0, returns P0-canonical cells
    directly. For player>0, the cpp helper rotates P0's canonical cells CW
    by `player` to get the actual board cell.

    To get a token's cell in the current player's POV (the encoder's frame
    of reference), use `position_to_cell_in_pov` below.
    """
    r, c = _cpp.position_to_cell(pos, player)
    return int(r), int(c)


def position_to_cell_in_pov(
    pos: int, token_owner: int, pov_player: int
) -> Tuple[int, int]:
    """Returns the cell of `token_owner`'s token at `pos`, as seen in
    `pov_player`'s perspective (i.e., the board rotated so pov_player
    appears as P0).

    Concretely: get the ACTUAL board cell (cpp's position_to_cell), then
    rotate CCW by `pov_player` to translate into pov_player's frame.

    Special case for `pos == BASE_POS`: the cpp helper returns the
    canonical home counter cell (2, 2) for that owner (in their own POV
    via the BASE_COORDS table), which is then rotated CW by `token_owner`
    in cpp. Re-rotating CCW by pov_player gives:
        actual = rotate_CW^token_owner ( (2,2) )
        pov    = rotate_CCW^pov_player ( actual )
               = rotate_CW^(token_owner - pov_player) ( (2,2) )

    So when token_owner == pov_player, pov = (2,2) regardless of who's
    playing. The home counter is at (2,2) in current player's POV. ✓
    """
    actual_r, actual_c = position_to_cell(pos, token_owner)
    return rotate_cell_ccw(actual_r, actual_c, pov_player)


# ─── Grid printer (general purpose; used by encoder.print_frame) ──────────
def print_grid(grid: np.ndarray, title: str = "", cell_width: int = 3) -> str:
    """Render a (15, 15) ndarray as a column-aligned ASCII grid string.

    Used by both the encoder's print_frame helper and the board viewer.
    """
    if grid.shape != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError(f"expected ({BOARD_SIZE}, {BOARD_SIZE}), got {grid.shape}")
    fmt = "{:>" + str(cell_width) + "}"
    lines = []
    if title:
        lines.append(title)
    # Column header
    header = "    " + "".join(fmt.format(c) for c in range(BOARD_SIZE))
    lines.append(header)
    for r in range(BOARD_SIZE):
        row_repr = "".join(fmt.format(int(v)) for v in grid[r])
        lines.append(f"{r:>2}  {row_repr}")
    return "\n".join(lines)
