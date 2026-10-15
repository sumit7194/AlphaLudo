"""Board geometry for the unified (V17) encoding — pure Python, independent of Rust.

Mirrors ludo_core::constants / geometry so the Rust encoder can be tested against
an independent implementation. Coordinates are (row, col) on the 15x15 board.
"""
from __future__ import annotations

BOARD = 15
N_CELLS = BOARD * BOARD
N_NODES = N_CELLS + 1          # + CLS readout node
CLS_INDEX = N_CELLS
BASE_POS, HOME_POS = -1, 99
SAFE_ABS = frozenset({0, 8, 13, 21, 26, 34, 39, 47})

# Player 0's track positions 0..50 (main loop) and 51..55 (home run).
PATH_P0 = [
    (6, 1), (6, 2), (6, 3), (6, 4), (6, 5),
    (5, 6), (4, 6), (3, 6), (2, 6), (1, 6), (0, 6),
    (0, 7), (0, 8),
    (1, 8), (2, 8), (3, 8), (4, 8), (5, 8),
    (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), (6, 14),
    (7, 14), (8, 14),
    (8, 13), (8, 12), (8, 11), (8, 10), (8, 9),
    (9, 8), (10, 8), (11, 8), (12, 8), (13, 8), (14, 8),
    (14, 7), (14, 6),
    (13, 6), (12, 6), (11, 6), (10, 6), (9, 6),
    (8, 5), (8, 4), (8, 3), (8, 2), (8, 1), (8, 0),
    (7, 0),
]
HOME_RUN_P0 = [(7, 1), (7, 2), (7, 3), (7, 4), (7, 5)]
LOOP_CLOSER = (6, 0)            # the 52nd loop cell (player 0 never stands on it)

# Relative-seat tables in the mover's frame (k = 0 me, 1 next, 2 opposite, 3 previous).
BASE_CELLS = [
    [(2, 2), (2, 3), (3, 2), (3, 3)],
    [(2, 12), (3, 12), (2, 11), (3, 11)],
    [(12, 12), (12, 11), (11, 12), (11, 11)],
    [(12, 2), (11, 2), (12, 3), (11, 3)],
]
DICE_CELLS = [(0, 0), (0, 14), (14, 14), (14, 0)]
SCORED_CELLS = [(7, 6), (6, 7), (7, 8), (8, 7)]


def idx(r: int, c: int) -> int:
    return r * BOARD + c


def rot_cw(r: int, c: int, k: int) -> tuple[int, int]:
    for _ in range(k % 4):
        r, c = c, 14 - r
    return r, c


def rot_ccw(r: int, c: int, k: int) -> tuple[int, int]:
    for _ in range(k % 4):
        r, c = 14 - c, r
    return r, c


def p0_track_cell(pos: int) -> tuple[int, int]:
    """Player 0's cell for track position 0..55."""
    return PATH_P0[pos] if pos <= 50 else HOME_RUN_P0[pos - 51]


def track_cell(pos: int, owner: int, pov: int) -> int:
    """Cell index, in `pov`'s frame, of `owner`'s token at track position 0..55.
    Relative to the mover only the seat offset matters: rotate P0's cell by
    (owner - pov) quarter turns clockwise."""
    r, c = p0_track_cell(pos)
    r, c = rot_cw(r, c, owner)       # actual board
    r, c = rot_ccw(r, c, pov)        # mover's frame
    return idx(r, c)


def is_safe_rel(pos: int, owner: int) -> bool:
    """Is `owner`'s track position safe? (Home run is always safe.)"""
    if pos > 50:
        return True
    return ((pos + 13 * owner) % 52) in SAFE_ABS
