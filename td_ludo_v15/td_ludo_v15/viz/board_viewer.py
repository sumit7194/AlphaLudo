"""Visual side-by-side viewer for V15 — shows the game board + the
encoder's per-cell triplet output for the same state, aligned cell-by-cell.

Use case: user runs `python -m td_ludo_v15.scripts.dump_state --seed 42 --moves 35`,
inspects the output to confirm that the encoding matches what they see on
the board.

The board renderer shows a 15×15 ASCII grid with markers:
    `.`        — path cell, empty
    `O`        — current player's token
    `X`        — opponent's token
    digit      — count when multiple tokens stacked on same cell
    `H`        — my home counter (with count of locked tokens)
    `h`        — opp home counter
    `>`        — my home stretch
    `<`        — opp home stretch
    `*`        — safe path cell (overlay shown separately)
    `M`/`O`    — MD/OD slots (when active for that frame)
    `$`        — MS/OS slots (showing scored count)
    ` ` (space) — dead cells, never used

The encoding-triplet grids show slot a (own counts), b (opp counts), c
(safety/playability), each as a separate 15×15 grid.

Output: a multi-line string for printing or saving to golden files.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

import td_ludo_v15_cpp as _cpp
from ..game.cells import (
    BOARD_SIZE,
    HOME_BASE_COUNTER,
    HOME_STRETCH_CELLS_P0,
    MD_CELL,
    MS_CELL,
    OD_CELL,
    OPP_HOME_BASE_COUNTER,
    OPP_HOME_STRETCH_CELLS,
    OS_CELL,
    position_to_cell_in_pov,
)
from ..game.encoder import encode_frame


_SAFE_POSITIONS = frozenset({0, 8, 13, 21, 26, 34, 39, 47})


def render_board(state, pov_player: int) -> np.ndarray:
    """Build a 15×15 char-array board view in `pov_player`'s POV.

    Token-counts on a cell appear as digits 1..4 (or 'O'/'X' for single
    tokens). Special cells use letter markers.
    """
    grid = np.full((BOARD_SIZE, BOARD_SIZE), " ", dtype="<U2")

    # Determine opp
    cp = int(state.current_player)
    active = state.active_players
    opp = next(p for p in range(_cpp.NUM_PLAYERS) if p != pov_player and active[p])

    # 1. Mark all shared path cells with `.` (empty path)
    for p in range(51):
        r, c = _cpp.position_to_cell(p, 0)
        grid[r, c] = "."

    # 2. Mark safe-cell overlay (replaces `.` with `*` for safe positions)
    for p in _SAFE_POSITIONS:
        r, c = _cpp.position_to_cell(p, 0)
        grid[r, c] = "*"

    # 3. My home stretch
    for (r, c) in HOME_STRETCH_CELLS_P0:
        grid[r, c] = ">"
    # 4. Opp home stretch
    for (r, c) in OPP_HOME_STRETCH_CELLS:
        grid[r, c] = "<"

    # 5. Home counters
    r, c = HOME_BASE_COUNTER
    grid[r, c] = "H"
    r, c = OPP_HOME_BASE_COUNTER
    grid[r, c] = "h"

    # 6. Place own tokens
    own_counts: dict = {}
    for pos in state.player_positions[pov_player]:
        pos = int(pos)
        if pos == _cpp.HOME_POS:
            continue
        if pos == _cpp.BASE_POS:
            cell = HOME_BASE_COUNTER
        else:
            cell = position_to_cell_in_pov(pos, pov_player, pov_player)
        own_counts[cell] = own_counts.get(cell, 0) + 1

    # 7. Place opp tokens
    opp_counts: dict = {}
    for pos in state.player_positions[opp]:
        pos = int(pos)
        if pos == _cpp.HOME_POS:
            continue
        if pos == _cpp.BASE_POS:
            cell = OPP_HOME_BASE_COUNTER
        else:
            cell = position_to_cell_in_pov(pos, opp, pov_player)
        opp_counts[cell] = opp_counts.get(cell, 0) + 1

    for (r, c), n in own_counts.items():
        existing = grid[r, c]
        if existing == "H":
            # My home counter: show count + 'H' marker
            grid[r, c] = f"{n}H"
        else:
            grid[r, c] = "O" if n == 1 else f"{n}O"
    for (r, c), n in opp_counts.items():
        existing = grid[r, c]
        if existing == "h":
            # Opp home counter
            grid[r, c] = f"{n}h"
        elif existing.strip() in (".", "*", ">", "<"):
            grid[r, c] = "X" if n == 1 else f"{n}X"
        elif "O" in existing or "H" in existing:
            # Mixed: own + opp on same cell (very rare on safe path cells)
            grid[r, c] = "OX"
        else:
            grid[r, c] = "X" if n == 1 else f"{n}X"

    # 8. Mark special cells (MD/OD/MS/OS) — these override the above
    r, c = MD_CELL
    dice = int(state.current_dice_roll)
    if cp == pov_player and dice > 0:
        grid[r, c] = f"M{dice}"  # MD active
    else:
        grid[r, c] = "M-"
    r, c = OD_CELL
    if cp == opp and dice > 0:
        grid[r, c] = f"D{dice}"  # OD active
    else:
        grid[r, c] = "D-"
    r, c = MS_CELL
    grid[r, c] = f"${int(state.scores[pov_player])}"  # MS: my scored count
    r, c = OS_CELL
    grid[r, c] = f"${int(state.scores[opp])}"  # OS: opp scored count

    return grid


def render_board_text(state, pov_player: Optional[int] = None) -> str:
    """Render the board as a single multi-line string."""
    if pov_player is None:
        pov_player = int(state.current_player)
    grid = render_board(state, pov_player)
    lines = [f"Board (POV player={pov_player}):"]
    fmt = "{:>3}"
    header = "    " + "".join(fmt.format(c) for c in range(BOARD_SIZE))
    lines.append(header)
    for r in range(BOARD_SIZE):
        row_str = "".join(fmt.format(grid[r, c]) for c in range(BOARD_SIZE))
        lines.append(f"{r:>2}  {row_str}")
    return "\n".join(lines)


def render_triplet_grid(frame: np.ndarray, slot_idx: int, title: str = "") -> str:
    """Render one slot (a/b/c) of the (15,15,3) frame as an ASCII grid.

    Cells with value -1 display as `-`. Non-negative values display as digits.
    """
    if frame.shape != (BOARD_SIZE, BOARD_SIZE, 3):
        raise ValueError(f"expected (15, 15, 3), got {frame.shape}")
    slot = frame[:, :, slot_idx]
    lines = [title] if title else []
    fmt = "{:>3}"
    header = "    " + "".join(fmt.format(c) for c in range(BOARD_SIZE))
    lines.append(header)
    for r in range(BOARD_SIZE):
        row_repr = []
        for c in range(BOARD_SIZE):
            v = int(slot[r, c])
            row_repr.append(fmt.format("-" if v == -1 else str(v)))
        lines.append(f"{r:>2}  {''.join(row_repr)}")
    return "\n".join(lines)


def render_side_by_side(state, pov_player: Optional[int] = None) -> str:
    """Render the board AND its (15,15,3) encoding stacked vertically.

    Format:
        === HEADER ===
        Board (POV player=cp):
        [15x15 board with markers]

        Encoding (POV player=cp):
        slot a (own count):
        [15x15 grid of a values]

        slot b (opp count):
        [15x15 grid of b values]

        slot c (safety/playability):
        [15x15 grid of c values]
    """
    if pov_player is None:
        pov_player = int(state.current_player)
    frame = encode_frame(state, pov_player=pov_player)
    board = render_board_text(state, pov_player=pov_player)
    parts = [
        "=" * 60,
        f"Board state (current_player={int(state.current_player)}, "
        f"dice={int(state.current_dice_roll)}, "
        f"scores={list(state.scores)})",
        "=" * 60,
        "",
        "BOARD LEGEND:",
        "  . = empty path cell        * = safe path cell",
        "  O = own token              X = opp token       (digit-prefixed for stacks)",
        "  H = my home counter        h = opp home counter",
        "  > = my home stretch        < = opp home stretch",
        "  M{d} = my dice (d)         D{d} = opp dice (d)    M- / D- = inactive",
        "  $n = scored count (n)      (blank) = dead cell",
        "",
        board,
        "",
        "ENCODING LEGEND:",
        "  -  = -1 (inactive / off-route / no value)",
        "  digit = count or dice value, depending on slot",
        "",
        render_triplet_grid(frame, 0, title="slot a (own count):"),
        "",
        render_triplet_grid(frame, 1, title="slot b (opp count):"),
        "",
        render_triplet_grid(frame, 2, title="slot c (safety: 1=safe&on-route, 0=unsafe&on-route, -1=off-route):"),
        "",
    ]
    return "\n".join(parts)
