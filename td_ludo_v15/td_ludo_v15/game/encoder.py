"""V15 per-cell triplet encoder.

Produces `(15, 15, 3)` per-frame arrays and stacks 8 chronological frames
into `(8, 15, 15, 3)` for the model.

Triplet slot semantics (see ../../V15_DESIGN_PLAN.md for full spec):
    a = my-token count at this cell (0..4 if I can be here, -1 if I cannot)
    b = opp-token count (same semantics, -1 if opp cannot be here)
    c = safety/playability flag from my POV
        c = 1: cell is on my route AND safe
        c = 0: cell is on my route AND unsafe
        c = -1: cell is NOT on my route

Home base = Option B (spread-fill rule, per V15_DESIGN_PLAN.md):
    When N of my tokens are at home base, the FIRST N cells in canonical
    order get (1, -1, 1); the remaining 4-N cells get (0, -1, 1).
    Canonical order for my home base (P0 POV): (2,2) → (2,3) → (3,2) → (3,3).
    Canonical order for opp home base (P0 POV of P2's base):
        (12,12) → (12,11) → (11,12) → (11,11).
    This is token-id-blind (any permutation of token IDs at home produces
    the same encoding) AND preserves the visual stack the model can read.

Scored tokens → tracked via MS/OS scalar slots only (see overrides below).

Special-cell overrides (replace the above where they collide):
    MD = (0, 0):   (-1, -1, dice) if my turn, else (-1, -1, -1)
    OD = (14, 14): (-1, -1, dice) if opp's turn, else (-1, -1, -1)
    MS = (7, 6):   (my_scored, -1, -1)
    OS = (7, 8):   (opp_scored, -1, -1)

Game-start padding: frames are filled with all-zeros `(0, 0, 0)` per the
AlphaGo convention. Model learns to recognize padding from training data.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

import td_ludo_v15_cpp as _cpp
from .cells import (
    BOARD_SIZE,
    HOME_BASE_CELLS_P0,
    HOME_STRETCH_CELLS_P0,
    MD_CELL,
    MS_CELL,
    OD_CELL,
    OPP_HOME_BASE_CELLS,
    OPP_HOME_STRETCH_CELLS,
    OS_CELL,
    position_to_cell_in_pov,
    print_grid,
)


# Safe path-position indices on the 52-cell loop (canonical, same in any POV).
_SAFE_POSITIONS = frozenset({0, 8, 13, 21, 26, 34, 39, 47})

# Convenience: 5 canonical safe cells in P0's POV (the 8 safe positions
# include 0 (=(6,1)) plus 7 more; precomputed via cpp at module load).
_SAFE_PATH_CELLS_POV: List[tuple] = [
    (int(r), int(c))
    for (r, c) in (_cpp.position_to_cell(p, 0) for p in _SAFE_POSITIONS)
]


def _opp_player_id_2p(state, pov_player: int) -> int:
    """Returns the OTHER active player in 2P mode (i.e., `pov_player`'s
    opponent in the game, *not* the opponent of `state.current_player`).

    For 2P, active players are {0, 2}. If pov is 0 → opp is 2, else 0.

    The earlier version of this helper took only `state` and used
    `state.current_player` as the reference — which broke for historical
    frames where the opp was about to move: opp_player would resolve to
    pov_player, causing the encoder to plot own tokens in the opp grid
    and zero out OD dice. Verified by the input-dump diagnostic showing
    MD=-1 AND OD=-1 in half the temporal frames, which was the root cause
    of V15 SL plateauing at 47% bot-eval despite a working architecture.
    Bug + fix: 2026-05-14.
    """
    active = state.active_players
    for p in range(_cpp.NUM_PLAYERS):
        if p != pov_player and active[p]:
            return p
    raise RuntimeError("No active opponent — invalid state for 2P encoding.")


def encode_frame(state, pov_player: Optional[int] = None) -> np.ndarray:
    """Encode a single GameState into a (15, 15, 3) int8 array in
    `pov_player`'s POV.

    If `pov_player` is None, uses `state.current_player`. The encoder
    rotates the board so that pov_player's spawn appears at the canonical
    P0 spawn cell (6, 1).

    Returns int8 (values fit in [-1, 6]).
    """
    if pov_player is None:
        pov_player = int(state.current_player)
    opp_player = _opp_player_id_2p(state, pov_player)

    frame = np.full((BOARD_SIZE, BOARD_SIZE, 3), -1, dtype=np.int8)

    # ── 1. Mark shared path cells (a=0, b=0, c=safety) in POV ────────────
    # In pov_player's POV, the 51-cell main path is at the same P0-canonical
    # cells regardless of who's POV (own path mapping is identity).
    for path_pos in range(51):  # positions 0..50
        # Use the cpp helper directly with player=0 — these P0-canonical
        # cells ARE current-player POV cells for an own token at that position.
        r, c = _cpp.position_to_cell(path_pos, 0)
        is_safe = path_pos in _SAFE_POSITIONS
        frame[r, c, 0] = 0
        frame[r, c, 1] = 0
        frame[r, c, 2] = 1 if is_safe else 0

    # ── 2. My home base (4 cells, spread-fill below) ────────────────────
    # All 4 home cells exist on my route and are safe. Start with the empty
    # state (a=0, b=-1, c=1). Spread-fill in step 6 increments the FIRST N
    # cells when N tokens are home (Option B canonical-order rule).
    for (r, c) in HOME_BASE_CELLS_P0:
        frame[r, c] = (0, -1, 1)  # safe by definition (home base)

    # ── 3. My home stretch ──────────────────────────────────────────────
    for sp_idx, (r, c) in enumerate(HOME_STRETCH_CELLS_P0):
        frame[r, c] = (0, -1, 1)  # always safe (own territory)

    # ── 4. Opp home base (4 cells, spread-fill below) ───────────────────
    for (r, c) in OPP_HOME_BASE_CELLS:
        frame[r, c] = (-1, 0, -1)

    # ── 5. Opp home stretch ─────────────────────────────────────────────
    for (r, c) in OPP_HOME_STRETCH_CELLS:
        frame[r, c] = (-1, 0, -1)

    # ── 6. Count tokens ─────────────────────────────────────────────────
    # Home-base tokens use the SPREAD-FILL rule (Option B): N tokens at base
    # → first N cells in canonical order get a=1, the rest stay a=0. Path
    # tokens use the normal sum-into-source-cell rule.
    own_positions = list(state.player_positions[pov_player])
    opp_positions = list(state.player_positions[opp_player])

    own_at_base = sum(1 for p in own_positions if int(p) == _cpp.BASE_POS)
    opp_at_base = sum(1 for p in opp_positions if int(p) == _cpp.BASE_POS)

    # Spread-fill own home base: first own_at_base cells → a=1, rest stay 0.
    for i, (r, c) in enumerate(HOME_BASE_CELLS_P0):
        frame[r, c, 0] = 1 if i < own_at_base else 0

    # Spread-fill opp home base: first opp_at_base cells → b=1, rest stay 0.
    for i, (r, c) in enumerate(OPP_HOME_BASE_CELLS):
        frame[r, c, 1] = 1 if i < opp_at_base else 0

    # Path tokens (non-base, non-scored): sum into source cell.
    for pos in own_positions:
        pos = int(pos)
        if pos == _cpp.HOME_POS or pos == _cpp.BASE_POS:
            continue  # base handled by spread-fill above; scored via MS
        r, c = position_to_cell_in_pov(pos, pov_player, pov_player)
        cur = int(frame[r, c, 0])
        if cur < 0:
            cur = 0
        frame[r, c, 0] = cur + 1

    for pos in opp_positions:
        pos = int(pos)
        if pos == _cpp.HOME_POS or pos == _cpp.BASE_POS:
            continue
        r, c = position_to_cell_in_pov(pos, opp_player, pov_player)
        cur = int(frame[r, c, 1])
        if cur < 0:
            cur = 0
        frame[r, c, 1] = cur + 1

    # ── 7. Special-cell overrides ───────────────────────────────────────
    cp = int(state.current_player)
    dice = int(state.current_dice_roll)
    # MD slot
    md_r, md_c = MD_CELL
    if cp == pov_player and dice > 0:
        frame[md_r, md_c] = (-1, -1, dice)
    else:
        frame[md_r, md_c] = (-1, -1, -1)
    # OD slot
    od_r, od_c = OD_CELL
    if cp == opp_player and dice > 0:
        frame[od_r, od_c] = (-1, -1, dice)
    else:
        frame[od_r, od_c] = (-1, -1, -1)
    # MS slot — my scored count
    own_scored = int(state.scores[pov_player])
    ms_r, ms_c = MS_CELL
    frame[ms_r, ms_c] = (own_scored, -1, -1)
    # OS slot — opp scored count
    opp_scored = int(state.scores[opp_player])
    os_r, os_c = OS_CELL
    frame[os_r, os_c] = (opp_scored, -1, -1)

    return frame


def encode_history(
    frame_history: Sequence,
    pov_player: int,
) -> np.ndarray:
    """Stack 8 frames into a (8, 15, 15, 3) int8 array.

    `frame_history` is a length-8 sequence of GameState-or-None (zero-pad
    for None entries — the AlphaGo padding convention).

    `pov_player` is fixed across all frames in the stack: every past frame
    is re-encoded in the CURRENT pov_player's perspective, so the temporal
    stack tracks "what the board looked like from MY perspective at each
    moment." This matches how AlphaZero-chess encodes history.
    """
    if len(frame_history) != 8:
        raise ValueError(f"expected 8 frames, got {len(frame_history)}")
    out = np.zeros((8, BOARD_SIZE, BOARD_SIZE, 3), dtype=np.int8)
    for t, st in enumerate(frame_history):
        if st is None:
            # All-zero pad. Already zeros from np.zeros initialization.
            continue
        out[t] = encode_frame(st, pov_player=pov_player)
    return out


# ─── Visualization helpers ────────────────────────────────────────────────


def print_frame(frame: np.ndarray, title: str = "") -> str:
    """Render a (15, 15, 3) frame as three side-by-side 15×15 grids `a`/`b`/`c`.

    Returns a single multi-line string for printing or saving to golden files.
    """
    if frame.shape != (BOARD_SIZE, BOARD_SIZE, 3):
        raise ValueError(f"expected ({BOARD_SIZE}, {BOARD_SIZE}, 3), got {frame.shape}")
    a = frame[:, :, 0]
    b = frame[:, :, 1]
    c = frame[:, :, 2]
    parts = []
    if title:
        parts.append(title)
        parts.append("")
    parts.append(print_grid(a, title="slot a (own count)"))
    parts.append("")
    parts.append(print_grid(b, title="slot b (opp count)"))
    parts.append("")
    parts.append(print_grid(c, title="slot c (safety: 1=safe, 0=unsafe, -1=off-route)"))
    return "\n".join(parts)


def print_history(history_arr: np.ndarray, frames_to_show: int = 1) -> str:
    """Render the last `frames_to_show` frames of an `(8, 15, 15, 3)` stack.

    The current frame is `t=7`. Older frames have decreasing t.
    """
    if history_arr.shape != (8, BOARD_SIZE, BOARD_SIZE, 3):
        raise ValueError(
            f"expected (8, {BOARD_SIZE}, {BOARD_SIZE}, 3), got {history_arr.shape}"
        )
    parts = []
    for k in range(max(0, 8 - frames_to_show), 8):
        label = f"=== frame t={k}" + (" (current)" if k == 7 else f" ({k - 7})") + " ==="
        parts.append(print_frame(history_arr[k], title=label))
        parts.append("")
    return "\n".join(parts)
