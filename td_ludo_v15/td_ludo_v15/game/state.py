"""Python wrapper around the V15 GameState that maintains the 8-frame
chronological history buffer the model expects as input.

Usage
-----
    from td_ludo_v15.game.state import V15GameWrapper

    g = V15GameWrapper.new_2p()
    g.set_dice(6)
    if g.dice == 0:
        # forfeit happened atomically
        continue
    cells = g.get_legal_source_cells()
    g.apply_move_from_cell(row, col)
    # history is updated automatically; encoder reads `g.frame_history`

The wrapper holds a 7-deep history (past frames) plus the current state =
8-frame chronological stack the encoder expects. Frames are captured at
each pre-decision moment (after dice is rolled, before move is applied),
mirroring the AlphaZero "pre-move state" convention.
"""
from __future__ import annotations

import collections
from typing import List, Tuple

import td_ludo_v15_cpp as _cpp

HISTORY_LEN = 7  # past frames; current is the 8th
TOTAL_FRAMES = 8


class V15GameWrapper:
    """Holds a V15 GameState plus a rolling 7-frame past-state buffer.

    The buffer is updated automatically on every successful move. To get
    the encoder's input, call `frame_history()` which returns the
    chronologically-ordered list `[t-7, t-6, ..., t-1, current]`
    (zero-padded with None at the start of the game).
    """

    def __init__(self, state: _cpp.GameState):
        self._state = state
        self._history: collections.deque[_cpp.GameState] = collections.deque(
            maxlen=HISTORY_LEN
        )

    # ─── Constructors ─────────────────────────────────────────────────────
    @classmethod
    def new_2p(cls) -> "V15GameWrapper":
        return cls(_cpp.create_initial_state_2p())

    @classmethod
    def new_4p(cls) -> "V15GameWrapper":
        return cls(_cpp.create_initial_state())

    # ─── State accessors ──────────────────────────────────────────────────
    @property
    def state(self) -> _cpp.GameState:
        return self._state

    @property
    def current_player(self) -> int:
        return int(self._state.current_player)

    @property
    def dice(self) -> int:
        return int(self._state.current_dice_roll)

    @property
    def is_terminal(self) -> bool:
        return bool(self._state.is_terminal)

    @property
    def winner(self) -> int:
        return int(_cpp.get_winner(self._state))

    # ─── Game ops (mutating) ──────────────────────────────────────────────
    def set_dice(self, dice_value: int) -> None:
        """Set dice. If 3rd consecutive 6 for current player, forfeit happens
        atomically: turn passes, dice becomes 0. Caller should check `.dice`.
        """
        self._state = _cpp.set_dice(self._state, int(dice_value))

    def pass_turn(self) -> None:
        """Pass turn to next active player (use when no legal moves)."""
        self._state = _cpp.pass_turn(self._state)

    def apply_move_from_cell(self, row: int, col: int) -> None:
        """Apply move with source cell (row, col). Captures the pre-move
        state into the history buffer BEFORE the move resolves — this is
        the AlphaZero convention (frame = state at decision time).
        """
        # Snapshot the pre-decision state into history first.
        self._history.append(self._state)
        # Then apply
        self._state = _cpp.apply_move_from_cell(self._state, int(row), int(col))

    # ─── Queries (delegated to cpp) ───────────────────────────────────────
    def get_legal_source_cells(self) -> List[Tuple[int, int]]:
        return [(int(r), int(c)) for r, c in _cpp.get_legal_source_cells(self._state)]

    def get_own_positions(self) -> List[int]:
        return [int(p) for p in _cpp.get_own_positions(self._state)]

    def get_opp_positions(self, opp_player: int) -> List[int]:
        return [int(p) for p in _cpp.get_opp_positions(self._state, int(opp_player))]

    # ─── Frame history (for the encoder) ──────────────────────────────────
    def frame_history(self) -> List[_cpp.GameState | None]:
        """Returns a length-8 list `[t-7, t-6, ..., t-1, current]`.

        Leading slots are `None` if fewer than 7 past frames are available.
        The encoder zero-pads those.
        """
        past: List[_cpp.GameState | None] = list(self._history)
        # Pad to HISTORY_LEN on the left with None
        pad = HISTORY_LEN - len(past)
        if pad > 0:
            past = [None] * pad + past
        return past + [self._state]
