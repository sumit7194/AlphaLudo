"""Unit tests for 4-player 5-feature encoder and graph topology."""
from __future__ import annotations

import numpy as np
import pytest

import td_ludo_v15_cpp as _cpp
from td_ludo_v15.game.cells import (
    BOARD_SIZE,
    DICE_CELLS_4P,
    HOME_BASE_CELLS_4P,
    HOME_STRETCH_CELLS_4P,
    SCORED_CELLS_4P,
)
from td_ludo_v15.game.encoder_4p import encode_frame_4p
from td_ludo_v15.game.graph_4p import (
    EDGE_TYPE_MATRIX_4P,
    EDGES_4P,
    EdgeType,
    NUM_EDGE_TYPES,
)


def test_initial_state_encoding_shape_and_types():
    state = _cpp.create_initial_state()
    frame = encode_frame_4p(state, pov_player=0)

    assert frame.shape == (BOARD_SIZE, BOARD_SIZE, 5)
    assert frame.dtype == np.int8

    # All 4 players start with 4 tokens in base
    for rel_idx in range(4):
        for r, c in HOME_BASE_CELLS_4P[rel_idx]:
            # The player whose base this is has 1 (spread-fill for all 4 tokens)
            assert frame[r, c, rel_idx] == 1
            # Other players cannot be in this base
            for other_idx in range(4):
                if other_idx != rel_idx:
                    assert frame[r, c, other_idx] == -1

    # In initial state, scores are 0
    for rel_idx in range(4):
        r, c = SCORED_CELLS_4P[rel_idx]
        assert frame[r, c, rel_idx] == 0
        assert frame[r, c, 4] == -1  # not a regular route cell

    # Dice roll is 0 initially -> all dice cells are -1
    for rel_idx in range(4):
        r, c = DICE_CELLS_4P[rel_idx]
        assert (frame[r, c] == -1).all()


def test_dice_corner_placement():
    state = _cpp.create_initial_state()
    # Advance to Player 1's turn with dice 6
    state = _cpp.pass_turn(state)
    assert state.current_player == 1
    state = _cpp.set_dice(state, 6)

    # Encode in Player 0 POV: Player 1 is next (rel_idx = 1)
    frame_p0 = encode_frame_4p(state, pov_player=0)
    p1_dice_cell = DICE_CELLS_4P[1]  # (0, 14)
    assert frame_p0[p1_dice_cell[0], p1_dice_cell[1], 4] == 6

    # Encode in Player 1 POV: Player 1 is mover (rel_idx = 0)
    frame_p1 = encode_frame_4p(state, pov_player=1)
    p0_dice_cell = DICE_CELLS_4P[0]  # (0, 0)
    assert frame_p1[p0_dice_cell[0], p0_dice_cell[1], 4] == 6


def test_rotational_symmetry():
    """Verify that a symmetric move by any seat produces identical encoding in their respective POV."""
    state = _cpp.create_initial_state()

    # Move Player 0 token out to spawn (pos 0) with a 6
    state = _cpp.set_dice(state, 6)
    state = _cpp.apply_move_from_cell(state, 2, 2)  # P0 base cell
    frame_p0 = encode_frame_4p(state, pov_player=0)

    # Now create another game and move Player 1 token out to spawn with a 6
    state2 = _cpp.create_initial_state()
    state2 = _cpp.pass_turn(state2)
    assert state2.current_player == 1
    state2 = _cpp.set_dice(state2, 6)
    # P1 base cell in world coordinates for P1: (2, 12)
    state2 = _cpp.apply_move_from_cell(state2, 2, 12)
    frame_p1 = encode_frame_4p(state2, pov_player=1)

    # Since P0 in state1 and P1 in state2 made the exact same move,
    # their encoded frames in their own POVs must be IDENTICAL!
    np.testing.assert_array_equal(frame_p0, frame_p1)


def test_graph_4p_matrix():
    assert EDGE_TYPE_MATRIX_4P.shape == (226, 226)
    assert EDGE_TYPE_MATRIX_4P.dtype == np.int8
    assert len(EDGES_4P) > 500

    # Test edge types exist
    edge_types_present = set(np.unique(EDGE_TYPE_MATRIX_4P))
    assert 0 in edge_types_present
    for t in range(1, NUM_EDGE_TYPES):
        assert t in edge_types_present, f"Missing edge type {EdgeType(t).name}"
