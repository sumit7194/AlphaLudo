"""Tests for cells.py — cell/index round-trip, special-cell uniqueness, etc."""
from __future__ import annotations

import pytest

import td_ludo_v15_cpp as _cpp
from td_ludo_v15.game import cells
from td_ludo_v15.game.cells import (
    BOARD_SIZE,
    CLS_INDEX,
    HOME_BASE_COUNTER,
    HOME_CENTER,
    HOME_STRETCH_CELLS_P0,
    MD_CELL,
    MS_CELL,
    NUM_BOARD_CELLS,
    NUM_NODES,
    OD_CELL,
    OPP_HOME_BASE_COUNTER,
    OPP_HOME_STRETCH_CELLS,
    OS_CELL,
    SPECIAL_CELLS,
    cell_to_index,
    index_to_cell,
    is_special_cell,
    position_to_cell,
)


def test_grid_constants():
    assert BOARD_SIZE == 15
    assert NUM_BOARD_CELLS == 225
    assert CLS_INDEX == 225
    assert NUM_NODES == 226


def test_cell_index_roundtrip():
    """Every cell maps to a unique index and back."""
    seen = set()
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            idx = cell_to_index(r, c)
            assert 0 <= idx < NUM_BOARD_CELLS
            assert idx not in seen, f"({r},{c}) collided with another cell"
            seen.add(idx)
            assert index_to_cell(idx) == (r, c)
    assert len(seen) == NUM_BOARD_CELLS


def test_cell_index_out_of_bounds():
    with pytest.raises(ValueError):
        cell_to_index(-1, 0)
    with pytest.raises(ValueError):
        cell_to_index(15, 0)
    with pytest.raises(ValueError):
        cell_to_index(0, 15)
    with pytest.raises(ValueError):
        index_to_cell(-1)
    with pytest.raises(ValueError):
        index_to_cell(225)


def test_special_cells_unique():
    """The 4 global cells are distinct."""
    assert len(set(SPECIAL_CELLS)) == 4


def test_special_cell_values():
    assert MD_CELL == (0, 0)
    assert OD_CELL == (14, 14)
    assert MS_CELL == (7, 6)
    assert OS_CELL == (7, 8)


def test_is_special_cell():
    assert is_special_cell(0, 0)
    assert is_special_cell(14, 14)
    assert is_special_cell(7, 6)
    assert is_special_cell(7, 8)
    # Non-specials
    assert not is_special_cell(7, 7)
    assert not is_special_cell(2, 2)  # home base counter, not "special" in this sense
    assert not is_special_cell(0, 1)


def test_home_counters_consistent():
    """Counter cells must not collide with each other or special cells."""
    counters = [HOME_BASE_COUNTER, OPP_HOME_BASE_COUNTER, HOME_CENTER]
    assert len(set(counters)) == 3  # all distinct
    # Counters are NOT one of the 4 globals (MS happens to coincide with HOME_CENTER
    # but that's not a counter cell collision — MS is a global slot at the same
    # (7,6) physical cell). The encoder will use MS-as-global semantic, never as
    # a home counter, so there's no ambiguity.
    assert HOME_BASE_COUNTER not in SPECIAL_CELLS  # (2,2) not in globals
    assert OPP_HOME_BASE_COUNTER not in SPECIAL_CELLS  # (11,11) not in globals


def test_home_stretch_p0():
    """P0's home stretch must run (7,1)..(7,5)."""
    assert HOME_STRETCH_CELLS_P0 == ((7, 1), (7, 2), (7, 3), (7, 4), (7, 5))
    # Each stretch position maps to its cell via cpp
    for idx, (r, c) in enumerate(HOME_STRETCH_CELLS_P0):
        pos = 51 + idx  # home stretch starts at position 51
        assert position_to_cell(pos, 0) == (r, c)


def test_opp_home_stretch_via_rotation():
    """In P0's POV, opp (=P2 in 2P) stretch is rotated 180°.

    For P2, stretch positions 51..55 in P2's POV map to cells (7,1)..(7,5).
    In the board's actual coordinates (P0's view), those cells rotate to
    (7,13)..(7,9).
    """
    expected = OPP_HOME_STRETCH_CELLS  # (7,13), (7,12), (7,11), (7,10), (7,9)
    for idx, exp_cell in enumerate(expected):
        pos = 51 + idx
        actual_cell = position_to_cell(pos, 2)  # in P2's perspective on the board
        assert actual_cell == exp_cell, (
            f"stretch pos {pos} for P2 → {actual_cell}, expected {exp_cell}"
        )


def test_path_position_zero_p0():
    """P0 path position 0 (just-spawned) lives at (6, 1) per the engine's
    PATH_COORDS_P0 table."""
    assert position_to_cell(0, 0) == (6, 1)


def test_path_position_zero_p2():
    """P2 path position 0 (just-spawned) is at (6,1) in P2's POV → after 180°
    rotation, (8, 13) in board's absolute coords."""
    # The legacy fix discovery (commit 1ff249f) established that the rotation
    # rule is k=current_player CCW around (7,7), so for P2: (r,c) -> ... twice.
    cell = position_to_cell(0, 2)
    assert cell == (8, 13), f"P2 spawn cell should be (8,13), got {cell}"


def test_base_position_p0():
    """Base position for P0 (slot 0) maps to (2,2) home counter."""
    assert position_to_cell(_cpp.BASE_POS, 0) == HOME_BASE_COUNTER


def test_home_position_p0_is_center():
    """HOME (scored) position maps to (7,6) in P0's absolute view (canonical).

    For other players, HOME rotates to a different actual-board cell:
        P0 → (7, 6)     no rotation
        P1 → (6, 7)     90° CW
        P2 → (7, 8)     180°
        P3 → (8, 7)     270° CW

    These are absolute board coords. After re-rotating to current_player's POV
    in the encoder, all players see THEIR HOME at (7, 6) in their POV — but
    that's an encoder-level transformation, not a position_to_cell property.
    """
    assert position_to_cell(_cpp.HOME_POS, 0) == HOME_CENTER  # (7, 6)
    # Validate the rotation pattern for the other players
    assert position_to_cell(_cpp.HOME_POS, 1) == (6, 7)
    assert position_to_cell(_cpp.HOME_POS, 2) == (7, 8)
    assert position_to_cell(_cpp.HOME_POS, 3) == (8, 7)


def test_rotate_cell_ccw_identity():
    from td_ludo_v15.game.cells import rotate_cell_ccw
    assert rotate_cell_ccw(0, 0, 0) == (0, 0)
    assert rotate_cell_ccw(7, 7, 0) == (7, 7)  # center stays
    # 4 rotations = identity
    for r in (0, 5, 7, 14):
        for c in (0, 3, 7, 14):
            assert rotate_cell_ccw(r, c, 4) == (r, c)


def test_rotate_cell_ccw_quarters():
    from td_ludo_v15.game.cells import rotate_cell_ccw
    # P0 spawn (6, 1) one CCW → (14-1, 6) = (13, 6)
    assert rotate_cell_ccw(6, 1, 1) == (13, 6)
    # 180° = two CCW: (6,1) → (13,6) → (14-6, 13) = (8, 13)
    assert rotate_cell_ccw(6, 1, 2) == (8, 13)
    # 270° = three CCW: → ... → (1, 8)
    assert rotate_cell_ccw(6, 1, 3) == (1, 8)


def test_rotate_cw_is_inverse_of_ccw():
    from td_ludo_v15.game.cells import rotate_cell_ccw, rotate_cell_cw
    for r in (0, 5, 7, 14):
        for c in (0, 3, 7, 14):
            for k in (0, 1, 2, 3):
                rr, cc = rotate_cell_ccw(r, c, k)
                back_r, back_c = rotate_cell_cw(rr, cc, k)
                assert (back_r, back_c) == (r, c), (
                    f"CW(CCW(({r},{c}),k={k}),k={k}) should be identity, "
                    f"got ({back_r},{back_c})"
                )


def test_position_to_cell_in_pov_own_token_canonical():
    """For OWN tokens (token_owner == pov_player), the POV cell should be
    P0-canonical regardless of who's playing."""
    from td_ludo_v15.game.cells import position_to_cell_in_pov
    # Position 0 → P0-canonical (6, 1). For any player p, their OWN pos=0
    # should appear at (6, 1) in their own POV.
    for pov_player in range(4):
        cell = position_to_cell_in_pov(0, pov_player, pov_player)
        assert cell == (6, 1), (
            f"own token at pos=0 should appear at (6,1) in POV={pov_player}, got {cell}"
        )


def test_position_to_cell_in_pov_opp_token_2p():
    """In 2P mode (current=0, opp=2): opp's pos=0 in P0's POV is at (8,13).

    This is the 180°-rotated spawn cell from P0's perspective: P2's pieces
    are on the diagonal-opposite side of the board.
    """
    from td_ludo_v15.game.cells import position_to_cell_in_pov
    cell = position_to_cell_in_pov(0, token_owner=2, pov_player=0)
    assert cell == (8, 13), f"opp(P2) spawn cell in P0 POV → expected (8,13), got {cell}"


def test_print_grid_shape_check():
    import numpy as np
    bad = np.zeros((10, 15), dtype=np.int8)
    with pytest.raises(ValueError):
        cells.print_grid(bad)
    good = np.arange(225, dtype=np.int32).reshape(15, 15)
    out = cells.print_grid(good)
    assert "0" in out and "224" in out
