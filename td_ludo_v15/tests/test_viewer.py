"""Smoke tests for the board viewer + dump_state CLI."""
from __future__ import annotations

from td_ludo_v15.game.state import V15GameWrapper
from td_ludo_v15.scripts.dump_state import play_random_moves
from td_ludo_v15.viz.board_viewer import (
    render_board,
    render_board_text,
    render_side_by_side,
    render_triplet_grid,
)


def test_render_board_initial():
    g = V15GameWrapper.new_2p()
    board = render_board(g.state, pov_player=0)
    assert board.shape == (15, 15)
    # My home counter should show 4H (4 locked tokens)
    assert board[2, 2] == "4H"
    # Opp home counter shows 4h
    assert board[12, 12] == "4h"
    # MD slot is inactive at game start
    assert board[0, 0] == "M-"
    # MS slot shows 0 scored
    assert board[7, 6] == "$0"


def test_render_board_text_runs():
    g = V15GameWrapper.new_2p()
    out = render_board_text(g.state, pov_player=0)
    assert "Board (POV player=0)" in out
    assert "4H" in out
    assert "4h" in out


def test_render_side_by_side_includes_all_three_slots():
    g = V15GameWrapper.new_2p()
    out = render_side_by_side(g.state, pov_player=0)
    assert "BOARD LEGEND" in out
    assert "slot a" in out
    assert "slot b" in out
    assert "slot c" in out
    # The board section should appear once, encoding section appears once
    assert out.count("Board (POV player") == 1


def test_play_random_moves_makes_progress():
    g = play_random_moves(seed=42, n_moves=10)
    # After 10 moves, at least one token should not be at base
    own_positions = list(g.state.player_positions[0])
    opp_positions = list(g.state.player_positions[2])
    # Some moves must have happened (at least one token spawned)
    non_base = sum(1 for p in own_positions + opp_positions if p != -1)
    assert non_base >= 1, f"expected at least 1 token spawned after 10 moves, got 0"
