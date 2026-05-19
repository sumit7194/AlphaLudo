"""Tests for V15GameWrapper — 8-frame history buffer behavior."""
from __future__ import annotations

import random

import td_ludo_v15_cpp as _cpp
from td_ludo_v15.game.state import HISTORY_LEN, TOTAL_FRAMES, V15GameWrapper


def test_new_2p_initial_state():
    g = V15GameWrapper.new_2p()
    assert g.current_player == 0
    assert g.dice == 0
    assert not g.is_terminal
    # History buffer starts empty
    frames = g.frame_history()
    assert len(frames) == TOTAL_FRAMES
    # First HISTORY_LEN slots are None (no history yet), last is current state
    assert all(f is None for f in frames[:HISTORY_LEN])
    assert frames[-1] is g.state


def test_set_dice_and_pass_turn():
    g = V15GameWrapper.new_2p()
    g.set_dice(5)
    assert g.dice == 5
    # No legal moves (all 4 tokens locked, need 6 to spawn)
    cells = g.get_legal_source_cells()
    assert cells == []
    g.pass_turn()
    assert g.current_player == 2
    assert g.dice == 0


def test_apply_move_advances_history():
    g = V15GameWrapper.new_2p()
    g.set_dice(6)
    cells = g.get_legal_source_cells()
    assert cells == [(2, 2)]  # home counter
    pre_state = g.state
    g.apply_move_from_cell(2, 2)
    # The pre-move state was pushed into history
    frames = g.frame_history()
    # frames[-1] is now post-move (g.state). frames[-2] should be the pre-move snapshot.
    assert frames[-1] is g.state
    assert frames[-2] is pre_state
    # Earlier slots remain None
    assert all(f is None for f in frames[: HISTORY_LEN - 1])


def test_history_caps_at_7():
    """After 8+ moves the history buffer should hold the last 7 pre-move states."""
    g = V15GameWrapper.new_2p()
    rng = random.Random(0xABCD)
    moves_made = 0
    while moves_made < 12 and not g.is_terminal:
        d = rng.randint(1, 6)
        g.set_dice(d)
        if g.dice == 0:
            continue
        cells = g.get_legal_source_cells()
        if not cells:
            g.pass_turn()
            continue
        chosen = cells[0]
        g.apply_move_from_cell(*chosen)
        moves_made += 1
    frames = g.frame_history()
    # All 8 slots should be filled (current + last 7 past)
    assert len(frames) == TOTAL_FRAMES
    assert frames[-1] is g.state
    # The first 7 slots should be non-None GameStates (history is full)
    if moves_made >= HISTORY_LEN:
        assert all(f is not None for f in frames), (
            f"after {moves_made} moves, expected full history, got Nones at "
            f"indices {[i for i,f in enumerate(frames) if f is None]}"
        )


def test_get_own_and_opp_positions():
    g = V15GameWrapper.new_2p()
    own = g.get_own_positions()
    opp = g.get_opp_positions(2)
    assert own == [_cpp.BASE_POS] * 4
    assert opp == [_cpp.BASE_POS] * 4
    # After P0 spawns one token
    g.set_dice(6)
    g.apply_move_from_cell(2, 2)
    own = g.get_own_positions()
    assert 0 in own  # spawned token now at position 0
    assert own.count(_cpp.BASE_POS) == 3  # other 3 still at base
