"""V15 engine vs legacy `td_ludo_cpp` parity tests.

We play N random games on both engines with identical RNG seeds. At each step
the legacy engine and V15 engine see the same dice roll, the same legal-move
list (via V15's internal slot-based API), the same picked slot. After every
step we assert that scores, player_positions, current_player, is_terminal,
and winner all match.

Why the slot-based API on V15: this test isolates "the rules logic is
identical." The cell-based API is verified separately (test_engine_parity
also asserts that the cell-based path produces the same result as the
slot-based path on a per-call basis — see test_cell_api_matches_slot_api).

The 10K-game run takes ~30-60s and is marked `@pytest.mark.slow`. The
default `pytest` run uses 100 games for fast CI feedback.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

import td_ludo_cpp as legacy
import td_ludo_v15_cpp as v15


MAX_STEPS_PER_GAME = 2000  # generous; real games finish in ~150-300


def play_legacy(seed: int, max_steps: int = MAX_STEPS_PER_GAME):
    """Play one game on the legacy engine. Returns (final_state, n_steps,
    winner). Forfeit + pass-turn handled in Python (mirrors how trainers do it).
    """
    rng = random.Random(seed)
    state = legacy.create_initial_state_2p()
    csix = [0, 0, 0, 0]
    steps = 0
    while not state.is_terminal and steps < max_steps:
        cp = int(state.current_player)
        # Roll dice
        d = rng.randint(1, 6)
        if d == 6:
            csix[cp] += 1
            if csix[cp] >= 3:
                # Forfeit
                csix[cp] = 0
                nxt = (cp + 1) % 4
                while not state.active_players[nxt]:
                    nxt = (nxt + 1) % 4
                state.current_player = nxt
                state.current_dice_roll = 0
                continue
        else:
            csix[cp] = 0
        state.current_dice_roll = d
        legal = legacy.get_legal_moves(state)
        if not legal:
            nxt = (cp + 1) % 4
            while not state.active_players[nxt]:
                nxt = (nxt + 1) % 4
            state.current_player = nxt
            state.current_dice_roll = 0
            continue
        slot = rng.choice(legal)
        state = legacy.apply_move(state, slot)
        steps += 1
    winner = legacy.get_winner(state) if state.is_terminal else -1
    return state, steps, winner


def play_v15(seed: int, max_steps: int = MAX_STEPS_PER_GAME):
    """Play one game on the V15 engine, mirroring the legacy driver exactly
    (same RNG seed → same dice → same slot pick). Uses V15's internal slot
    API to keep action-selection deterministic between engines.
    """
    rng = random.Random(seed)
    state = v15.create_initial_state_2p()
    steps = 0
    while not state.is_terminal and steps < max_steps:
        d = rng.randint(1, 6)
        state = v15.set_dice(state, d)
        if state.current_dice_roll == 0:
            # Forfeit happened atomically inside set_dice
            continue
        legal = v15._internal_get_legal_token_slots(state)
        if not legal:
            state = v15.pass_turn(state)
            continue
        slot = rng.choice(legal)
        state = v15._internal_apply_move_by_slot(state, slot)
        steps += 1
    winner = v15.get_winner(state) if state.is_terminal else -1
    return state, steps, winner


def assert_states_equal(legacy_state, v15_state, ctx: str = ""):
    """Step-level state equality."""
    assert int(legacy_state.current_player) == int(v15_state.current_player), (
        f"{ctx}: current_player differs "
        f"legacy={legacy_state.current_player} v15={v15_state.current_player}"
    )
    assert int(legacy_state.current_dice_roll) == int(v15_state.current_dice_roll), (
        f"{ctx}: dice differs"
    )
    assert bool(legacy_state.is_terminal) == bool(v15_state.is_terminal), (
        f"{ctx}: is_terminal differs"
    )
    np.testing.assert_array_equal(
        np.asarray(legacy_state.player_positions),
        np.asarray(v15_state.player_positions),
        err_msg=f"{ctx}: player_positions differ",
    )
    np.testing.assert_array_equal(
        np.asarray(legacy_state.scores),
        np.asarray(v15_state.scores),
        err_msg=f"{ctx}: scores differ",
    )


def play_step_by_step_compare(seed: int, max_steps: int = MAX_STEPS_PER_GAME):
    """Run both engines in lock-step, comparing after every step."""
    rng_l = random.Random(seed)
    rng_v = random.Random(seed)
    leg = legacy.create_initial_state_2p()
    vee = v15.create_initial_state_2p()
    csix = [0, 0, 0, 0]
    steps = 0
    while not leg.is_terminal and not vee.is_terminal and steps < max_steps:
        cp_l = int(leg.current_player)
        cp_v = int(vee.current_player)
        assert cp_l == cp_v, f"step {steps}: current_player diverged"
        d = rng_l.randint(1, 6)
        d_v = rng_v.randint(1, 6)
        assert d == d_v, "RNG diverged (shouldn't happen with same seed)"

        # Drive legacy via Python forfeit logic
        if d == 6:
            csix[cp_l] += 1
            if csix[cp_l] >= 3:
                csix[cp_l] = 0
                nxt = (cp_l + 1) % 4
                while not leg.active_players[nxt]:
                    nxt = (nxt + 1) % 4
                leg.current_player = nxt
                leg.current_dice_roll = 0
                # Drive V15 too
                vee = v15.set_dice(vee, d)
                assert int(vee.current_dice_roll) == 0, (
                    f"step {steps}: V15 should have forfeited"
                )
                assert_states_equal(leg, vee, f"step {steps} (forfeit)")
                continue
        else:
            csix[cp_l] = 0
        leg.current_dice_roll = d
        vee = v15.set_dice(vee, d)
        assert int(vee.current_dice_roll) == d, (
            f"step {steps}: V15 dice should be {d}, got {vee.current_dice_roll}"
        )

        legal_l = legacy.get_legal_moves(leg)
        legal_v = v15._internal_get_legal_token_slots(vee)
        assert list(legal_l) == list(legal_v), (
            f"step {steps}: legal-slot lists differ\n"
            f"  legacy: {list(legal_l)}\n"
            f"  v15   : {list(legal_v)}"
        )

        if not legal_l:
            nxt = (cp_l + 1) % 4
            while not leg.active_players[nxt]:
                nxt = (nxt + 1) % 4
            leg.current_player = nxt
            leg.current_dice_roll = 0
            vee = v15.pass_turn(vee)
            assert_states_equal(leg, vee, f"step {steps} (pass-turn)")
            continue

        slot = rng_l.choice(legal_l)
        slot_v = rng_v.choice(legal_v)
        assert slot == slot_v, "RNG diverged"
        leg = legacy.apply_move(leg, slot)
        vee = v15._internal_apply_move_by_slot(vee, slot)
        assert_states_equal(leg, vee, f"step {steps} (after move slot={slot})")
        steps += 1
    return steps, leg, vee


# ─── Tests ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", list(range(20)))
def test_parity_fast(seed):
    """Fast parity: 20 games, full step-by-step state compare."""
    steps, leg, vee = play_step_by_step_compare(seed)
    # Both should be terminal (or both at max-steps)
    assert int(leg.is_terminal) == int(vee.is_terminal)
    if leg.is_terminal:
        assert int(legacy.get_winner(leg)) == int(v15.get_winner(vee))


@pytest.mark.slow
@pytest.mark.parametrize("seed_offset", [0, 1000, 5000])
def test_parity_slow(seed_offset):
    """Slow parity: 3 sweeps of 1000 games each starting at different seed
    offsets to exercise the RNG space."""
    n_games = 1000
    for s in range(seed_offset, seed_offset + n_games):
        steps, leg, vee = play_step_by_step_compare(s)
        assert int(leg.is_terminal) == int(vee.is_terminal), f"seed {s}: terminality differs"
        if leg.is_terminal:
            assert int(legacy.get_winner(leg)) == int(v15.get_winner(vee)), (
                f"seed {s}: winners differ"
            )


def test_cell_api_matches_slot_api():
    """For the same state, applying via cell-API should match applying via
    the internal slot-API on V15. This validates that V15's public
    cell-based path is a faithful wrapper around the internal slot logic.
    """
    rng = random.Random(0xABCD)
    state = v15.create_initial_state_2p()
    steps = 0
    while not state.is_terminal and steps < 500:
        d = rng.randint(1, 6)
        state = v15.set_dice(state, d)
        if state.current_dice_roll == 0:
            continue
        legal_slots = v15._internal_get_legal_token_slots(state)
        if not legal_slots:
            state = v15.pass_turn(state)
            continue
        # Pick a cell, find the slot V15's cell-API would internally pick
        # for that cell (lowest-indexed legal slot at the cell). Verify the
        # slot-API applied to THAT slot gives the same result as the cell-API.
        cell_choices = v15.get_legal_source_cells(state)
        cell = rng.choice(cell_choices)
        cp = int(state.current_player)
        # Find lowest legal slot at this cell — must match cell-API's tiebreak
        slot = -1
        for s in legal_slots:
            pos = int(state.player_positions[cp][s])
            if pos == v15.BASE_POS:
                scell = v15.position_to_cell(v15.BASE_POS, cp)
            else:
                scell = v15.position_to_cell(pos, cp)
            if scell == cell:
                slot = s
                break
        assert slot >= 0, "expected at least one legal slot at the chosen cell"
        # Slot-based application
        via_slot = v15._internal_apply_move_by_slot(state, slot)
        # Cell-based application
        via_cell = v15.apply_move_from_cell(state, cell[0], cell[1])
        # They should match on player_positions, scores, current_player.
        np.testing.assert_array_equal(
            np.asarray(via_slot.player_positions),
            np.asarray(via_cell.player_positions),
            err_msg=f"step {steps}: player_positions differ between slot/cell apply",
        )
        np.testing.assert_array_equal(
            np.asarray(via_slot.scores), np.asarray(via_cell.scores),
            err_msg=f"step {steps}: scores differ between slot/cell apply",
        )
        assert int(via_slot.current_player) == int(via_cell.current_player), (
            f"step {steps}: current_player differs"
        )
        # Advance via cell-API for the next iteration
        state = via_cell
        steps += 1
    # Reach here if game terminated or step cap hit — either is fine.


def test_three_six_forfeit_atomic():
    """Three consecutive 6 rolls → V15's set_dice atomically forfeits."""
    s = v15.create_initial_state_2p()
    s = v15.set_dice(s, 6)
    assert int(s.consecutive_sixes[int(s.current_player)]) == 1
    assert int(s.current_dice_roll) == 6
    # Pretend the player moved; for simplicity, just spawn from base
    cells = v15.get_legal_source_cells(s)
    assert (2, 2) in cells
    s = v15.apply_move_from_cell(s, 2, 2)
    # Bonus turn, dice reset to 0, still cp=0
    assert int(s.current_player) == 0
    assert int(s.current_dice_roll) == 0
    # Roll second 6
    s = v15.set_dice(s, 6)
    assert int(s.consecutive_sixes[0]) == 2
    s = v15.apply_move_from_cell(s, *v15.get_legal_source_cells(s)[0])
    # Roll third 6 → forfeit
    s = v15.set_dice(s, 6)
    assert int(s.current_dice_roll) == 0, "third six should forfeit (dice cleared)"
    assert int(s.consecutive_sixes[0]) == 0, "consecutive counter should reset"
    assert int(s.current_player) == 2, "turn should have passed to P2 (2P mode)"
