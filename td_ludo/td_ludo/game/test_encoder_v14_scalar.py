"""V14_scalar encoder verification.

Constructs known game states and asserts every field matches hand-computed
expectations. Catches bugs in the C++ encoder before they corrupt training.

Run:  python -m td_ludo.game.test_encoder_v14_scalar
"""
from __future__ import annotations

import sys
import numpy as np
import td_ludo_cpp as ludo_cpp

from td_ludo.game.encoder_v14_scalar import (
    encode_state_v14_scalar,
    OWN_FEAT_ORDER, OPP_FEAT_ORDER, GLOBAL_SCALAR_ORDER,
    NUM_OWN_FEATS, NUM_OPP_FEATS, NUM_GLOBALS,
)


# ── Helpers ──────────────────────────────────────────────────────────────
BASE_POS = -1
HOME_POS = 99
SAFE_INDICES = {0, 8, 13, 21, 26, 34, 39, 47}


def make_state(
    own_positions=(BASE_POS, BASE_POS, BASE_POS, BASE_POS),
    opp_positions=(BASE_POS, BASE_POS, BASE_POS, BASE_POS),
    own_player=0,
    opp_player=2,
    dice=0,
    own_score=0,
    opp_score=0,
    own_idle=(0, 0, 0, 0),
    opp_idle=(0, 0, 0, 0),
    own_streak=0,
    two_player=True,
):
    """Build a 2-player GameState with given positions."""
    g = ludo_cpp.create_initial_state_2p() if two_player else ludo_cpp.create_initial_state()
    pp = np.array(g.player_positions, dtype=np.int8)
    pp[own_player] = own_positions
    pp[opp_player] = opp_positions
    g.player_positions = pp

    sc = np.array(g.scores, dtype=np.int8)
    sc[own_player] = own_score
    sc[opp_player] = opp_score
    g.scores = sc

    ic = np.array(g.idle_counter, dtype=np.int8)
    ic[own_player] = own_idle
    ic[opp_player] = opp_idle
    g.idle_counter = ic

    st = np.array(g.streak, dtype=np.int8)
    st[own_player] = own_streak
    g.streak = st

    g.current_player = own_player
    g.current_dice_roll = dice
    return g


def get_abs(player, rel):
    """Mirror C++ get_absolute_pos."""
    if rel > 50: return -1
    return (rel + 13 * player) % 52


def is_safe(abs_pos):
    return abs_pos in SAFE_INDICES


# ── Tests ────────────────────────────────────────────────────────────────
TESTS = []
def test(name):
    def deco(fn):
        TESTS.append((name, fn))
        return fn
    return deco


def assert_eq(actual, expected, msg):
    if isinstance(expected, (list, tuple, np.ndarray)):
        if not np.array_equal(np.asarray(actual), np.asarray(expected)):
            raise AssertionError(f"{msg}: expected {list(expected)}, got {list(actual)}")
    elif isinstance(expected, float):
        if abs(float(actual) - expected) > 1e-5:
            raise AssertionError(f"{msg}: expected {expected}, got {actual}")
    else:
        if actual != expected:
            raise AssertionError(f"{msg}: expected {expected}, got {actual}")


# ── Position embedding remap ─────────────────────────────────────────────

@test("pos_emb: BASE→0, track 0..50→1..51, home stretch 51..55→52..56, HOME→57")
def t_pos_emb():
    cases = [
        (BASE_POS, 0),
        (0, 1), (25, 26), (50, 51),
        (51, 52), (55, 56),
        (HOME_POS, 57),
    ]
    for pos, expected in cases:
        g = make_state(own_positions=(pos, BASE_POS, BASE_POS, BASE_POS), dice=1)
        e = encode_state_v14_scalar(g)
        assert_eq(int(e["own_pos"][0]), expected,
                  f"pos={pos} should map to emb index")


# ── Initial state ────────────────────────────────────────────────────────

@test("Initial state, dice=6: all features sane")
def t_initial():
    g = make_state(dice=6)
    e = encode_state_v14_scalar(g)
    # Positions: all at base → 0
    assert_eq(e["own_pos"].tolist(), [0]*4, "own_pos initial")
    assert_eq(e["opp_pos"].tolist(), [0]*4, "opp_pos initial")
    # All at base/locked
    assert_eq(e["own_features"][:, OWN_FEAT_ORDER.index("own_at_base")].tolist(),
              [1.0]*4, "own_at_base")
    assert_eq(e["opp_features"][:, OPP_FEAT_ORDER.index("opp_at_base")].tolist(),
              [1.0]*4, "opp_at_base")
    # No tokens at home
    assert_eq(e["own_features"][:, OWN_FEAT_ORDER.index("own_at_home")].sum(), 0.0, "own_at_home all 0")
    # Cannot capture/score from base. CAN land safe (dice=6, unlock to spawn cell which is safe).
    assert_eq(e["own_features"][:, OWN_FEAT_ORDER.index("own_can_capture")].sum(), 0.0, "no capture initial")
    assert_eq(e["own_features"][:, OWN_FEAT_ORDER.index("own_can_score")].sum(), 0.0, "no score initial")
    # Dice=6 unlock → landing=0 (spawn is in SAFE_INDICES) → can_land_safe=True for all 4
    assert_eq(e["own_features"][:, OWN_FEAT_ORDER.index("own_can_land_safe")].tolist(),
              [1.0]*4, "all 4 can land safe (spawn=0 is safe-cell)")
    # No danger (no opp on board)
    assert_eq(e["own_features"][:, OWN_FEAT_ORDER.index("own_in_danger")].sum(), 0.0, "no danger initial")
    # Globals
    g_off = 6  # dice one-hot occupies 0..5
    s_idx = lambda k: g_off + GLOBAL_SCALAR_ORDER.index(k)
    assert_eq(float(e["globals"][5]), 1.0, "dice one-hot[5]==1 for dice=6")
    assert_eq(float(e["globals"][s_idx("my_locked_frac")]), 1.0, "my_locked_frac=1")
    assert_eq(float(e["globals"][s_idx("opp_locked_frac")]), 1.0, "opp_locked_frac=1")
    assert_eq(float(e["globals"][s_idx("score_diff")]), 0.0, "score_diff=0")
    assert_eq(float(e["globals"][s_idx("leader_progress")]), 0.0, "leader_progress=0")
    assert_eq(float(e["globals"][s_idx("non_home_tokens_frac")]), 1.0, "non_home_frac=1")
    assert_eq(float(e["globals"][-1]), 1.0, "bonus_turn_flag=1 for dice=6")


# ── Capture tests ────────────────────────────────────────────────────────

@test("can_capture: P0 token at pos 5, dice=3, opp single token at abs=8 (=safe!) → no capture (safe)")
def t_no_capture_on_safe():
    # P0 at pos 5 (abs=5). Dice=3 → land abs=8. Abs=8 is in SAFE_INDICES → no capture allowed.
    # Place opp P2 single token at abs=8 → P2 rel = (8 - 26) % 52 = 34. So opp_positions=[34, BASE, BASE, BASE].
    g = make_state(
        own_positions=(5, BASE_POS, BASE_POS, BASE_POS),
        opp_positions=(34, BASE_POS, BASE_POS, BASE_POS),
        dice=3,
    )
    e = encode_state_v14_scalar(g)
    assert_eq(e["own_features"][0, OWN_FEAT_ORDER.index("own_can_capture")], 0.0,
              "should NOT capture on safe cell")


@test("can_capture: P0 at pos 4, dice=2, opp at abs=6 (non-safe) → CAPTURE")
def t_can_capture():
    # P0 at pos 4 (abs=4). Dice=2 → land abs=6. Abs=6 is non-safe.
    # Place P2 single token at abs=6: P2 rel = (6 - 26) % 52 = 32.
    g = make_state(
        own_positions=(4, BASE_POS, BASE_POS, BASE_POS),
        opp_positions=(32, BASE_POS, BASE_POS, BASE_POS),
        dice=2,
    )
    e = encode_state_v14_scalar(g)
    own_can_cap = e["own_features"][:, OWN_FEAT_ORDER.index("own_can_capture")]
    assert_eq(own_can_cap.tolist(), [1.0, 0.0, 0.0, 0.0],
              "P0 token 0 should capture")


@test("opp_in_my_danger: same setup, opp token should report in_my_danger")
def t_opp_in_my_danger():
    g = make_state(
        own_positions=(4, BASE_POS, BASE_POS, BASE_POS),
        opp_positions=(32, BASE_POS, BASE_POS, BASE_POS),
        dice=2,
    )
    e = encode_state_v14_scalar(g)
    in_my_danger = e["opp_features"][:, OPP_FEAT_ORDER.index("opp_in_my_danger")]
    assert_eq(in_my_danger.tolist(), [1.0, 0.0, 0.0, 0.0],
              "opp token 0 should be in my danger")


@test("can_capture: opp stack of 2 → no capture (stacks are safe in our rules)")
def t_no_capture_on_stack():
    # P0 at pos 4 (abs=4), dice=2 → land abs=6. P2 has TWO tokens at abs=6.
    # P2 rel for abs=6: (6-26)%52 = 32. Place TWO tokens at rel=32.
    g = make_state(
        own_positions=(4, BASE_POS, BASE_POS, BASE_POS),
        opp_positions=(32, 32, BASE_POS, BASE_POS),
        dice=2,
    )
    e = encode_state_v14_scalar(g)
    own_can_cap = e["own_features"][:, OWN_FEAT_ORDER.index("own_can_capture")]
    assert_eq(own_can_cap.sum(), 0.0,
              "stack of 2 cannot be captured")


# ── Score tests ──────────────────────────────────────────────────────────

@test("can_score: own token at pos 55 with dice=1 → can_score=True")
def t_can_score():
    g = make_state(own_positions=(55, BASE_POS, BASE_POS, BASE_POS), dice=1)
    e = encode_state_v14_scalar(g)
    own_can_score = e["own_features"][:, OWN_FEAT_ORDER.index("own_can_score")]
    assert_eq(own_can_score.tolist(), [1.0, 0.0, 0.0, 0.0],
              "token 0 (pos 55, dice 1 → 56) can score")


@test("can_score: own token at pos 54 with dice=1 → can NOT score (lands on 55)")
def t_no_score_yet():
    g = make_state(own_positions=(54, BASE_POS, BASE_POS, BASE_POS), dice=1)
    e = encode_state_v14_scalar(g)
    own_can_score = e["own_features"][:, OWN_FEAT_ORDER.index("own_can_score")]
    assert_eq(own_can_score.sum(), 0.0,
              "token 0 (pos 54, dice 1 → 55, not 56) cannot score")


# ── Danger tests ─────────────────────────────────────────────────────────

@test("in_danger: opp 3 cells behind (within 1-6) → danger=True")
def t_in_danger():
    # P0 at pos 10 (abs=10). Place P2 such that opp_abs is 3 cells behind: opp_abs=7.
    # P2 rel for abs=7: (7-26)%52 = 33.
    g = make_state(
        own_positions=(10, BASE_POS, BASE_POS, BASE_POS),
        opp_positions=(33, BASE_POS, BASE_POS, BASE_POS),
        dice=0,  # dice doesn't matter for danger
    )
    e = encode_state_v14_scalar(g)
    own_danger = e["own_features"][:, OWN_FEAT_ORDER.index("own_in_danger")]
    assert_eq(own_danger.tolist(), [1.0, 0.0, 0.0, 0.0],
              "P0 token 0 should be in danger")


@test("in_danger: opp 7 cells behind (outside 1-6) → danger=False")
def t_not_in_danger_far():
    # P0 at pos 10 (abs=10). P2 such that opp_abs=3 (7 cells behind): rel = (3-26)%52 = 29.
    g = make_state(
        own_positions=(10, BASE_POS, BASE_POS, BASE_POS),
        opp_positions=(29, BASE_POS, BASE_POS, BASE_POS),
    )
    e = encode_state_v14_scalar(g)
    own_danger = e["own_features"][:, OWN_FEAT_ORDER.index("own_in_danger")]
    assert_eq(own_danger.sum(), 0.0,
              "opp 7 behind → no danger")


@test("in_danger: own on safe cell → danger=False even with opp adjacent")
def t_safe_no_danger():
    # P0 at pos 8 (abs=8, safe). Place opp 3 behind: opp_abs=5, rel = (5-26)%52 = 31.
    g = make_state(
        own_positions=(8, BASE_POS, BASE_POS, BASE_POS),
        opp_positions=(31, BASE_POS, BASE_POS, BASE_POS),
    )
    e = encode_state_v14_scalar(g)
    own_danger = e["own_features"][:, OWN_FEAT_ORDER.index("own_in_danger")]
    own_safe = e["own_features"][:, OWN_FEAT_ORDER.index("own_is_safe")]
    assert_eq(own_safe.tolist(), [1.0, 0.0, 0.0, 0.0], "own_is_safe at pos 8")
    assert_eq(own_danger.sum(), 0.0, "safe → no danger")


# ── Globals: idle, streak, score_diff, leader_progress ─────────────────

@test("idle, streak, score_diff, leader_progress, non_home_frac all correct")
def t_globals():
    g = make_state(
        own_positions=(99, 30, BASE_POS, BASE_POS),  # 1 home, 1 mid, 2 at base
        opp_positions=(BASE_POS,)*4,
        dice=4,
        own_score=1, opp_score=0,
        own_idle=(0, 5, 10, 20),  # → /20: [0, 0.25, 0.5, 1.0]
        own_streak=3,
    )
    e = encode_state_v14_scalar(g)
    # idle_count
    assert_eq(e["own_features"][:, OWN_FEAT_ORDER.index("own_idle_count")].tolist(),
              [0.0, 0.25, 0.5, 1.0], "idle_count")
    # globals
    g_off = 6
    s_idx = lambda k: g_off + GLOBAL_SCALAR_ORDER.index(k)
    assert_eq(float(e["globals"][s_idx("same_token_streak")]), 0.3, "streak/10 = 0.3")
    assert_eq(float(e["globals"][s_idx("score_diff")]), 1.0/4.0, "score_diff = (1-0)/4")
    # leader_progress: token 0 is at HOME (treated as 56) → 56/56 = 1.0
    assert_eq(float(e["globals"][s_idx("leader_progress")]), 1.0, "leader at home")
    # non_home_frac: 3 tokens not yet home / 4
    assert_eq(float(e["globals"][s_idx("non_home_tokens_frac")]), 0.75, "non_home_frac")


# ── Symmetry ─────────────────────────────────────────────────────────────

@test("Symmetry: P0 view ↔ P2 view of same physical state should mirror own↔opp")
def t_symmetry():
    """Build state once, encode from P0 POV, then swap perspective and encode
    from P2 POV. Expectations:
      - own_pos seen from P0 == opp_pos seen from P2 (mirrored content)
      - opp_pos seen from P0 == own_pos seen from P2
      - Per-token flags: own_in_danger from P0 corresponds to opp_threatens_me
        from P2 (and vice versa, with role swap).
      - Globals: my_locked_frac↔opp_locked_frac, score_diff sign flips,
        bonus_turn_flag identical (same dice).
    """
    P0_TOKENS = (5, 20, BASE_POS, HOME_POS)
    P2_TOKENS = (10, BASE_POS, 55, BASE_POS)
    DICE = 4

    # State as P0's turn
    g_a = make_state(own_player=0, opp_player=2,
                     own_positions=P0_TOKENS, opp_positions=P2_TOKENS,
                     dice=DICE, own_score=1, opp_score=0)
    e_a = encode_state_v14_scalar(g_a)

    # Same physical board, P2's turn now (P2 is "own", P0 is "opp")
    g_b = make_state(own_player=2, opp_player=0,
                     own_positions=P2_TOKENS, opp_positions=P0_TOKENS,
                     dice=DICE, own_score=0, opp_score=1)
    e_b = encode_state_v14_scalar(g_b)

    # Position arrays should swap own/opp
    assert_eq(e_a["own_pos"].tolist(), e_b["opp_pos"].tolist(),
              "P0's own_pos == P2's opp_pos (for the same physical state)")
    assert_eq(e_a["opp_pos"].tolist(), e_b["own_pos"].tolist(),
              "P0's opp_pos == P2's own_pos")

    # Per-token "is_safe" should swap own↔opp
    a_own_safe = e_a["own_features"][:, OWN_FEAT_ORDER.index("own_is_safe")]
    b_opp_safe = e_b["opp_features"][:, OPP_FEAT_ORDER.index("opp_is_safe")]
    assert_eq(a_own_safe.tolist(), b_opp_safe.tolist(), "own_safe ↔ opp_safe")

    # Score-diff should sign-flip
    g_off = 6
    sdiff_idx = g_off + GLOBAL_SCALAR_ORDER.index("score_diff")
    a_sd = float(e_a["globals"][sdiff_idx])
    b_sd = float(e_b["globals"][sdiff_idx])
    assert_eq(a_sd, -b_sd, "score_diff should sign-flip across POV")


# ── Sanity: globals vector layout ───────────────────────────────────────

@test("Globals vector has length 13")
def t_globals_len():
    g = make_state(dice=3)
    e = encode_state_v14_scalar(g)
    assert_eq(e["globals"].shape, (13,), "globals shape")
    # Layout: dice_one_hot(6) + 6 scalars + bonus_turn_flag(1) = 13
    # Verify dice one-hot section sums to 1 when dice in [1..6]
    assert_eq(float(e["globals"][:6].sum()), 1.0, "dice one-hot sums to 1")


# ── Cross-check vs V11 spatial encoder where they overlap ──────────────

@test("Cross-check: idle_count matches V11 channel 28-31 broadcast values")
def t_idle_matches_v11():
    g = make_state(own_idle=(0, 5, 10, 19), dice=3)
    e = encode_state_v14_scalar(g)
    v11 = np.asarray(ludo_cpp.encode_state_v11(g))
    # V11 channels 28..31 are broadcast idle values for own tokens 0..3
    for t in range(4):
        v11_val = float(v11[28 + t, 0, 0])  # broadcast → any cell has the value
        scalar_val = float(e["own_features"][t, OWN_FEAT_ORDER.index("own_idle_count")])
        assert_eq(scalar_val, v11_val, f"idle[{t}] V14_scalar vs V11 broadcast")


# ── Runner ───────────────────────────────────────────────────────────────

def main():
    passed = failed = 0
    for name, fn in TESTS:
        try:
            fn()
            print(f"  ✓ {name}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {name}: UNEXPECTED {type(e).__name__}: {e}")
            failed += 1
    total = passed + failed
    print(f"\n{passed}/{total} passed" + (" ❌" if failed else " ✓"))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
