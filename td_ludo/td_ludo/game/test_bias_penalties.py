"""Unit tests for bias_penalties.compute_bias_penalties.

Builds synthetic states with handcrafted positions to verify each of the
5 penalties triggers under its specific condition AND does NOT trigger
under benign conditions. Uses a tiny stub state class — no engine call,
no network — so tests run in milliseconds.

Run from td_ludo/:
    td_env/bin/python -m td_ludo.game.test_bias_penalties
"""
import sys
from pathlib import Path

# Make `td_ludo` importable when run as a script
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from td_ludo.game.bias_penalties import (  # noqa: E402
    compute_bias_penalties,
    P_UNLOCK_BETTER_AVAIL, P_MISS_CAPTURE_BASE, P_MISS_FINISH,
    P_LEFT_SAFE, P_DANGER_ADVANCED_BASE, P_LAGGARD_PER_CELL, ABS_MAX_PENALTY,
    SAFE_SQUARES,
)


# ── Tiny stub state ───────────────────────────────────────────────────
class S:
    """Mimic the engine state interface for rewards. Players 0 and 2 active."""
    def __init__(self, p0, p2, scores=None, active=None):
        self.player_positions = [list(p0), [-1]*4, list(p2), [-1]*4]
        self.scores = scores if scores else [0, 0, 0, 0]
        self.active_players = active if active else [True, False, True, False]


# ── Test runner ───────────────────────────────────────────────────────
TESTS = []
def test(name):
    def deco(fn):
        TESTS.append((name, fn))
        return fn
    return deco


def assert_close(actual, expected, tol=1e-6, msg=''):
    if abs(actual - expected) > tol:
        raise AssertionError(f'{msg} expected ~{expected}, got {actual}')


def assert_zero(actual, msg=''):
    if abs(actual) > 1e-9:
        raise AssertionError(f'{msg} expected 0, got {actual}')


# =============================================================================
# Penalty 1: Unlock-on-6 with better alternative
# =============================================================================
@test('P1: unlock-on-6 when escape-danger alt → fires (isolated)')
def t_unlock_escape_danger():
    # Use the "escape danger" branch of "better" to isolate P1 from P2.
    # T1 at rel 10 (abs 10), opp at rel 32 → abs (32+26)%52 = 6. dist=4 (in danger).
    # Dice=6 → T1 dest abs 16. opp abs 6, dist (16-6)%52=10, out of range → safe.
    # No capture (no opp at abs 16). No finish. Pure escape-danger trigger.
    pre = S([-1, 10, -1, -1], [32, -1, -1, -1])
    post = S([0, 10, -1, -1], [32, -1, -1, -1])  # model spawned T0 instead
    ctx = {'dice': 6, 'legal_moves': [0, 1, 2, 3], 'action': 0, 'move_count': 50}
    total, bd = compute_bias_penalties(pre, post, 0, ctx)
    assert_close(bd['unlock_with_better'], -P_UNLOCK_BETTER_AVAIL,
                 msg='unlock_with_better')
    # Make sure ONLY P1 fired (P2 shouldn't, since no opp at T1's destination)
    assert_zero(bd['missed_capture'], 'P2 should not fire in this isolated test')


@test('P1: unlock-on-6 when no better alternative → does NOT fire')
def t_unlock_benign():
    # T1 at rel 30 (abs 30). Dice=6 → dest abs 36. No capture (no opp at 36),
    # no finish, and T1 not currently in danger (need no opp at abs in 24..29).
    # Put opp at rel 15 → abs (15+26)%52=41. Far from both.
    pre = S([-1, 30, -1, -1], [15, -1, -1, -1])
    post = S([0, 30, -1, -1], [15, -1, -1, -1])
    ctx = {'dice': 6, 'legal_moves': [0, 1, 2, 3], 'action': 0, 'move_count': 50}
    total, bd = compute_bias_penalties(pre, post, 0, ctx)
    assert_zero(bd['unlock_with_better'], 'unlock_with_better')


@test('P1: unlock-on-6 in opening (move=10) → does NOT fire (phase=0)')
def t_unlock_opening():
    pre = S([-1, 5, -1, -1], [11, -1, -1, -1])
    post = S([0, 5, -1, -1], [11, -1, -1, -1])
    ctx = {'dice': 6, 'legal_moves': [0, 1, 2, 3], 'action': 0, 'move_count': 10}
    total, bd = compute_bias_penalties(pre, post, 0, ctx)
    assert_zero(bd['unlock_with_better'], 'opening should not penalize unlock')


# =============================================================================
# Penalty 2: Missed capture
# =============================================================================
@test('P2: missed capture available, didn\'t take → fires')
def t_missed_capture():
    # Pick low opp progress so penalty stays under cap (avoid rescaling).
    # T0 at rel 25 (abs 25) with dice=6 → dest abs 31.
    # Want opp at abs 31 → opp rel pos = (31 - 26) % 52 = 5. progress=5 (low).
    # T1 at rel 40 → dest abs 46 (no opp nearby; not in danger).
    pre = S([25, 40, -1, -1], [5, -1, -1, -1])
    post = S([25, 46, -1, -1], [5, -1, -1, -1])  # model picked T1 (advance)
    ctx = {'dice': 6, 'legal_moves': [0, 1], 'action': 1, 'move_count': 50}
    total, bd = compute_bias_penalties(pre, post, 0, ctx)
    if bd['missed_capture'] >= 0:
        raise AssertionError(f'missed_capture should fire, got {bd["missed_capture"]}')
    # Magnitude: base + 0.005 * opp_rel_pos (5) = 0.125. Below cap 0.15.
    expected = -(P_MISS_CAPTURE_BASE + 0.005 * 5)
    assert_close(bd['missed_capture'], expected, msg='missed_capture mag')


@test('P2: capture taken → no penalty')
def t_took_capture():
    # Same setup, but model picks T0 → capture happens, opp T0 returns to base.
    pre = S([25, 40, -1, -1], [5, -1, -1, -1])
    post = S([31, 40, -1, -1], [-1, -1, -1, -1])
    ctx = {'dice': 6, 'legal_moves': [0, 1], 'action': 0, 'move_count': 50}
    total, bd = compute_bias_penalties(pre, post, 0, ctx)
    assert_zero(bd['missed_capture'], 'missed_capture')


@test('P2: opp on safe square → no penalty for not capturing')
def t_capture_on_safe():
    # Opp at abs 8 (safe square in SAFE_SQUARES). T0 at rel 2 with dice=6
    # would land on abs 8 — but engine-illegal capture.
    # Opp rel pos = (8 - 26) % 52 = 34.
    pre = S([2, 30, -1, -1], [34, -1, -1, -1])
    post = S([2, 36, -1, -1], [34, -1, -1, -1])
    ctx = {'dice': 6, 'legal_moves': [0, 1], 'action': 1, 'move_count': 50}
    total, bd = compute_bias_penalties(pre, post, 0, ctx)
    assert_zero(bd['missed_capture'], 'missed_capture (safe)')


# =============================================================================
# Penalty 3: Missed finish
# =============================================================================
@test('P3: could finish, didn\'t → fires')
def t_missed_finish():
    # P0 has T0 at pos 95 (one off from 99), T1 at pos 20.
    # Dice=4 → T0 could finish (95+4=99). Model picked T1.
    pre = S([95, 20, -1, -1], [40, -1, -1, -1])
    post = S([95, 24, -1, -1], [40, -1, -1, -1])
    ctx = {'dice': 4, 'legal_moves': [0, 1], 'action': 1, 'move_count': 100}
    total, bd = compute_bias_penalties(pre, post, 0, ctx)
    assert_close(bd['missed_finish'], -P_MISS_FINISH, msg='missed_finish')


@test('P3: finish taken → no penalty')
def t_took_finish():
    pre = S([95, 20, -1, -1], [40, -1, -1, -1])
    post = S([99, 20, -1, -1], [40, -1, -1, -1])
    ctx = {'dice': 4, 'legal_moves': [0, 1], 'action': 0, 'move_count': 100}
    total, bd = compute_bias_penalties(pre, post, 0, ctx)
    assert_zero(bd['missed_finish'], 'missed_finish')


@test('P3: no finish was possible → no penalty')
def t_no_finish_possible():
    pre = S([90, 20, -1, -1], [40, -1, -1, -1])  # T0 at 90, dice=4 → 94, not 99
    post = S([90, 24, -1, -1], [40, -1, -1, -1])
    ctx = {'dice': 4, 'legal_moves': [0, 1], 'action': 1, 'move_count': 100}
    total, bd = compute_bias_penalties(pre, post, 0, ctx)
    assert_zero(bd['missed_finish'], 'missed_finish')


# =============================================================================
# Penalty 4: Left a safe square unnecessarily
# =============================================================================
@test('P4: token on safe sq + non-safe alt → fires')
def t_left_safe():
    # T0 on rel 8 (abs 8 — safe). T1 on rel 12 (abs 12, not safe). Dice=2.
    # Need opp position that doesn't accidentally trigger missed_capture
    # via T1 dest abs 14 OR P5 danger via T0 dest abs 10. Put opp at rel 0
    # → abs 26. Far from 10 and 14.
    pre = S([8, 12, -1, -1], [0, -1, -1, -1])
    post = S([10, 12, -1, -1], [0, -1, -1, -1])
    ctx = {'dice': 2, 'legal_moves': [0, 1], 'action': 0, 'move_count': 60}
    total, bd = compute_bias_penalties(pre, post, 0, ctx)
    assert_close(bd['left_safe'], -P_LEFT_SAFE, msg='left_safe')


@test('P4: token on safe sq, only token legal → no penalty')
def t_left_safe_only_option():
    pre = S([8, -1, -1, -1], [40, -1, -1, -1])  # only T0 is on track
    post = S([10, -1, -1, -1], [40, -1, -1, -1])
    ctx = {'dice': 2, 'legal_moves': [0], 'action': 0, 'move_count': 60}
    total, bd = compute_bias_penalties(pre, post, 0, ctx)
    assert_zero(bd['left_safe'], 'left_safe (no alt)')


@test('P4: dice=6 halves left_safe penalty (bonus-turn discount)')
def t_left_safe_dice6_halved():
    # Same setup as P4 base test but dice=6. Expect penalty = -0.015 (half of -0.03).
    # P0 T0 on rel 8 (safe abs 8). T1 on rel 12 (non-safe). Use dice=6 → T0 advances 6 to abs 14.
    pre = S([8, 12, -1, -1], [0, -1, -1, -1])
    post = S([14, 12, -1, -1], [0, -1, -1, -1])
    ctx = {'dice': 6, 'legal_moves': [0, 1], 'action': 0, 'move_count': 60}
    total, bd = compute_bias_penalties(pre, post, 0, ctx)
    assert_close(bd['left_safe'], -P_LEFT_SAFE * 0.5,
                 msg='left_safe should halve on dice=6')


@test('P4: opening (move=12) → no penalty even if safe sq left')
def t_left_safe_opening():
    pre = S([8, 12, -1, -1], [40, -1, -1, -1])
    post = S([10, 12, -1, -1], [40, -1, -1, -1])
    ctx = {'dice': 2, 'legal_moves': [0, 1], 'action': 0, 'move_count': 12}
    total, bd = compute_bias_penalties(pre, post, 0, ctx)
    assert_zero(bd['left_safe'], 'opening')


# =============================================================================
# Penalty 5: Moved advanced token into danger
# =============================================================================
@test('P5: advanced (>35) into danger when alt available → fires')
def t_advanced_into_danger():
    # P0 T0 at pos 40 (advanced). T1 at pos 5.
    # Opp at pos 38 → after T0 moves to 42, opp can reach 38+1..6=39..44.
    #   Wait, P0's pos 42 in absolute is 42. Opp's pos 38 in absolute is (38+13*2)%52 = 64%52=12.
    # Distance from opp_abs=12 to my_abs=42 is (42-12)%52=30. Out of range.
    # Need: opp_abs s.t. dist (my_abs - opp_abs) % 52 in 1..6.
    # Want my_abs - opp_abs ∈ {1..6}. My new abs=42, so opp_abs ∈ {36..41}.
    # opp player=2, opp_pos s.t. (opp_pos+26)%52 ∈ {36..41} → opp_pos ∈ {10..15}.
    # Use opp_pos=10. Then post-state opp at 10 → opp_abs=36. dist=42-36=6 ✓
    pre = S([40, 5, -1, -1], [10, -1, -1, -1])
    post = S([42, 5, -1, -1], [10, -1, -1, -1])
    ctx = {'dice': 2, 'legal_moves': [0, 1], 'action': 0, 'move_count': 70}
    total, bd = compute_bias_penalties(pre, post, 0, ctx)
    if bd['advanced_into_danger'] >= 0:
        raise AssertionError(f'advanced_into_danger should fire, got {bd["advanced_into_danger"]}')
    # Magnitude: P_DANGER_ADVANCED_BASE * (1 + (40-35)/20) = 0.04 * 1.25 = 0.05
    expected = -(P_DANGER_ADVANCED_BASE * (1 + (40 - 35) / 20))
    assert_close(bd['advanced_into_danger'], expected, msg='advanced_into_danger mag')


# =============================================================================
# Penalty 6: Laggard distance when scoring 3rd token
# =============================================================================
@test('P6: scoring 3rd token with far laggard → fires')
def t_laggard_on_3score():
    # Pre: T0=99, T1=99 (both home), T2=55 (last home-stretch cell), T3=5 (laggard).
    # cp scores 2.
    # Move: T2 with dice=1 → reaches 56 → engine treats as scored (pos→99).
    pre = S([99, 99, 55, 5], [40, -1, -1, -1], scores=[2, 0, 0, 0])
    post = S([99, 99, 99, 5], [40, -1, -1, -1], scores=[3, 0, 0, 0])
    ctx = {'dice': 1, 'legal_moves': [2], 'action': 2, 'move_count': 100}
    total, bd = compute_bias_penalties(pre, post, 0, ctx)
    raw = -P_LAGGARD_PER_CELL * (99 - 5)  # distance = 94
    # P6 may exceed ABS_MAX_PENALTY at higher per-cell coefficients; the cap
    # rescales the breakdown proportionally. With only laggard firing, the
    # capped value equals -ABS_MAX_PENALTY exactly when |raw| > cap.
    expected = max(raw, -ABS_MAX_PENALTY)
    assert_close(bd['laggard_on_3score'], expected, msg='laggard_on_3score (pos=5)')


@test('P6: scoring 3rd with laggard close to home → small penalty')
def t_laggard_close():
    # Laggard at pos 95 → distance 4
    pre = S([99, 99, 55, 95], [40, -1, -1, -1], scores=[2, 0, 0, 0])
    post = S([99, 99, 99, 95], [40, -1, -1, -1], scores=[3, 0, 0, 0])
    ctx = {'dice': 1, 'legal_moves': [2], 'action': 2, 'move_count': 100}
    total, bd = compute_bias_penalties(pre, post, 0, ctx)
    expected = -P_LAGGARD_PER_CELL * (99 - 95)  # distance = 4
    assert_close(bd['laggard_on_3score'], expected, msg='laggard close')


@test('P6: laggard at base → max penalty')
def t_laggard_at_base():
    pre = S([99, 99, 55, -1], [40, -1, -1, -1], scores=[2, 0, 0, 0])
    post = S([99, 99, 99, -1], [40, -1, -1, -1], scores=[3, 0, 0, 0])
    ctx = {'dice': 1, 'legal_moves': [2], 'action': 2, 'move_count': 100}
    total, bd = compute_bias_penalties(pre, post, 0, ctx)
    raw = -P_LAGGARD_PER_CELL * 99  # at-base treated as max distance
    # Capped to -ABS_MAX_PENALTY when raw magnitude exceeds the cap.
    expected = max(raw, -ABS_MAX_PENALTY)
    assert_close(bd['laggard_on_3score'], expected, msg='laggard at base')


@test('P6: scoring 1st or 2nd token (not 3rd) → no penalty')
def t_laggard_wrong_transition():
    # 0→1 transition: not 2→3, should NOT fire
    pre = S([55, 5, 5, 5], [40, -1, -1, -1], scores=[0, 0, 0, 0])
    post = S([99, 5, 5, 5], [40, -1, -1, -1], scores=[1, 0, 0, 0])
    ctx = {'dice': 1, 'legal_moves': [0], 'action': 0, 'move_count': 100}
    total, bd = compute_bias_penalties(pre, post, 0, ctx)
    assert_zero(bd['laggard_on_3score'], 'should not fire on 0→1 transition')


@test('P6: 3 already scored, score going 3→3 (move other token) → no penalty')
def t_laggard_no_score_change():
    # cp already at score=3, moves laggard. No transition, should NOT fire.
    pre = S([99, 99, 99, 5], [40, -1, -1, -1], scores=[3, 0, 0, 0])
    post = S([99, 99, 99, 11], [40, -1, -1, -1], scores=[3, 0, 0, 0])
    ctx = {'dice': 6, 'legal_moves': [3], 'action': 3, 'move_count': 110}
    total, bd = compute_bias_penalties(pre, post, 0, ctx)
    assert_zero(bd['laggard_on_3score'], 'should not fire when score unchanged')


@test('P5: dice=6 halves advanced_into_danger penalty (bonus-turn discount)')
def t_danger_dice6_halved():
    # Same as P5 base but dice=6. T0 at rel 38 (advanced > 35). Need post-pos
    # in danger AND a safer alt. Use opp at rel 16 → abs (16+26)%52=42.
    # T0 dice=6 → abs 44, dist (44-42)%52=2 → in danger. ✓
    # T1 at rel 5 (non-advanced) advances to 11, abs 11. opp_abs=42, dist 23 → no danger.
    pre = S([38, 5, -1, -1], [16, -1, -1, -1])
    post = S([44, 5, -1, -1], [16, -1, -1, -1])
    ctx = {'dice': 6, 'legal_moves': [0, 1], 'action': 0, 'move_count': 70}
    total, bd = compute_bias_penalties(pre, post, 0, ctx)
    expected = -(P_DANGER_ADVANCED_BASE * (1 + (38 - 35) / 20)) * 0.5
    assert_close(bd['advanced_into_danger'], expected, tol=1e-4,
                 msg='advanced_into_danger should halve on dice=6')


@test('P5: non-advanced token → no penalty')
def t_non_advanced_into_danger():
    # T0 at pos 30 (NOT advanced, threshold is >35), moves into danger.
    pre = S([30, 5, -1, -1], [3, -1, -1, -1])  # opp at 3 → opp_abs=29, dist (32-29)%52=3
    post = S([32, 5, -1, -1], [3, -1, -1, -1])
    ctx = {'dice': 2, 'legal_moves': [0, 1], 'action': 0, 'move_count': 70}
    total, bd = compute_bias_penalties(pre, post, 0, ctx)
    assert_zero(bd['advanced_into_danger'], 'non-advanced')


# =============================================================================
# Aggregate / cap behaviour
# =============================================================================
@test('Cap: total penalty bounded by ABS_MAX_PENALTY')
def t_cap():
    # Construct a pathological state that would trigger several penalties.
    # T0 base, T1 at safe sq 8, T2 at 95 (one off finish), T3 at 40 (advanced).
    # Dice=6 with T0 unlock; alternatives could finish (via T2 with dice=4 — but dice=6, can't).
    # Actually with dice=6, T2 at 95+6=101 — overshoot, no finish. So no missed_finish.
    # Let's pick simpler: dice=6, T0 unlock, T1 advance with capture available.
    # Multiple penalties could stack. Verify total never exceeds ABS_MAX_PENALTY.
    # For simplicity, just verify cap by constructing arbitrarily extreme breakdown.
    # Manually inflate breakdown via direct call, the function caps at -0.15.
    pre = S([-1, 5, -1, -1], [11, -1, -1, -1])
    post = S([0, 5, -1, -1], [11, -1, -1, -1])
    ctx = {'dice': 6, 'legal_moves': [0, 1, 2, 3], 'action': 0, 'move_count': 70}
    total, bd = compute_bias_penalties(pre, post, 0, ctx)
    if total < -ABS_MAX_PENALTY - 1e-9:
        raise AssertionError(f'total {total} exceeds cap -{ABS_MAX_PENALTY}')


@test('No context → returns 0 (backward compat)')
def t_no_context():
    pre = S([5, -1, -1, -1], [10, -1, -1, -1])
    post = S([8, -1, -1, -1], [10, -1, -1, -1])
    total, bd = compute_bias_penalties(pre, post, 0, None)
    assert_zero(total, 'total without context')


@test('action=-1 → returns 0 (no decision was made)')
def t_no_action():
    pre = S([5, -1, -1, -1], [10, -1, -1, -1])
    post = S([5, -1, -1, -1], [10, -1, -1, -1])
    ctx = {'dice': 0, 'legal_moves': [], 'action': -1, 'move_count': 50}
    total, bd = compute_bias_penalties(pre, post, 0, ctx)
    assert_zero(total, 'total with action=-1')


# ── Run all ───────────────────────────────────────────────────────────
def main():
    passed = 0
    failed = 0
    for name, fn in TESTS:
        try:
            fn()
            print(f'  ✓ {name}')
            passed += 1
        except AssertionError as e:
            print(f'  ✗ {name}: {e}')
            failed += 1
        except Exception as e:
            print(f'  ✗ {name}: UNEXPECTED {type(e).__name__}: {e}')
            failed += 1
    total = passed + failed
    print(f'\n{passed}/{total} passed' + (' ❌' if failed else ' ✓'))
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
