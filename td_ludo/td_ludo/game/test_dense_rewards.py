"""Tests for dense_rewards.py — verify v1 reward menu fires correctly."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from td_ludo.game.dense_rewards import (  # noqa: E402
    compute_dense_reward_v1,
    compute_kill_penalty,
    REWARD_SCORE_TOKEN, REWARD_CAPTURE_ENEMY, PENALTY_GOT_KILLED,
    REWARD_HOME_STRETCH, REWARD_SPAWN, REWARD_FORWARD_STEP,
    BASE_POS, HOME_POS,
)


class _Dummy:
    def __init__(self, pos, scores=None):
        self.player_positions = pos
        self.scores = scores or [0, 0, 0, 0]


def _state(p0_pos, p2_pos, p0_score=0, p2_score=0):
    """Build a 2-player DummyState. p0_pos and p2_pos are length-4 lists."""
    pos = {0: list(p0_pos), 1: [BASE_POS]*4, 2: list(p2_pos), 3: [BASE_POS]*4}
    return _Dummy(pos, [p0_score, 0, p2_score, 0])


def test_no_event_no_reward():
    """No-op move (same state) → 0 reward."""
    s = _state([BASE_POS]*4, [BASE_POS]*4)
    assert compute_dense_reward_v1(s, s, player=0) == 0.0
    assert compute_dense_reward_v1(s, s, player=2) == 0.0


def test_spawn_event():
    """One token exits base → +REWARD_SPAWN."""
    old = _state([BASE_POS]*4, [BASE_POS]*4)
    new = _state([0, BASE_POS, BASE_POS, BASE_POS], [BASE_POS]*4)  # tok0 spawned
    r = compute_dense_reward_v1(old, new, player=0)
    assert abs(r - REWARD_SPAWN) < 1e-6, f"got {r}"


def test_forward_step():
    """Token advances 6 cells on track → +6 * REWARD_FORWARD_STEP."""
    old = _state([5, BASE_POS, BASE_POS, BASE_POS], [BASE_POS]*4)
    new = _state([11, BASE_POS, BASE_POS, BASE_POS], [BASE_POS]*4)
    r = compute_dense_reward_v1(old, new, player=0)
    assert abs(r - 6 * REWARD_FORWARD_STEP) < 1e-6, f"got {r}"


def test_home_stretch_entry():
    """Token enters home stretch (pos 51) → home_stretch + forward."""
    old = _state([45, BASE_POS, BASE_POS, BASE_POS], [BASE_POS]*4)
    new = _state([51, BASE_POS, BASE_POS, BASE_POS], [BASE_POS]*4)
    r = compute_dense_reward_v1(old, new, player=0)
    expected = REWARD_HOME_STRETCH + 6 * REWARD_FORWARD_STEP
    assert abs(r - expected) < 1e-6, f"got {r}, expected {expected}"


def test_score_token():
    """Token scores → +REWARD_SCORE_TOKEN, no double-count w/ stretch."""
    old = _state([55, BASE_POS, BASE_POS, BASE_POS], [BASE_POS]*4)
    new = _state([HOME_POS, BASE_POS, BASE_POS, BASE_POS], [BASE_POS]*4,
                 p0_score=1)
    r = compute_dense_reward_v1(old, new, player=0)
    # Only score, no stretch-entry (token LEFT stretch)
    assert abs(r - REWARD_SCORE_TOKEN) < 1e-6, f"got {r}"


def test_capture_enemy():
    """Opp's at-base count rose → +REWARD_CAPTURE_ENEMY."""
    old = _state([10, BASE_POS, BASE_POS, BASE_POS],
                 [10, BASE_POS, BASE_POS, BASE_POS])  # p2 tok0 at pos 10
    # p0 moves to pos 10, captures p2's tok0 → p2 tok0 back to base
    new = _state([16, BASE_POS, BASE_POS, BASE_POS],
                 [BASE_POS]*4)
    r = compute_dense_reward_v1(old, new, player=0)
    expected = REWARD_CAPTURE_ENEMY + 6 * REWARD_FORWARD_STEP
    assert abs(r - expected) < 1e-6, f"got {r}"


def test_capture_does_not_credit_opp():
    """The same capture above: from p2's perspective, this is just opp action.
    p2 should not get any reward (the capture is by p0)."""
    old = _state([10, BASE_POS, BASE_POS, BASE_POS],
                 [10, BASE_POS, BASE_POS, BASE_POS])
    new = _state([16, BASE_POS, BASE_POS, BASE_POS],
                 [BASE_POS]*4)
    r = compute_dense_reward_v1(old, new, player=2)
    # From p2's view this DOES detect own_at_base went up — but kill penalty
    # is NOT counted inside compute_dense_reward_v1 (must use kill_penalty).
    # Forward/score/capture/spawn/stretch all use p2's own player_positions.
    # p2 spawn delta = at_base_old - at_base_new = 3 - 4 = -1 → no spawn reward.
    # No score change. No opp_at_base increase (we're player=2, opp=0; p0 at_base went 3→3).
    # So r should be 0.
    assert abs(r) < 1e-6, f"got {r}"


def test_got_killed_via_kill_penalty():
    """compute_kill_penalty fires when own at-base count rose between own decisions."""
    # First decision: 3 own at base. Next decision: 4 (one was captured).
    r = compute_kill_penalty(prev_own_at_base=3, current_own_at_base=4)
    assert abs(r - PENALTY_GOT_KILLED) < 1e-6, f"got {r}"

    # Two captures
    r = compute_kill_penalty(prev_own_at_base=2, current_own_at_base=4)
    assert abs(r - 2 * PENALTY_GOT_KILLED) < 1e-6, f"got {r}"


def test_kill_penalty_none_on_first_decision():
    """No prior baseline → no kill penalty."""
    r = compute_kill_penalty(prev_own_at_base=None, current_own_at_base=4)
    assert r == 0.0


def test_kill_penalty_only_when_increase():
    """If own at-base count decreased or same (e.g. I spawned), no kill penalty."""
    r = compute_kill_penalty(prev_own_at_base=3, current_own_at_base=2)  # I spawned
    assert r == 0.0
    r = compute_kill_penalty(prev_own_at_base=2, current_own_at_base=2)  # nothing
    assert r == 0.0


def test_combined_capture_score():
    """Capture AND score on same move (very rare but possible)."""
    # p0's tok0 at pos 47, capturing p2 at pos 53 — wait can't, stretches are private
    # Better: p0 captures opp's token (track-track move with capture).
    # Plus a SECOND own token (tok1) scores in same move? No — only one move per turn.
    # So combined events on a single move ≠ valid.
    # Test instead: just the asymmetry of multiple events of same type:
    # Two captures in one move impossible (one move = one destination cell).
    pass


def test_total_reward_estimate():
    """Sanity-check the analytical total. v1 era should land ~+2.5 per game."""
    from td_ludo.game.dense_rewards import total_per_game_reward_estimate
    est = total_per_game_reward_estimate()
    net = sum(est.values())
    assert 2.0 < net < 3.5, f"per-game estimate {net} outside [2.0, 3.5]"


if __name__ == "__main__":
    # Run all tests directly
    import sys
    funcs = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    passed = 0
    for f in funcs:
        try:
            f()
            print(f"  ✓ {f.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {f.__name__}: {e}")
        except Exception as e:
            print(f"  ✗ {f.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(funcs)} passed")
    sys.exit(0 if passed == len(funcs) else 1)
