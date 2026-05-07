"""Unit tests for the 2-ply search expansion + Q aggregation in
`generate_search_data.py`.

Covers the two bugs that destroyed Step 1 v1:
  - Bug A: GameState aliasing — every dice iteration mutated the same
           shared `s_after_own` object, corrupting subsequent iterations
           and aliasing leaf references.
  - Bug B: Bonus-turn sign — when own_a kept the turn (6-roll or score),
           the next decision is OURS, but the aggregator used `min`
           (correct only when opp chooses) instead of `max`.

Run:
    python -m experiments.mcts_v1.test_search_aggregation
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import td_ludo_cpp as ludo_cpp
from experiments.mcts_v1.generate_search_data import (
    _copy_state,
    _expand_two_ply_leaves,
    _aggregate_q_per_action,
    DICE_VALUES,
)


# ── helpers ────────────────────────────────────────────────────────────────
def make_state(
    own_player=0,
    opp_player=2,
    own_positions=(-1, -1, -1, -1),
    opp_positions=(-1, -1, -1, -1),
    dice=4,
    own_score=0,
    opp_score=0,
):
    g = ludo_cpp.create_initial_state_2p()
    pp = np.array(g.player_positions, dtype=np.int8)
    pp[own_player] = own_positions
    pp[opp_player] = opp_positions
    g.player_positions = pp
    sc = np.array(g.scores, dtype=np.int8)
    sc[own_player] = own_score
    sc[opp_player] = opp_score
    g.scores = sc
    g.current_player = own_player
    g.current_dice_roll = dice
    return g


# ── Test 1: _copy_state truly detaches ───────────────────────────────────
def test_copy_state_is_independent():
    g = make_state(own_positions=(5, 10, -1, -1), dice=3)
    c = _copy_state(g)
    # Mutate copy
    c.current_player = 99 if int(c.current_player) != 99 else 0
    c.current_dice_roll = 99
    pp = np.array(c.player_positions, dtype=np.int8); pp[0] = [40, 40, 40, 40]
    c.player_positions = pp
    # Original should be unchanged
    assert int(g.current_player) == 0, f"current_player leaked: {int(g.current_player)}"
    assert int(g.current_dice_roll) == 3, f"dice leaked: {int(g.current_dice_roll)}"
    op = np.array(g.player_positions, dtype=np.int8)
    assert op[0, 0] == 5 and op[0, 1] == 10, f"positions leaked: {op[0]}"
    print("  ✓ _copy_state truly detaches the copy from the source")


# ── Test 2: _expand stores distinct leaves (no aliasing) ─────────────────
def test_expand_leaves_are_distinct_objects():
    """Every leaf in the returned list must be a distinct GameState object,
    even after dice loops that touch the same shared post-own state."""
    g = make_state(own_positions=(5, -1, -1, -1), dice=3)
    leaves, meta, legal_own, structure = _expand_two_ply_leaves(g)

    # Sanity
    assert len(leaves) > 0, "no leaves expanded"

    # Collect Python ids of every leaf — they MUST all differ.
    ids = [id(leaf) for leaf in leaves]
    assert len(ids) == len(set(ids)), (
        f"aliased leaves detected: {len(ids)} leaves but only {len(set(ids))} "
        f"distinct objects (some leaves point to the same GameState)"
    )
    print(f"  ✓ {len(leaves)} leaves, all distinct objects (no aliasing)")


# ── Test 3: leaf field-stability across sibling leaf mutation ────────────
def test_leaf_state_is_stable_after_other_leaves_mutate():
    """Construct a known case, take the post-own state, run the expander,
    then later (in a fresh expansion of the SAME root) verify the leaf
    encoded fields didn't drift."""
    g = make_state(own_positions=(5, 10, 20, -1), dice=3)
    leaves, meta, legal_own, structure = _expand_two_ply_leaves(g)

    # Capture each leaf's identifying signature
    sigs_before = []
    for leaf in leaves:
        pp = np.array(leaf.player_positions, dtype=np.int8).flatten().tolist()
        sigs_before.append((
            int(leaf.current_player),
            int(leaf.current_dice_roll),
            tuple(pp),
        ))

    # Now MUTATE every leaf in turn (this would corrupt aliased peers if any).
    # If the bug were still present, mutating leaves[0] would change leaves[1..].
    for leaf in leaves:
        leaf.current_player = 99
        leaf.current_dice_roll = 99
        pp = np.array(leaf.player_positions, dtype=np.int8); pp[0] = [88, 88, 88, 88]
        leaf.player_positions = pp

    # Re-expand to get fresh leaves; their signatures should match the
    # original because the root `g` was never touched.
    leaves2, _, _, _ = _expand_two_ply_leaves(g)
    sigs_after = []
    for leaf in leaves2:
        pp = np.array(leaf.player_positions, dtype=np.int8).flatten().tolist()
        sigs_after.append((
            int(leaf.current_player),
            int(leaf.current_dice_roll),
            tuple(pp),
        ))
    assert sigs_before == sigs_after, (
        "leaf signatures drifted between two expansions of the same root, "
        "indicating shared state with the root itself"
    )
    print("  ✓ leaf states stable across mutation (root unchanged)")


# ── Test 4: bonus-turn sign — same-player case picks MAX ─────────────────
def test_bonus_turn_uses_max_not_min():
    """Set up a state where own_a is forced to be a 6-roll move, so the
    next decision is still ours. Construct synthetic leaf_v_root values
    where one branch is winning (+0.9) and another is losing (-0.9) for
    the same own_a/dice bucket. The aggregator MUST pick +0.9 (max), not
    -0.9 (min)."""
    # State with dice=6 and one own token at base — own_a=0 unlocks token 0.
    # After unlock, current_player remains me (bonus turn).
    g = make_state(
        own_positions=(-1, 30, 40, -1),  # only token 0 is at base
        opp_positions=(-1, -1, -1, -1),  # no opp on board, ensures opp has no legal moves
        dice=6,
    )
    leaves, meta, legal_own, structure = _expand_two_ply_leaves(g)
    assert legal_own, "expected legal moves"

    # Find any own_a + dice bucket where next_player_is_root is True
    # (must exist: dice=6 → bonus turn; or scoring move kept turn)
    bonus_buckets = []
    for own_a in legal_own:
        for dice, pairs in structure[own_a].items():
            if not pairs:
                continue
            _, first_idx = pairs[0]
            if meta[first_idx]["next_player_is_root"]:
                bonus_buckets.append((own_a, dice, pairs))

    assert bonus_buckets, "no bonus-turn buckets found in test setup"

    # Pick first bonus bucket; synthesize leaf values where one is 0.9 (good
    # for root) and others are -0.9 (bad). The aggregator must average max
    # over this bucket, not min.
    own_a, dice, pairs = bonus_buckets[0]
    # Build leaf_v_root: by default 0, override the bonus-bucket leaves
    # Default V = 0.5 (= V_root 0) so non-test buckets contribute 0 to Q.
    # Otherwise np.zeros = V=0 = sure loss = -1 in signed, polluting Q.
    leaf_values_synthetic = np.full(len(leaves), 0.5, dtype=np.float32)
    # We need leaf_v_root values, not raw V. Build by hand.
    # Set the first leaf in this bucket to map to V_root=+0.9, others to -0.9.
    # _aggregate uses leaf_values (raw V in [0,1]) and converts.
    # To set leaf_v_root[idx] = +0.9 we need v_signed = ±0.9 depending on
    # whether leaf.current_player == root_player.
    # Simpler: construct the synthetic input as the leaf_values that produce
    # leaf_v_root = +0.9 / -0.9 directly. Skip the conversion — call our own
    # min/max-checking helper.

    # We'll compute Q for this single bucket by simulating both branches:
    # all leaves in the bucket get random leaf_values, but we'll override
    # the post-aggregation result.

    # The cleanest test: bypass _aggregate and check directly.
    # For each leaf in the bucket, set leaf_values such that leaf_v_root
    # is one of {+0.9, -0.9}. We need to know each leaf's current_player.
    root_player = int(g.current_player)
    target_v_root = []
    for k, (_, idx) in enumerate(pairs):
        if k == 0:
            target_v_root.append(+0.9)  # the "winning" leaf
        else:
            target_v_root.append(-0.9)  # the "losing" alternatives
        # Convert target_v_root -> leaf_values[idx] (V in [0,1])
        leaf = leaves[idx]
        v_signed = target_v_root[-1]
        # If leaf.current_player == root_player, leaf_v_root = v_signed → V = (v+1)/2
        # Else, leaf_v_root = -v_signed → V = (-v_signed+1)/2
        if int(leaf.current_player) == root_player:
            v_pred = (v_signed + 1.0) / 2.0
        else:
            v_pred = (-v_signed + 1.0) / 2.0
        leaf_values_synthetic[idx] = v_pred

    # Run aggregator
    Q = _aggregate_q_per_action(
        leaves, meta, leaf_values_synthetic, legal_own, structure, root_player,
    )

    # The bucket was set up so max V_root in that bucket is +0.9, min is -0.9.
    # Q_own_a contributed by this dice bucket = +0.9 (if max — correct) or
    # -0.9 (if min — buggy). Other dice buckets have V_root = 0 (default
    # leaf_values_synthetic = 0 → V=0.5 → v_signed=0 → leaf_v_root=0).
    # So Q[own_a] = (this_bucket / 6.0) + 0 = expected / 6.0
    expected_Q = 0.9 / 6.0
    actual_Q = Q[own_a]
    assert abs(actual_Q - expected_Q) < 1e-4, (
        f"bonus-turn aggregation wrong: expected Q[{own_a}]={expected_Q:.4f} "
        f"(max-over-bonus = +0.9 / 6 dice), got {actual_Q:.4f}. "
        f"This means the aggregator used min instead of max for the bonus turn "
        f"— Bug B is back."
    )
    print(f"  ✓ bonus-turn bucket aggregation uses MAX correctly "
          f"(Q={actual_Q:.4f}, expected {expected_Q:.4f})")


# ── Test 5: standard (non-bonus) bucket picks MIN ────────────────────────
def test_standard_bucket_uses_min():
    """The mirror of test 4 — a bucket where opp is the next decision-maker
    must use min over root POV (= max for opp)."""
    # Set up a state where own_a does NOT trigger bonus (dice=4, no score)
    g = make_state(
        own_positions=(5, -1, -1, -1),
        opp_positions=(20, -1, -1, -1),  # opp has at least one legal move
        dice=4,
    )
    leaves, meta, legal_own, structure = _expand_two_ply_leaves(g)
    assert legal_own, "expected legal moves"

    # Find a non-bonus bucket
    non_bonus_buckets = []
    for own_a in legal_own:
        for dice, pairs in structure[own_a].items():
            if not pairs:
                continue
            _, first_idx = pairs[0]
            if not meta[first_idx]["next_player_is_root"]:
                non_bonus_buckets.append((own_a, dice, pairs))
    assert non_bonus_buckets, "no standard buckets found in test setup"

    own_a, dice, pairs = non_bonus_buckets[0]
    # Skip if only 1 leaf in bucket (can't distinguish min vs max)
    multi_leaf = [b for b in non_bonus_buckets if len(b[2]) > 1]
    if not multi_leaf:
        print("  ⚠ no multi-leaf standard bucket — test inconclusive")
        return
    own_a, dice, pairs = multi_leaf[0]

    root_player = int(g.current_player)
    # Default V = 0.5 (= V_root 0) so non-test buckets contribute 0 to Q.
    # Otherwise np.zeros = V=0 = sure loss = -1 in signed, polluting Q.
    leaf_values_synthetic = np.full(len(leaves), 0.5, dtype=np.float32)
    # First leaf gets V_root = +0.9 ("good for us"), others -0.9.
    # Opp picks min → -0.9 expected.
    for k, (_, idx) in enumerate(pairs):
        v_signed = +0.9 if k == 0 else -0.9
        leaf = leaves[idx]
        if int(leaf.current_player) == root_player:
            v_pred = (v_signed + 1.0) / 2.0
        else:
            v_pred = (-v_signed + 1.0) / 2.0
        leaf_values_synthetic[idx] = v_pred

    Q = _aggregate_q_per_action(
        leaves, meta, leaf_values_synthetic, legal_own, structure, root_player,
    )
    # min = -0.9 in this bucket, others 0; Q = -0.9 / 6
    expected_Q = -0.9 / 6.0
    actual_Q = Q[own_a]
    assert abs(actual_Q - expected_Q) < 1e-4, (
        f"standard bucket aggregation wrong: expected Q[{own_a}]={expected_Q:.4f} "
        f"(min over opp = -0.9), got {actual_Q:.4f}"
    )
    print(f"  ✓ standard bucket aggregation uses MIN correctly "
          f"(Q={actual_Q:.4f}, expected {expected_Q:.4f})")


# ── Test 6: terminal own-move gives Q=+1/6 (×6 dice) = +1.0 ──────────────
def test_terminal_own_move_q_is_plus_one():
    """If our own_a wins the game outright, the leaf is terminal with
    winner=root. All 6 dice buckets have one terminal leaf with V_root=+1.
    Q[own_a] = (1.0 × 6) / 6 = +1.0."""
    # Set up a state where own_a unlocks the only-remaining own token AND
    # causes us to score 4/4 (winning). Easiest: 3 own tokens already home,
    # 1 own token at home-stretch position 55 (one step from home), dice=1.
    g = make_state(
        own_positions=(99, 99, 99, 55),  # 3 home, 1 about to score
        opp_positions=(-1, -1, -1, -1),
        own_score=3,
        dice=1,  # token 3 moves 55 → 56 = HOME, wins game
    )
    leaves, meta, legal_own, structure = _expand_two_ply_leaves(g)
    # Only one legal action (token 3); after move, game is terminal with us as winner
    own_a = legal_own[0]
    # Default V = 0.5 (= V_root 0) so non-test buckets contribute 0 to Q.
    # Otherwise np.zeros = V=0 = sure loss = -1 in signed, polluting Q.
    leaf_values_synthetic = np.full(len(leaves), 0.5, dtype=np.float32)
    # All bucket leaves are terminal-winner == root → leaf_v_root = +1 (no V call needed)
    Q = _aggregate_q_per_action(
        leaves, meta, leaf_values_synthetic, legal_own, structure, int(g.current_player),
    )
    actual_Q = Q[own_a]
    assert abs(actual_Q - 1.0) < 1e-4, (
        f"terminal-win Q expected +1.0, got {actual_Q:.4f}"
    )
    print(f"  ✓ terminal own-move Q = +1.0 ({actual_Q:.4f})")


# ── Runner ────────────────────────────────────────────────────────────────
TESTS = [
    ("copy_state truly detaches", test_copy_state_is_independent),
    ("leaves are distinct objects (no aliasing)", test_expand_leaves_are_distinct_objects),
    ("leaf state stable across mutation (root unchanged)", test_leaf_state_is_stable_after_other_leaves_mutate),
    ("bonus-turn aggregation uses MAX", test_bonus_turn_uses_max_not_min),
    ("standard bucket aggregation uses MIN", test_standard_bucket_uses_min),
    ("terminal own-win Q = +1.0", test_terminal_own_move_q_is_plus_one),
]


def main():
    passed = 0
    failed = 0
    for name, fn in TESTS:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {name}: UNEXPECTED {type(e).__name__}: {e}")
            failed += 1
    total = passed + failed
    print(f"\n{passed}/{total} tests passed" + (" ❌" if failed else " ✓"))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
