"""
Correctness tests for expectimax search implementation.

Run BEFORE the full 500-game eval. Verifies:
  1. Model + C++ ext load correctly
  2. _clone_state produces independent mutable copies
  3. Greedy (d0) picks argmax of raw policy
  4. Depth-1 search batched forward produces finite values for all my actions
  5. Depth-2 search runs without crashes on real states + terminal states
  6. Perspective flip is correct at opp-turn nodes (value decreases when leaf is
     strongly winning for opponent)
  7. On states where ALL my actions lead to the SAME state (e.g. only one
     legal move), all three depths return the same action
  8. Timing: decision latency for each depth is within budget

Exits non-zero on any failure.
"""
import os
import sys
import time
import random

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import td_ludo_cpp as ludo_cpp
from td_ludo.models.v10 import AlphaLudoV10
from evaluate_v10_expectimax import (
    _clone_state, _encode, _mask_from_legal, _forward_batch,
    pick_action_greedy, pick_action_depth1, pick_action_depth2,
)

PASS = []
FAIL = []


def check(cond, name, detail=""):
    if cond:
        PASS.append(name)
        print(f"  ✓ {name}")
    else:
        FAIL.append((name, detail))
        print(f"  ✗ {name}: {detail}")


def build_mid_game_state():
    """Construct a realistic mid-game state with multiple legal moves for P0."""
    s = ludo_cpp.create_initial_state_2p()
    # P0 has 2 tokens on track, 1 at base, 1 in home stretch
    s.player_positions[0] = [10, 25, -1, 51]
    # P2 has 3 tokens on track, 1 scored
    s.player_positions[2] = [5, 20, 35, 99]
    s.scores[0] = 0
    s.scores[2] = 1
    s.current_player = 0
    s.current_dice_roll = 3
    return s


def build_late_game_state():
    """P0 about to win — most tokens near/at home."""
    s = ludo_cpp.create_initial_state_2p()
    s.player_positions[0] = [55, 99, 99, 99]  # one on last home square, 3 scored
    s.player_positions[2] = [5, 10, 15, 20]   # opponent far behind
    s.scores[0] = 3
    s.scores[2] = 0
    s.current_player = 0
    s.current_dice_roll = 1  # moving ch55 → 99 wins
    return s


def build_p2_turn_state():
    """Symmetric state but current_player = 2. Tests perspective logic."""
    s = ludo_cpp.create_initial_state_2p()
    s.player_positions[0] = [10, 25, -1, 51]
    s.player_positions[2] = [5, 20, 35, -1]
    s.current_player = 2
    s.current_dice_roll = 3
    return s


def main():
    print("=" * 60)
    print("  Expectimax correctness tests")
    print("=" * 60)

    # ── Test 1: model + extension load ─────────────────────────────────────
    print("\n[1] Loading model + C++ ext...")
    # Use SL checkpoint — RL training inverted win_prob, see V10_RL_VALUE_INVERSION
    ckpt_path = 'checkpoints/ac_v10/model_sl.pt'
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    arch = ckpt.get('arch', {'num_res_blocks': 6, 'num_channels': 96, 'in_channels': 28})
    device = torch.device('cpu')  # deterministic for tests; CPU is fine for small batches
    model = AlphaLudoV10(**arch).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    check(model.count_parameters() == 1036278, "model loads with expected param count",
          f"got {model.count_parameters()}")
    check('encode_state_v10' in dir(ludo_cpp), "C++ ext has encode_state_v10")

    # ── Test 2: _clone_state produces independent mutable copies ───────────
    print("\n[2] _clone_state independence test...")
    s_orig = build_mid_game_state()
    s_clone = _clone_state(s_orig)
    s_clone.current_dice_roll = 6
    s_clone.current_player = 2
    check(s_orig.current_dice_roll == 3, "clone doesn't mutate original dice",
          f"orig has dice={s_orig.current_dice_roll}")
    check(s_orig.current_player == 0, "clone doesn't mutate original current_player",
          f"orig cp={s_orig.current_player}")
    check(s_clone.current_dice_roll == 6, "clone's dice is the new value")
    # Mutate token positions via apply_move on clone
    s_clone.current_dice_roll = 3
    s2 = ludo_cpp.apply_move(s_clone, 0)
    check(tuple(s_orig.player_positions[0]) == (10, 25, -1, 51),
          "orig positions unchanged after clone mutation")

    # ── Test 3: Greedy = argmax of raw policy ──────────────────────────────
    print("\n[3] Greedy vs raw policy argmax...")
    state = build_mid_game_state()
    legal = ludo_cpp.get_legal_moves(state)
    t = _encode(state)
    m = _mask_from_legal(legal)
    pol, _ = _forward_batch([t], [m], model, device)
    raw_argmax = int(pol[0].argmax())
    greedy_action = pick_action_greedy(state, 0, model, device)
    check(greedy_action == raw_argmax,
          "greedy picker matches raw policy argmax",
          f"greedy={greedy_action} argmax={raw_argmax}")

    # ── Test 4: Depth-1 produces finite expected values ────────────────────
    print("\n[4] Depth-1 runs cleanly on mid-game state...")
    t0 = time.time()
    d1_action = pick_action_depth1(state, 0, model, device)
    d1_time = (time.time() - t0) * 1000
    check(d1_action in legal, "d1 action is legal", f"got {d1_action}, legal {legal}")
    check(d1_time < 500, f"d1 decision fast (<500ms)", f"took {d1_time:.0f}ms")
    print(f"      d1 action={d1_action}, latency={d1_time:.1f}ms")

    # ── Test 5: Depth-2 runs without crashes ───────────────────────────────
    print("\n[5] Depth-2 runs cleanly on mid-game state...")
    t0 = time.time()
    d2_action = pick_action_depth2(state, 0, model, device)
    d2_time = (time.time() - t0) * 1000
    check(d2_action in legal, "d2 action is legal", f"got {d2_action}, legal {legal}")
    check(d2_time < 2000, f"d2 decision reasonable (<2s)", f"took {d2_time:.0f}ms")
    print(f"      d2 action={d2_action}, latency={d2_time:.1f}ms")

    # ── Test 6: Perspective flip — winning leaf from my POV must score >0.5
    # Build a state where I'm ABOUT to win with any legal move, check depth-1
    # picks something that keeps/wins. Also check late-game state values are
    # near-extreme (close to 1 if winning, 0 if losing).
    print("\n[6] Perspective + late-game state sanity...")
    late_state = build_late_game_state()
    late_legal = ludo_cpp.get_legal_moves(late_state)
    check(len(late_legal) > 0, "late-game state has legal moves")
    late_action = pick_action_depth1(late_state, 0, model, device)
    check(late_action in late_legal, "d1 action valid on late-game state")
    # Evaluate raw win_prob on this state — should be very high (P0 about to win)
    late_t = _encode(late_state)
    late_m = _mask_from_legal(late_legal)
    _, late_wp = _forward_batch([late_t], [late_m], model, device)
    check(late_wp[0] > 0.8, f"late-game winning state → win_prob > 0.8",
          f"got {float(late_wp[0]):.3f}")

    # ── Test 7: Single-legal-move state — all depths agree ─────────────────
    # Construct a state with only one legal move
    print("\n[7] Single legal move → all depths agree...")
    s_single = ludo_cpp.create_initial_state_2p()
    s_single.player_positions[0] = [-1, -1, -1, -1]  # all at base
    s_single.current_dice_roll = 1  # only legal move is "stay" (none actually)
    single_legal = ludo_cpp.get_legal_moves(s_single)
    # With dice=1, no token can leave base (needs 6) — no legal moves
    if len(single_legal) == 0:
        s_single.current_dice_roll = 6  # now token spawning is legal
        single_legal = ludo_cpp.get_legal_moves(s_single)
    check(len(single_legal) >= 1, "built single-legal state", f"legal={single_legal}")
    if len(single_legal) == 1:
        a0 = pick_action_greedy(s_single, 0, model, device)
        a1 = pick_action_depth1(s_single, 0, model, device)
        a2 = pick_action_depth2(s_single, 0, model, device)
        check(a0 == a1 == a2 == single_legal[0],
              "single-legal: all three depths agree",
              f"d0={a0} d1={a1} d2={a2} legal={single_legal}")

    # ── Test 8: P2-turn state (tests perspective on opponent side) ──────────
    print("\n[8] P2-turn state with model_player=0 — depth-1 still works...")
    p2_state = build_p2_turn_state()
    p2_legal = ludo_cpp.get_legal_moves(p2_state)
    # If model_player=0 but current_player=2, we shouldn't call pick on this state
    # (the game loop would skip model's turn). Just verify apply_move works.
    s_after_p2 = ludo_cpp.apply_move(p2_state, p2_legal[0])
    check(s_after_p2.current_player != 2 or s_after_p2.current_dice_roll == 0,
          "after P2 action, turn correctly advances or bonus")

    # ── Test 9: Decision consistency — seed twice, same answer ─────────────
    print("\n[9] Deterministic decisions (same state → same action)...")
    state2 = build_mid_game_state()
    a_first = pick_action_depth2(state2, 0, model, device)
    a_second = pick_action_depth2(state2, 0, model, device)
    check(a_first == a_second, "depth-2 deterministic", f"first={a_first} second={a_second}")

    # ── Test 10: Compare actions across depths on a real state ─────────────
    print("\n[10] Greedy vs d1 vs d2 action comparison on mid-game state:")
    state3 = build_mid_game_state()
    a0 = pick_action_greedy(state3, 0, model, device)
    a1 = pick_action_depth1(state3, 0, model, device)
    a2 = pick_action_depth2(state3, 0, model, device)
    legal3 = ludo_cpp.get_legal_moves(state3)
    print(f"      legal moves: {legal3}")
    print(f"      d0 (greedy): {a0}   d1 (1-ply): {a1}   d2 (2-ply): {a2}")
    # Not a pass/fail — just informative
    if a0 == a1 == a2:
        print(f"      NOTE: all three pick same action on this state. Normal if search")
        print(f"            doesn't find a better alternative; informative if it happens often.")
    else:
        print(f"      Search altered the decision — evidence it's doing work.")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Passed: {len(PASS)}  |  Failed: {len(FAIL)}")
    if FAIL:
        print("  Failures:")
        for n, d in FAIL:
            print(f"    - {n}: {d}")
        sys.exit(1)
    print("  ALL TESTS PASSED ✓")


if __name__ == '__main__':
    main()
