#!/usr/bin/env python3
"""
Debug script to trace token cutting behavior in detail.
This helps verify that the arriving player cuts the stationary player.
"""
import sys
sys.path.insert(0, 'src')

import ludo_cpp
import numpy as np

def print_state(state, label=""):
    print(f"\n{'='*60}")
    print(f"STATE: {label}")
    print(f"{'='*60}")
    print(f"Current Player: P{state.current_player}")
    print(f"Dice Roll: {state.current_dice_roll}")
    print(f"Terminal: {state.is_terminal}")
    print()
    for p in range(4):
        positions = [state.player_positions[p][t] for t in range(4)]
        abs_positions = []
        for pos in positions:
            if pos == -1:
                abs_positions.append("BASE")
            elif pos == 99:
                abs_positions.append("HOME")
            elif pos <= 50:
                abs_pos = (pos + p * 13) % 52
                abs_positions.append(f"{pos}(abs:{abs_pos})")
            else:
                abs_positions.append(f"HR{pos-50}")
        print(f"  P{p}: {abs_positions} | Score: {state.scores[p]}")
    print()


def test_cut_scenario_1():
    """
    Scenario: P1 moves to a position where P0 already has a token.
    Expected: P0's token gets cut (sent to base).
    """
    print("\n" + "="*70)
    print("SCENARIO 1: P1 arrives at P0's position -> P0 should be CUT")
    print("="*70)
    
    state = ludo_cpp.create_initial_state()
    
    # P0 has a token at relative position 1 (absolute position 1)
    state.player_positions[0][0] = 1
    
    # P1 has a token at relative position 39
    # P1's absolute position = (39 + 1*13) % 52 = 52 % 52 = 0
    # Wait, that's position 0, not 1!
    
    # To land on absolute 1, P1 needs relative position where:
    # (rel + 13) % 52 = 1
    # rel = (1 - 13) % 52 = -12 % 52 = 40
    state.player_positions[1][0] = 40  # This is at absolute position 1
    
    # But wait, if P1 is at 40 and we want them to MOVE to 1...
    # Let's put P1 at 39 and have them roll 1 to land at 40 (abs 1)
    # (39 + 13) % 52 = 52 % 52 = 0
    # (40 + 13) % 52 = 53 % 52 = 1 ✓
    
    state.player_positions[1][0] = 39  # At absolute 0
    state.current_player = 1
    state.current_dice_roll = 1  # Will move to 40 (abs 1)
    
    print_state(state, "BEFORE P1's move")
    
    # P1 rolls 1, moving from rel 39 to rel 40
    moves = ludo_cpp.get_legal_moves(state)
    print(f"Legal moves for P1: {moves}")
    
    if 0 not in moves:
        print("ERROR: Token 0 should be a legal move!")
        return False
    
    next_state = ludo_cpp.apply_move(state, 0)
    
    print_state(next_state, "AFTER P1's move")
    
    # Verify results
    p0_pos = next_state.player_positions[0][0]
    p1_pos = next_state.player_positions[1][0]
    
    print(f"P0 Token 0: {p0_pos} (expected: -1/BASE)")
    print(f"P1 Token 0: {p1_pos} (expected: 40)")
    
    if p0_pos == -1 and p1_pos == 40:
        print("✅ CORRECT! P0 was cut, P1 is at new position.")
        return True
    else:
        print("❌ WRONG! Cut logic is broken!")
        return False


def test_cut_scenario_2():
    """
    Scenario: P0 moves to a position where P1 already has a token.
    Expected: P1's token gets cut (sent to base).
    """
    print("\n" + "="*70)
    print("SCENARIO 2: P0 arrives at P1's position -> P1 should be CUT")
    print("="*70)
    
    state = ludo_cpp.create_initial_state()
    
    # P1 has token at relative 40 = absolute (40+13)%52 = 1
    state.player_positions[1][0] = 40
    
    # P0 starts at position 0 and rolls 1 to land at position 1
    state.player_positions[0][0] = 0
    state.current_player = 0
    state.current_dice_roll = 1
    
    print_state(state, "BEFORE P0's move")
    
    moves = ludo_cpp.get_legal_moves(state)
    print(f"Legal moves for P0: {moves}")
    
    next_state = ludo_cpp.apply_move(state, 0)
    
    print_state(next_state, "AFTER P0's move")
    
    # Verify results
    p0_pos = next_state.player_positions[0][0]
    p1_pos = next_state.player_positions[1][0]
    
    print(f"P0 Token 0: {p0_pos} (expected: 1)")
    print(f"P1 Token 0: {p1_pos} (expected: -1/BASE)")
    
    if p0_pos == 1 and p1_pos == -1:
        print("✅ CORRECT! P1 was cut, P0 is at new position.")
        return True
    else:
        print("❌ WRONG! Cut logic is broken!")
        return False


def test_safe_zone():
    """
    Scenario: P0 moves to a SAFE position where P1 is.
    Expected: No cut happens (safe zones protect).
    """
    print("\n" + "="*70)
    print("SCENARIO 3: P0 arrives at P1's SAFE position -> No cut")
    print("="*70)
    
    state = ludo_cpp.create_initial_state()
    
    # Safe positions (absolute): 0, 8, 13, 21, 26, 34, 39, 47
    # P1 at relative position that maps to absolute 8 (a safe zone)
    # (rel + 13) % 52 = 8 -> rel = (8 - 13) % 52 = -5 % 52 = 47
    state.player_positions[1][0] = 47  # Absolute 8 (safe)
    
    # P0 at position 7, rolls 1 to land at 8 (safe)
    state.player_positions[0][0] = 7
    state.current_player = 0
    state.current_dice_roll = 1
    
    print_state(state, "BEFORE P0's move")
    
    next_state = ludo_cpp.apply_move(state, 0)
    
    print_state(next_state, "AFTER P0's move")
    
    p0_pos = next_state.player_positions[0][0]
    p1_pos = next_state.player_positions[1][0]
    
    print(f"P0 Token 0: {p0_pos} (expected: 8)")
    print(f"P1 Token 0: {p1_pos} (expected: 47, NOT cut)")
    
    if p0_pos == 8 and p1_pos == 47:
        print("✅ CORRECT! Safe zone protected P1.")
        return True
    else:
        print("❌ WRONG! Safe zone logic is broken!")
        return False


if __name__ == "__main__":
    print("="*70)
    print("TOKEN CUTTING DEBUG TEST")
    print("="*70)
    
    results = []
    results.append(("Scenario 1: Arriving player cuts stationary", test_cut_scenario_1()))
    results.append(("Scenario 2: Reverse direction cut", test_cut_scenario_2()))
    results.append(("Scenario 3: Safe zone protection", test_safe_zone()))
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("All tests passed! Cut logic is correct.")
    else:
        print("Some tests failed! Investigate the issues above.")
    
    sys.exit(0 if all_passed else 1)
