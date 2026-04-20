"""
Smoke test for V10 encoder — verify every channel produces correct values
across a variety of hand-crafted game states.

This is a must-run before training: if any channel has buggy values, we'd
train the model on bad data and burn time.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import td_ludo_cpp as ludo_cpp


BOARD = 15
CELLS = BOARD * BOARD
EXPECTED_CHANNELS = 28

CHANNEL_NAMES = [
    "0: My Token 0", "1: My Token 1", "2: My Token 2", "3: My Token 3",
    "4: Opp Density", "5: Safe Zones", "6: My Home Path", "7: Opp Home Path",
    "8: Score Diff", "9: My Locked %", "10: Opp Locked %",
    "11: Dice=1", "12: Dice=2", "13: Dice=3", "14: Dice=4", "15: Dice=5", "16: Dice=6",
    "17: Opp Token 0", "18: Opp Token 1", "19: Opp Token 2", "20: Opp Token 3",
    "21: Danger Map", "22: Capture Opp Map", "23: Safe Landing",
    "24: Bonus Turn Flag", "25: Two-Roll Capture",
    "26: non_home_tokens_frac (NEW)", "27: my_leader_progress (NEW)",
]

PASSED = []
FAILED = []


def check(cond, name, detail=""):
    if cond:
        PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


def make_state(p0_pos=(-1, -1, -1, -1), p2_pos=(-1, -1, -1, -1),
               cp=0, dice=0, p0_score=0, p2_score=0):
    """Build a state from explicit positions."""
    state = ludo_cpp.create_initial_state_2p()
    state.player_positions[0] = list(p0_pos)
    state.player_positions[2] = list(p2_pos)
    state.scores[0] = p0_score
    state.scores[2] = p2_score
    state.current_player = cp
    state.current_dice_roll = dice
    return state


def encode(state):
    return np.array(ludo_cpp.encode_state_v10(state))


# ═══════════════════════════════════════════════════════════════
# Test 1: Shape and basic sanity
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 1: Shape + basic sanity")
print("=" * 60)
state = ludo_cpp.create_initial_state_2p()
t = encode(state)
check(t.shape == (28, 15, 15), "shape == (28, 15, 15)", f"got {t.shape}")
check(not np.any(np.isnan(t)), "no NaN values")
check(not np.any(np.isinf(t)), "no Inf values")
check((t >= 0).all(), "all values >= 0")


# ═══════════════════════════════════════════════════════════════
# Test 2: Token position channels (ch 0-3)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 2: Token position channels (ch 0-3)")
print("=" * 60)

# Each of player 0's 4 tokens at base should light exactly 1 cell in ch 0-3
state = make_state(p0_pos=(-1, -1, -1, -1), cp=0)
t = encode(state)
for i in range(4):
    check(int(t[i].sum()) == 1,
          f"ch{i} (my token {i}) lights 1 base cell",
          f"sum={t[i].sum()}")

# Put token 0 on the main track at pos 5
state = make_state(p0_pos=(5, -1, -1, -1), cp=0)
t = encode(state)
check(int(t[0].sum()) == 1, "ch0 token 0 on main track lights 1 cell")
check(int(t[1].sum()) == 1 and int(t[2].sum()) == 1 and int(t[3].sum()) == 1,
      "ch1-3 still light 1 base cell each")


# ═══════════════════════════════════════════════════════════════
# Test 3: Dice one-hot (ch 11-16)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 3: Dice one-hot (ch 11-16)")
print("=" * 60)
for d in range(1, 7):
    state = make_state(cp=0, dice=d)
    t = encode(state)
    # Channel for this dice should be all-1, others should be 0
    dice_ch = 10 + d
    check(t[dice_ch].sum() == CELLS,
          f"dice={d} lights ch{dice_ch} fully",
          f"sum={t[dice_ch].sum()}")
    for other_d in range(1, 7):
        if other_d == d:
            continue
        other_ch = 10 + other_d
        check(t[other_ch].sum() == 0,
              f"dice={d}: ch{other_ch} empty",
              f"sum={t[other_ch].sum()}")


# ═══════════════════════════════════════════════════════════════
# Test 4: Bonus turn flag (ch 24)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 4: Bonus turn flag (ch 24)")
print("=" * 60)
for d in [1, 2, 3, 4, 5]:
    state = make_state(cp=0, dice=d)
    t = encode(state)
    check(t[24].sum() == 0, f"dice={d}: ch24 (bonus) empty",
          f"sum={t[24].sum()}")
state = make_state(cp=0, dice=6)
t = encode(state)
check(t[24].sum() == CELLS, "dice=6: ch24 fully lit",
      f"sum={t[24].sum()}")


# ═══════════════════════════════════════════════════════════════
# Test 5: NEW ch 26 non_home_tokens_frac
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 5: ch 26 non_home_tokens_frac (NEW)")
print("=" * 60)

# All 4 tokens at base — not scored, count=4, frac=1.0
state = make_state(p0_pos=(-1, -1, -1, -1), cp=0)
t = encode(state)
check(abs(t[26].sum() - CELLS * 1.0) < 0.01,
      "4 non-home tokens → ch26 == 1.0 broadcast",
      f"sum={t[26].sum()}, expected {CELLS}")

# 3 at base, 1 scored (99) — frac=3/4=0.75
state = make_state(p0_pos=(99, -1, -1, -1), cp=0)
t = encode(state)
check(abs(t[26].sum() - CELLS * 0.75) < 0.5,
      "3 non-home, 1 scored → ch26 == 0.75",
      f"sum={t[26].sum()}, expected {CELLS * 0.75}")

# 2 scored, 2 on board — frac=2/4=0.5
state = make_state(p0_pos=(99, 99, 10, -1), cp=0)
t = encode(state)
check(abs(t[26].sum() - CELLS * 0.5) < 0.5,
      "2 scored, 2 remaining → ch26 == 0.5",
      f"sum={t[26].sum()}, expected {CELLS * 0.5}")

# 3 scored, 1 remaining — frac=1/4=0.25 (FORCED MODE!)
state = make_state(p0_pos=(99, 99, 99, 5), cp=0)
t = encode(state)
check(abs(t[26].sum() - CELLS * 0.25) < 0.5,
      "3 scored, 1 remaining → ch26 == 0.25 (FORCED MODE)",
      f"sum={t[26].sum()}, expected {CELLS * 0.25}")

# All 4 scored — frac=0
state = make_state(p0_pos=(99, 99, 99, 99), cp=0)
t = encode(state)
check(t[26].sum() < 0.01, "4 scored → ch26 == 0", f"sum={t[26].sum()}")


# ═══════════════════════════════════════════════════════════════
# Test 6: NEW ch 27 my_leader_progress
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 6: ch 27 my_leader_progress (NEW)")
print("=" * 60)

# All at base — progress 0
state = make_state(p0_pos=(-1, -1, -1, -1), cp=0)
t = encode(state)
check(t[27].sum() < 0.01, "all base → ch27 == 0", f"sum={t[27].sum()}")

# Leader at pos 28 (halfway-ish) — progress ≈ 28/56 = 0.5
state = make_state(p0_pos=(28, -1, -1, -1), cp=0)
t = encode(state)
expected = 28.0 / 56.0 * CELLS
check(abs(t[27].sum() - expected) < 1.0,
      f"pos 28 → ch27 ≈ 28/56 = 0.5",
      f"sum={t[27].sum()}, expected {expected:.1f}")

# Leader at pos 55 (last home run cell) — progress ≈ 55/56 ≈ 0.982
state = make_state(p0_pos=(55, 10, -1, -1), cp=0)
t = encode(state)
expected = 55.0 / 56.0 * CELLS
check(abs(t[27].sum() - expected) < 1.0,
      "pos 55 → ch27 ≈ 55/56",
      f"sum={t[27].sum()}, expected {expected:.1f}")

# One token home (99) — progress = 1.0
state = make_state(p0_pos=(99, -1, -1, -1), cp=0)
t = encode(state)
check(abs(t[27].sum() - CELLS) < 0.01,
      "pos 99 → ch27 == 1.0 broadcast",
      f"sum={t[27].sum()}")

# Leader should be the most-advanced token, not the first
state = make_state(p0_pos=(5, 10, 30, -1), cp=0)
t = encode(state)
expected = 30.0 / 56.0 * CELLS
check(abs(t[27].sum() - expected) < 1.0,
      "multiple tokens, leader at 30 → ch27 ≈ 30/56",
      f"sum={t[27].sum()}, expected {expected:.1f}")


# ═══════════════════════════════════════════════════════════════
# Test 7: Perspective rotation (player 2's view should mirror player 0)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 7: Perspective rotation (cp=0 vs cp=2 symmetric)")
print("=" * 60)

# Set up symmetric state: each player has the same relative token distribution
state_p0 = make_state(p0_pos=(10, -1, -1, -1), p2_pos=(20, -1, -1, -1), cp=0)
state_p2 = make_state(p0_pos=(20, -1, -1, -1), p2_pos=(10, -1, -1, -1), cp=2)
t0 = encode(state_p0)
t2 = encode(state_p2)

# Both views should have ch 0 (my token 0) lit at relatively same position
# (from each player's own rotated perspective). We don't check exact cells —
# just that both encoders produce the same per-channel SUMS:
for ch in range(28):
    check(abs(t0[ch].sum() - t2[ch].sum()) < 0.01,
          f"ch{ch:2d} has same sum from both perspectives",
          f"t0={t0[ch].sum():.2f} vs t2={t2[ch].sum():.2f}")


# ═══════════════════════════════════════════════════════════════
# Test 8: Legacy channels match V6.3 (for channels 0-24)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 8: V10 ch 0-24 matches V6.3 ch 0-24 (same logic)")
print("=" * 60)

# Compare on a varied state
state = make_state(p0_pos=(5, 30, -1, 99), p2_pos=(40, 10, -1, -1), cp=0, dice=6)
v10 = encode(state)
v6_3 = np.array(ludo_cpp.encode_state_v6_3(state, 0))

# Channels 0-24 should match exactly
for ch in range(25):
    diff = np.abs(v10[ch] - v6_3[ch]).max()
    check(diff < 1e-5,
          f"ch{ch:2d} V10 matches V6.3 (except ch25+)",
          f"max diff={diff}")

# V10 ch 25 should match V6.3 ch 26 (two_roll_capture, shifted down)
diff = np.abs(v10[25] - v6_3[26]).max()
check(diff < 1e-5,
      "V10 ch25 (two_roll) == V6.3 ch26 (two_roll)",
      f"max diff={diff}")


# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  PASSED: {len(PASSED)}")
print(f"  FAILED: {len(FAILED)}")
if FAILED:
    print("\n  Failures:")
    for name, detail in FAILED:
        print(f"    - {name}: {detail}")
    sys.exit(1)
else:
    print("\n  ALL CHANNELS VERIFIED ✓")
