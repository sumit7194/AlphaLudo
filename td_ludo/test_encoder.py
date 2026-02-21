"""
Test suite for the 11-channel tensor encoder.
Verifies that C++ and Python encoders produce identical outputs,
and that the channel layout matches the expected 11-channel spec.

Run: td_env/bin/python test_encoder.py
"""
import sys
import os
import numpy as np
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import td_ludo_cpp as ludo_cpp
from tensor_utils import state_to_tensor_mastery

BOARD_SIZE = 15
NUM_CHANNELS = 11
BASE_POS = -1

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}: {detail}")

# ============================================================
# Test 1: Tensor shape
# ============================================================
print("\n--- Test 1: C++ Tensor Shape ---")
env = ludo_cpp.VectorGameState(4, True) # 2-player mode
tensor = env.get_state_tensor()
check("Shape is (4, 11, 15, 15)", tensor.shape == (4, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE),
      f"got {tensor.shape}")

# ============================================================
# Test 2: No dice channels
# ============================================================
print("\n--- Test 2: No Dice Channels ---")
env2 = ludo_cpp.VectorGameState(1, True)
game = env2.get_game(0)
# Initial state has roll=0.
tensor2 = env2.get_state_tensor()[0]

dice_found = False
for ch in range(NUM_CHANNELS):
    if np.all(tensor2[ch] == 1.0) and ch < 8:
        dice_found = True
        print(f"    Channel {ch} is all 1.0 — suspicious for spatial channel")
check("No dice-like broadcast in spatial channels", not dice_found)

# ============================================================
# Test 3: Initial state properties
# ============================================================
print("\n--- Test 3: Initial State (C++) ---")
env3 = ludo_cpp.VectorGameState(1, True)
t = env3.get_state_tensor()[0]

# Ch 0-3: Each token should have exactly 1 non-zero cell (in base)
for ch in range(4):
    nz = np.count_nonzero(t[ch])
    check(f"Ch {ch} (my token) has 1 non-zero cell", nz == 1, f"got {nz}")

check("Ch 8 (score diff) is 0 at start", np.allclose(t[8], 0.0), f"got {t[8, 0, 0]}")
check("Ch 9 (my locked) is 1.0 at start", np.allclose(t[9], 1.0), f"got {t[9, 0, 0]}")

# ============================================================
# Test 4: Inactive players excluded (2-Player Mode)
# ============================================================
print("\n--- Test 4: Inactive Players Excluded (2P) ---")
env4 = ludo_cpp.VectorGameState(1, True)
g4 = env4.get_game(0)
ap = g4.active_players
check("P0 active", ap[0] == True)
check("P1 inactive", ap[1] == False)
check("P2 active", ap[2] == True)
check("P3 inactive", ap[3] == False)

t4 = env4.get_state_tensor()[0]
# Ch 4: Only P2 tokens (4 × 0.25 = 1.0 total)
opp_sum = t4[4].sum()
check(f"Ch 4 (opponent) sum ≈ 1.0 (4 tokens × 0.25)", abs(opp_sum - 1.0) < 0.01,
      f"got {opp_sum:.4f}")

# ============================================================
# Test 5: Opp locked correct denominator
# ============================================================
print("\n--- Test 5: Opp Locked Denominator ---")
opp_locked = t4[10, 0, 0]
check("Ch 10 (opp locked) = 1.0 at start (4/4 active opp tokens)", 
      abs(opp_locked - 1.0) < 0.01, f"got {opp_locked:.4f}")

# ============================================================
# Test 6: C++ vs Python encoder match
# ============================================================
print("\n--- Test 6: C++ vs Python Encoder Match ---")
env6 = ludo_cpp.VectorGameState(4, True)

# Play random moves to get diverse states
for _ in range(50):
    legal_moves_batch = env6.get_legal_moves()
    actions = []
    for i in range(4):
        moves = legal_moves_batch[i]
        if moves:
            actions.append(random.choice(moves))
        else:
            actions.append(-1)
    env6.step(actions)

cpp_tensor = env6.get_state_tensor()

all_match = True
for i in range(4):
    game = env6.get_game(i)
    if game.is_terminal:
        continue
    py_tensor = state_to_tensor_mastery(game)
    
    if not np.allclose(cpp_tensor[i], py_tensor, atol=1e-5):
        all_match = False
        for ch in range(NUM_CHANNELS):
            if not np.allclose(cpp_tensor[i, ch], py_tensor[ch], atol=1e-5):
                cpp_sum = cpp_tensor[i, ch].sum()
                py_sum = py_tensor[ch].sum()
                max_diff = np.max(np.abs(cpp_tensor[i, ch] - py_tensor[ch]))
                print(f"    Game {i}, Ch {ch}: C++={cpp_sum:.4f} Python={py_sum:.4f} maxdiff={max_diff:.6f}")

check("C++ and Python encoders match", all_match)

# ============================================================
# Test 7: Model forward pass
# ============================================================
print("\n--- Test 7: Model Forward Pass ---")
try:
    import torch
    from model import AlphaLudoV3
    
    model = AlphaLudoV3()
    dummy = torch.randn(2, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    policy, value, aux = model(dummy)
    check("Policy shape (2, 4)", policy.shape == (2, 4), f"got {policy.shape}")
    check("Value shape (2, 1)", value.shape == (2, 1), f"got {value.shape}")
except Exception as e:
    check("Model forward pass", False, str(e))

# ============================================================
# Test 8: Channel value ranges after gameplay
# ============================================================
print("\n--- Test 8: Channel Value Ranges ---")
env8 = ludo_cpp.VectorGameState(8, True)
for _ in range(100):
    legal_moves_batch = env8.get_legal_moves()
    actions = []
    for i in range(8):
        moves = legal_moves_batch[i]
        if moves:
            actions.append(random.choice(moves))
        else:
            actions.append(-1)
    env8.step(actions)

t8 = env8.get_state_tensor()
ranges_ok = True
for ch in range(NUM_CHANNELS):
    mn = t8[:, ch].min()
    mx = t8[:, ch].max()
    if ch <= 3:
        ok = mn >= 0 and mx <= 1.0
    elif ch == 4:
        ok = mn >= 0 and mx <= 1.0
    elif ch == 5:
        ok = mn >= 0 and mx <= 0.5 + 0.01
    elif ch <= 7:
        ok = mn >= 0 and mx <= 1.0
    elif ch == 8:
        ok = mn >= -1.0 and mx <= 1.0
    else:
        ok = mn >= 0 and mx <= 1.0
    if not ok:
        ranges_ok = False
        print(f"    Ch {ch}: [{mn:.3f}, {mx:.3f}] out of expected range")

check("All channels in expected value ranges", ranges_ok)

# ============================================================
# Summary
# ============================================================
print(f"\n{'=' * 60}")
total = passed + failed
print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
if failed == 0:
    print("🎉 ALL TESTS PASSED!")
else:
    print("⚠️  Some tests failed — review output above")
    sys.exit(1)
