from src.replay_buffer_mastery import ReplayBufferMastery
from collections import deque
import pickle
import os

# Test 1: Verify maxlen initialization
rb = ReplayBufferMastery(max_size=200000)
print(f"Test 1: Requested 200k, Got maxlen={rb.buffer.maxlen}")

# Test 2: Verify load behavior
# Create a dummy file with 60k items
dummy_data = [('state', 'idx', 'p', 'v')] * 60000
with open("test_buffer.pkl", "wb") as f:
    pickle.dump(dummy_data, f)

rb.load("test_buffer.pkl")
print(f"Test 2: Loaded 60k items. Buffer len={len(rb.buffer)}. Maxlen={rb.buffer.maxlen}")
if len(rb.buffer) == 60000:
    print("SUCCESS: Buffer stored >50k items")
else:
    print("FAILURE: Buffer capped")

os.remove("test_buffer.pkl")
