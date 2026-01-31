
import pickle
import torch
import os
import sys

def inspect_buffer(path):
    if not os.path.exists(path):
        print(f"Buffer file not found: {path}")
        return

    print(f"Loading buffer from {path}...")
    try:
        with open(path, 'rb') as f:
            buffer_data = pickle.load(f)
        
        # Buffer is typically a wrapper or a deque
        # Logic depends on ReplayBufferMastery implementation.
        # It usually stores self.buffer which is a deque of (state, policy, value)
        
        # Let's assume it's the class instance dumped, or list.
        # ReplayBufferMastery.save dumps the whole object? Or list?
        # Checking replay_buffer_mastery.py... it uses pickle.dump(self.buffer, f) usually.
        # Let's try to access it as list.
        
        data = buffer_data
        if hasattr(buffer_data, 'buffer'):
            data = buffer_data.buffer
            
        print(f"Buffer contains {len(data)} items.")
        
        if len(data) == 0:
            print("Buffer is empty.")
            return

        # Inspect first item
        state, policy, value = data[0]
        
        print(f"Sample 0 State Shape: {state.shape}")
        print(f"Sample 0 Policy Shape: {policy.shape}")
        print(f"Sample 0 Value: {value}")
        
        # Verify 18 Channels
        if state.shape[0] == 18:
            print("SUCCESS: Input tensor has 18 channels.")
        else:
            print(f"FAILURE: Input tensor has {state.shape[0]} channels (Expected 18).")

        # Verify Spatial Dimensions
        if state.shape[1] == 15 and state.shape[2] == 15:
            print("SUCCESS: Input tensor is 15x15.")
        else:
            print(f"FAILURE: Input tensor is {state.shape[1]}x{state.shape[2]} (Expected 15x15).")
            
    except Exception as e:
        print(f"Error inspecting buffer: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_buffer(sys.argv[1])
    else:
        # Default async path
        inspect_buffer("checkpoints_mastery/mastery_v1/replay_buffer.pkl")
