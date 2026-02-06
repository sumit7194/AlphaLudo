
import pickle
import os
import sys
import glob
import torch
import numpy as np
# import matplotlib.pyplot as plt

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import ludo_cpp

BUFFER_PATTERN = "data/kickstart_buffer.pkl.part_*"

def verify_buffer():
    files = glob.glob(BUFFER_PATTERN)
    if not files:
        print("No buffer shards found yet.")
        return

    print(f"Found {len(files)} shards.")
    total_samples = 0
    bot_wins = {0:0, 1:0, 2:0, 3:0} # Player indices
    policy_dist = []
    
    # Check random shard
    shard_path = np.random.choice(files)
    print(f"Inspecting shard: {shard_path}")
    
    with open(shard_path, 'rb') as f:
        data = pickle.load(f)
        
    print(f"Shard contains {len(data)} samples.")
    total_samples += len(data)
    
    # Inspect content
    valid_states = 0
    moves_distribution = np.zeros(4)
    
    for i, (state_tensor, policy, value) in enumerate(data[:100]): # Check first 100
        # Policy is one-hot?
        if torch.is_tensor(policy):
            p = policy.numpy()
        else:
            p = policy
            
        move_idx = np.argmax(p)
        moves_distribution[move_idx] += 1
        
        # Value check
        v = value.item()
        if abs(v) != 1.0:
            print(f"WARNING: Value {v} is not +1/-1 at idx {i}")
            
        # State Tensor Shape
        if state_tensor.shape != torch.Size([21, 15, 15]):
             print(f"WARNING: Invalid shape {state_tensor.shape} at idx {i}")
        else:
             valid_states += 1

    print(f"Verified {min(len(data), 100)} samples structure.")
    print(f"Moves Dist (First 100): {moves_distribution}")
    
    # Plot histogram if requested (text based)
    p_counts = [0,0,0,0]
    for _, policy, _ in data:
         p_counts[np.argmax(policy)] += 1
    
    print(f"Full Shard Move Distribution: {p_counts}")
    print(f"Valid Tensor Count: {valid_states}/{min(len(data), 100)}")

if __name__ == "__main__":
    verify_buffer()
