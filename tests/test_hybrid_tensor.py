
import sys
import os
import torch
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath("."))

from src.tensor_utils_mastery import state_to_tensor_mastery
from src.model_mastery import AlphaLudoTopNet
import ludo_cpp

def test_hybrid_tensor():
    print("Testing Hybrid v6 Tensor Generation...")
    
    # 1. Create State
    state = ludo_cpp.create_initial_state()
    
    # 2. Python Tensor Generation
    try:
        tensor_py = state_to_tensor_mastery(state)
        print(f"Python Tensor Shape: {tensor_py.shape}")
        assert tensor_py.shape == (18, 15, 15)
        print("Python Tensor OK.")
    except Exception as e:
        print(f"Python Tensor Failed: {e}")
        return

    # 3. Model Inference (Python Tensor)
    model = AlphaLudoTopNet()
    try:
        batch = tensor_py.unsqueeze(0) # (1, 18, 15, 15)
        p, v = model(batch)
        print(f"Model Output: p={p.shape}, v={v.shape}")
        print("Model Inference OK.")
    except Exception as e:
        print(f"Model Inference Failed: {e}")
        return

    # 4. C++ Tensor Generation (Mock MCTS)
    # We can't easily call write_state_tensor directly from python without exposed binding
    # But MCTS exposes get_leaf_tensors.
    print("Testing C++ MCTS Tensor Generation...")
    mcts = ludo_cpp.MCTSEngine(16, 1.0) # Batch 16
    
    # Set roots (16 states)
    states = [ludo_cpp.create_initial_state() for _ in range(16)]
    mcts.set_roots(states)
    
    mcts.select_leaves()
    tensors = mcts.get_leaf_tensors()
    print(f"C++ Leaf Tensors Shape: {tensors.shape}")
    
    if tensors.shape[1] != 18:
        print("ERROR: C++ Tensor Channel Mismatch!")
    else:
        print("C++ Tensor Channels OK.")
        
    # Validation of content?
    # Check if sum is non-zero (Safe zones should be present)
    s = np.sum(tensors)
    print(f"Tensor Sum: {s}")
    if s == 0:
        print("WARNING: Tensor is all zeros!")

if __name__ == "__main__":
    test_hybrid_tensor()
