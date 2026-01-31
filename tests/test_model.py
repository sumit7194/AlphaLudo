import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import ludo_cpp
from tensor_utils import state_to_tensor
from model import AlphaLudoNet

def test_tensor_conversion():
    state = ludo_cpp.create_initial_state()
    # Modify state to test specific channels
    state.player_positions[0][0] = 0 # P0 at start
    state.player_positions[1][0] = 0 # P1 at start (relative 0 -> abs 13)
    state.current_dice_roll = 3
    state.current_player = 2
    
    tensor = state_to_tensor(state)
    
    # Check shape
    assert tensor.shape == (8, 15, 15)
    
    # Check Channel 0 (P0 mask)
    # P0 at 0 maps to (6, 1) locally. P0 rotation is 0. So (6, 1).
    assert tensor[0, 6, 1] == 1.0
    
    # Check Channel 1 (P1 mask)
    # P1 at 0 locally. P1 rotation 90 deg.
    # Local (6, 1). Rotate 90: (r, c) -> (c, 14-r) = (1, 14-6) = (1, 8).
    assert tensor[1, 1, 8] == 1.0
    
    # Check Channel 6 (Dice)
    # Roll 3 -> 3/6 = 0.5
    assert torch.all(tensor[6] == 0.5)
    
    # Check Channel 7 (Turn)
    # Player 2 -> (2+1)*0.25 = 0.75
    assert torch.all(tensor[7] == 0.75)
    
    print("Tensor conversion test passed.")

def test_model_forward():
    batch_size = 4
    dummy_input = torch.randn(batch_size, 8, 15, 15)
    model = AlphaLudoNet()
    
    policy, value = model(dummy_input)
    
    # Check output shapes
    assert policy.shape == (batch_size, 4)
    assert value.shape == (batch_size, 1)
    
    # Check value range (-1 to 1 due to Tanh)
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0)
    
    # Check policy is probabilities (Softmax)
    # Sum across dim 1 should be approx 1.0
    sums = torch.sum(policy, dim=1)
    assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5)
    
    print("Model forward pass test passed.")

def test_model_backward():
    model = AlphaLudoNet()
    dummy_input = torch.randn(2, 8, 15, 15)
    target_policy = torch.randn(2, 4).softmax(dim=1)
    target_value = torch.randn(2, 1)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Forward
    policy, value = model(dummy_input)
    
    # Loss
    loss_p = torch.nn.functional.cross_entropy(policy, target_policy) # Technically cross_entropy takes logits
    # But for dummy test it's fine. Or use MSE.
    loss_v = torch.nn.functional.mse_loss(value, target_value)
    total_loss = loss_p + loss_v
    
    # Backward
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print("Model backward pass test passed.")

if __name__ == "__main__":
    test_tensor_conversion()
    test_model_forward()
    test_model_backward()
