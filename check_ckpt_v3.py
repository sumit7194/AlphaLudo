import torch
import os

try:
    path = "checkpoints/checkpoint.pt"
    if os.path.exists(path):
        ckpt = torch.load(path, map_location='cpu')
        print(f"Total Epochs: {ckpt.get('total_epochs')}")
        print(f"Total Steps: {ckpt.get('total_steps')}")
    else:
        print("No checkpoint found.")
except Exception as e:
    print(f"Error: {e}")
