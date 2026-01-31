import torch
import os

try:
    path = "checkpoints/checkpoint.pt"
    if os.path.exists(path):
        ckpt = torch.load(path, map_location='cpu')
        print(f"Keys: {ckpt.keys()}")
        print(f"Epoch: {ckpt.get('epoch')}")
    else:
        print("No checkpoint found.")
except Exception as e:
    print(f"Error: {e}")
