import torch
import os

ckpt_path = 'checkpoints_mastery/mastery_no6_v1/model_latest.pt'
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False) # weights_only=False to avoid pickle warning/error if simpler
    print("Checkpoint keys:", ckpt.keys())
else:
    print("Checkpoint not found")
