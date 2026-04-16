"""Initialize V6.3 model from V6.1 weights.

Transfers CNN backbone + policy/value heads (24→27ch stem expansion).
Also carries over return normalization stats and best_win_rate.
Auxiliary capture head keeps random init.
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import AlphaLudoV5
from td_ludo.models.v6_3 import AlphaLudoV63

v61_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/ac_v6_1_strategic/model_best.pt"
out_dir = "checkpoints/ac_v6_3_capture"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(os.path.join(out_dir, "ghosts"), exist_ok=True)

print(f"Loading V6.1 from {v61_path}...")
ckpt = torch.load(v61_path, map_location='cpu', weights_only=False)
v61_sd = ckpt.get('model_state_dict', ckpt)

print("Creating V6.3 (128ch, 10res, 27in, aux capture head)...")
model = AlphaLudoV63(num_res_blocks=10, num_channels=128, in_channels=27)
model.load_v61_weights(v61_sd)

# Build checkpoint payload
payload = {'model_state_dict': model.state_dict()}

# Return normalization stats (CRITICAL — without these, value loss
# spikes to ~2.1 on first PPO update, corrupting the CNN trunk)
if 'return_running_mean' in ckpt and 'return_running_std' in ckpt:
    payload['return_running_mean'] = ckpt['return_running_mean']
    payload['return_running_std'] = ckpt['return_running_std']
    payload['return_stats_initialized'] = True
    print(f"  return_running_mean = {ckpt['return_running_mean']:.4f}")
    print(f"  return_running_std  = {ckpt['return_running_std']:.4f}")
else:
    print("  WARNING: V6.1 checkpoint has no return stats")

# Counters: reset games/updates but carry best_win_rate
payload['total_games'] = 0
payload['total_updates'] = 0
payload['best_win_rate'] = ckpt.get('best_win_rate', 0.0)
payload['last_ghost_game'] = 0
payload['metrics_history'] = []
payload['last_eval_wr'] = ckpt.get('last_eval_wr', None)
print(f"  best_win_rate carried over = {payload['best_win_rate']}")

out_path = os.path.join(out_dir, "model_sl.pt")
torch.save(payload, out_path)
print(f"\nSaved to {out_path}")
print(f"V6.3 params: {model.count_parameters():,}")
