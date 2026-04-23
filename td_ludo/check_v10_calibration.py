"""
Quick calibration check for V10 win_prob head.

Feeds 5 hand-crafted states spanning the P0-scoring progression. A properly
calibrated win_prob increases monotonically with number of scored tokens.
The V10.2 fix (BCE loss + drop SmoothL1 value loss) should keep this
monotonic across RL training.

Usage (run on any V10 checkpoint):
  python check_v10_calibration.py --ckpt checkpoints/ac_v10/model_sl.pt
  python check_v10_calibration.py --ckpt checkpoints/ac_v10/model_latest.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import td_ludo_cpp as ludo_cpp
from td_ludo.models.v10 import AlphaLudoV10


def build_state(p0_scored, dice=3):
    """P0 progression, P2 at fixed early positions."""
    s = ludo_cpp.create_initial_state_2p()
    s.player_positions[2] = [5, 10, 15, 20]
    positions = [99] * p0_scored + [5] * (4 - p0_scored)
    s.player_positions[0] = positions[:4]
    s.scores[0] = p0_scored
    s.current_player = 0
    s.current_dice_roll = dice
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    arch = ckpt.get('arch', {'num_res_blocks': 6, 'num_channels': 96, 'in_channels': 28})
    model = AlphaLudoV10(**arch)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"checkpoint: {args.ckpt}")
    print(f"  total_games: {ckpt.get('total_games', 'n/a')}  "
          f"total_updates: {ckpt.get('total_updates', 'n/a')}")
    print()
    print(f"  {'P0 scored':<12}{'win_prob':<12}{'direction'}")

    prev = None
    monotone_increase = True
    for scored in range(5):
        s = build_state(scored)
        t = torch.from_numpy(np.array(ludo_cpp.encode_state_v10(s))).float().unsqueeze(0)
        mask = torch.ones(1, 4)
        with torch.no_grad():
            _, wp, _ = model(t, mask)
        val = wp.item()
        arrow = ""
        if prev is not None:
            if val > prev + 0.01:
                arrow = "  ↑"
            elif val < prev - 0.01:
                arrow = "  ↓"
                monotone_increase = False
            else:
                arrow = "  ="
        print(f"  {scored:<12}{val:<12.3f}{arrow}")
        prev = val

    print()
    if monotone_increase:
        print("  ✓ Monotone increasing — win_prob calibrated")
    else:
        print("  ✗ NOT monotone increasing — win_prob is corrupted/inverted")


if __name__ == '__main__':
    main()
