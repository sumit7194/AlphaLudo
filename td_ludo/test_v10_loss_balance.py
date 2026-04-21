"""
Quick verification that the SmoothL1 fix rebalances the multi-task loss
as expected. Runs one real batch through V10 slim + computes each loss
component. Compares to the old MSE values on the same input.

Expected:
  - moves_loss (SmoothL1) should be ~10-20 range, not 300+
  - policy component (×1.0) should be ~40-60% of total
  - moves component (×0.02) should be ~3-10% of total (not 90%)
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from td_ludo.models.v10 import AlphaLudoV10


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # Load one chunk of real V6.1-only data to get realistic inputs
    chunk = np.load('checkpoints/sl_data_v10/chunk_0000.npz')
    idx = np.random.choice(len(chunk['states']), size=512, replace=False)
    states = torch.from_numpy(chunk['states'][idx]).float()
    policies = torch.from_numpy(chunk['policies'][idx]).float()
    masks = torch.from_numpy(chunk['legal_masks'][idx]).float()
    won = torch.from_numpy(chunk['won'][idx]).float()
    moves = torch.from_numpy(chunk['moves_remaining'][idx]).float()

    # Fresh V10 slim
    model = AlphaLudoV10(num_res_blocks=6, num_channels=96, in_channels=28)
    model.train()
    pol, win, mv = model(states, masks)

    # Each loss separately
    log_student = torch.log(pol + 1e-8)
    pol_loss = F.kl_div(log_student, policies, reduction='batchmean')
    win_loss = F.binary_cross_entropy(win.clamp(1e-6, 1-1e-6), won)
    moves_loss_mse = F.mse_loss(mv, moves)           # OLD behavior
    moves_loss_sl1 = F.smooth_l1_loss(mv, moves)     # NEW behavior

    W_POL, W_WIN, W_MOV = 1.0, 0.5, 0.003  # matches new train_sl_v10 defaults

    total_old = W_POL*pol_loss + W_WIN*win_loss + W_MOV*moves_loss_mse
    total_new = W_POL*pol_loss + W_WIN*win_loss + W_MOV*moves_loss_sl1

    def pct(x, total): return x / total.item() * 100

    print("=" * 60)
    print("  V10 Multi-task loss balance test")
    print(f"  Batch: 512 random states from chunk_0000.npz (V6.1-only)")
    print(f"  Weights: policy={W_POL}  win={W_WIN}  moves={W_MOV}")
    print("=" * 60)

    print("\n  OLD (MSE on moves):")
    print(f"    Raw policy KL:        {pol_loss.item():>8.4f}")
    print(f"    Raw win BCE:          {win_loss.item():>8.4f}")
    print(f"    Raw moves MSE:        {moves_loss_mse.item():>8.4f}  ← dominates")
    print(f"    Weighted policy:      {(W_POL*pol_loss).item():>8.4f}  ({pct(W_POL*pol_loss.item(), total_old):.1f}%)")
    print(f"    Weighted win:         {(W_WIN*win_loss).item():>8.4f}  ({pct(W_WIN*win_loss.item(), total_old):.1f}%)")
    print(f"    Weighted moves:       {(W_MOV*moves_loss_mse).item():>8.4f}  ({pct(W_MOV*moves_loss_mse.item(), total_old):.1f}%)")
    print(f"    Total:                {total_old.item():>8.4f}")

    print("\n  NEW (SmoothL1 on moves):")
    print(f"    Raw policy KL:        {pol_loss.item():>8.4f}")
    print(f"    Raw win BCE:          {win_loss.item():>8.4f}")
    print(f"    Raw moves SmoothL1:   {moves_loss_sl1.item():>8.4f}  ({moves_loss_mse.item() / max(moves_loss_sl1.item(), 1e-6):.0f}x smaller)")
    print(f"    Weighted policy:      {(W_POL*pol_loss).item():>8.4f}  ({pct(W_POL*pol_loss.item(), total_new):.1f}%)")
    print(f"    Weighted win:         {(W_WIN*win_loss).item():>8.4f}  ({pct(W_WIN*win_loss.item(), total_new):.1f}%)")
    print(f"    Weighted moves:       {(W_MOV*moves_loss_sl1).item():>8.4f}  ({pct(W_MOV*moves_loss_sl1.item(), total_new):.1f}%)")
    print(f"    Total:                {total_new.item():>8.4f}")

    # Backward pass test — make sure gradients flow with new loss
    print("\n  Gradient check:")
    model.zero_grad()
    total_new.backward()
    grad_norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            # Aggregate by head
            if 'policy_fc' in name: key = 'policy_head'
            elif 'win_fc' in name: key = 'win_head'
            elif 'moves_fc' in name: key = 'moves_head'
            else: key = 'backbone'
            grad_norms[key] = grad_norms.get(key, 0) + p.grad.norm().item()
    for k, v in grad_norms.items():
        print(f"    {k:<14s} grad norm: {v:.4f}")

    # Verdict
    pol_pct_new = pct(W_POL*pol_loss.item(), total_new)
    mov_pct_new = pct(W_MOV*moves_loss_sl1.item(), total_new)
    print("\n  Verdict:")
    if pol_pct_new > 30 and mov_pct_new < 20:
        print(f"    ✓ Loss balance looks healthy (pol={pol_pct_new:.1f}%, moves={mov_pct_new:.1f}%)")
    else:
        print(f"    ✗ Loss still imbalanced (pol={pol_pct_new:.1f}%, moves={mov_pct_new:.1f}%)")


if __name__ == '__main__':
    main()
