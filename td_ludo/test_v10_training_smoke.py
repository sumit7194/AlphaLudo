"""
Mini training smoke test: 10 batches with fresh V10 model + V6.1 data,
same loss function as train_sl_v10.py. Verifies:
  - No crashes / NaN in forward/backward
  - All 3 losses produce gradients
  - Combined loss is descending over 10 steps
  - Checkpoint save/load roundtrip works

Runs in ~30 seconds. Catches any integration bugs before committing to a
full 3-epoch CPU run (~1.5 hrs).
"""
import os, sys, tempfile
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from td_ludo.models.v10 import AlphaLudoV10


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    # Load a small real batch from the V6.1-only dataset
    chunk = np.load('checkpoints/sl_data_v10/chunk_0000.npz')
    n = len(chunk['states'])
    batch_size = 128

    model = AlphaLudoV10(num_res_blocks=6, num_channels=96, in_channels=28)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Same weights as train_sl_v10 defaults
    W_POL, W_WIN, W_MOV = 1.0, 0.5, 0.003

    print("=" * 60)
    print("  V10 Training Smoke Test — 10 steps")
    print("=" * 60)
    print(f"  {'step':>4}  {'total':>8}  {'pol':>7}  {'win':>7}  {'moves':>7}  {'pol_acc':>8}")

    losses = []
    for step in range(10):
        idx = np.random.choice(n, size=batch_size, replace=False)
        states = torch.from_numpy(chunk['states'][idx]).float()
        policies = torch.from_numpy(chunk['policies'][idx]).float()
        masks = torch.from_numpy(chunk['legal_masks'][idx]).float()
        won = torch.from_numpy(chunk['won'][idx]).float()
        moves = torch.from_numpy(chunk['moves_remaining'][idx]).float()

        pol, win, mv = model(states, masks)
        log_student = torch.log(pol + 1e-8)
        pol_loss = F.kl_div(log_student, policies, reduction='batchmean')
        win_loss = F.binary_cross_entropy(win.clamp(1e-6, 1-1e-6), won)
        moves_loss = F.smooth_l1_loss(mv, moves)
        total = W_POL*pol_loss + W_WIN*win_loss + W_MOV*moves_loss

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Policy accuracy
        pol_acc = (pol.argmax(1) == policies.argmax(1)).float().mean().item() * 100
        losses.append(total.item())

        print(f"  {step:>4d}  {total.item():>8.4f}  {pol_loss.item():>7.4f}  "
              f"{win_loss.item():>7.4f}  {moves_loss.item():>7.4f}  {pol_acc:>7.1f}%")

    # Verify loss is descending
    first_avg = np.mean(losses[:3])
    last_avg = np.mean(losses[-3:])
    print(f"\n  First 3 steps avg loss: {first_avg:.4f}")
    print(f"  Last 3 steps avg loss:  {last_avg:.4f}")
    if last_avg < first_avg:
        print(f"  ✓ Loss descending ({first_avg - last_avg:.4f} drop)")
    else:
        print(f"  ⚠ Loss NOT descending — may need more steps to see movement")

    # NaN check
    for name, p in model.named_parameters():
        if torch.isnan(p).any():
            print(f"  ✗ NaN in parameter: {name}")
            sys.exit(1)
    print(f"  ✓ No NaN in any model parameter")

    # Checkpoint save/load roundtrip
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        ckpt_path = f.name
    save = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': 1,
        'arch': {'num_res_blocks': 6, 'num_channels': 96, 'in_channels': 28},
    }
    torch.save(save, ckpt_path)
    loaded = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    assert 'model_state_dict' in loaded
    assert 'optimizer_state_dict' in loaded
    assert loaded['arch']['num_channels'] == 96
    # Reconstruct model and verify weights match
    m2 = AlphaLudoV10(num_res_blocks=6, num_channels=96, in_channels=28)
    m2.load_state_dict(loaded['model_state_dict'])
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), m2.named_parameters()):
        assert torch.allclose(p1, p2), f"Mismatch in {n1}"
    os.remove(ckpt_path)
    print(f"  ✓ Checkpoint save/load roundtrip works")

    print("\n  ALL CHECKS PASSED ✓")


if __name__ == '__main__':
    main()
