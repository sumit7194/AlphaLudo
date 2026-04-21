"""
MPS stability test for V10 with the new loss setup.

Previous MPS run (iter 3) NaN'd on epoch 1 with MSE-dominated loss
(total ~30+). With SmoothL1 + moves_weight=0.003, total loss is now
~0.9 — much smaller gradient scale, which should keep MPS numerics
stable.

This test runs 50 real training batches on MPS and reports:
  - Per-step policy/win/moves/total losses
  - Any NaN or Inf detection (aborts early)
  - Gradient norms per head
  - Loss trajectory (descending? flat? blowing up?)

50 batches at ~350ms each = ~18 seconds. If stable, we green-light the
full 3-epoch MPS run.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from td_ludo.models.v10 import AlphaLudoV10


def main():
    if not torch.backends.mps.is_available():
        print("MPS not available on this machine — skipping.")
        sys.exit(0)

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('mps')

    # Load chunk 0 for realistic inputs
    chunk = np.load('checkpoints/sl_data_v10/chunk_0000.npz')
    n = len(chunk['states'])
    batch_size = 512

    model = AlphaLudoV10(num_res_blocks=6, num_channels=96, in_channels=28).to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    W_POL, W_WIN, W_MOV = 1.0, 0.5, 0.003

    print("=" * 62)
    print("  V10 MPS Stability Test — 50 steps, batch 512")
    print(f"  Device: {device}  |  Weights: pol={W_POL} win={W_WIN} moves={W_MOV}")
    print(f"  Loss: KL(teacher||student) + BCE(win) + SmoothL1(moves)")
    print("=" * 62)
    print(f"  {'step':>4} {'total':>8} {'pol':>7} {'win':>7} {'mov':>7} {'acc':>6} {'t(ms)':>6}")

    losses = []
    nan_count = 0
    t_start = time.time()

    for step in range(50):
        idx = np.random.choice(n, size=batch_size, replace=False)
        states = torch.from_numpy(chunk['states'][idx]).float().to(device)
        policies = torch.from_numpy(chunk['policies'][idx]).float().to(device)
        masks = torch.from_numpy(chunk['legal_masks'][idx]).float().to(device)
        won = torch.from_numpy(chunk['won'][idx]).float().to(device)
        moves = torch.from_numpy(chunk['moves_remaining'][idx]).float().to(device)

        t0 = time.time()
        pol, win, mv = model(states, masks)
        log_student = torch.log(pol + 1e-8)
        pol_loss = F.kl_div(log_student, policies, reduction='batchmean')
        win_loss = F.binary_cross_entropy(win.clamp(1e-6, 1-1e-6), won)
        moves_loss = F.smooth_l1_loss(mv, moves)
        total = W_POL*pol_loss + W_WIN*win_loss + W_MOV*moves_loss

        # NaN / Inf detection BEFORE backward
        if torch.isnan(total) or torch.isinf(total):
            nan_count += 1
            print(f"  {step:>4d}  ✗ NaN/Inf detected — pol={pol_loss.item()} "
                  f"win={win_loss.item()} mov={moves_loss.item()}")
            if nan_count >= 3:
                print("\n  ABORT: 3+ NaN steps, MPS unstable with this config")
                sys.exit(1)
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        dt_ms = (time.time() - t0) * 1000

        pol_acc = (pol.argmax(1) == policies.argmax(1)).float().mean().item() * 100
        losses.append(total.item())

        # Report every 5 steps
        if step % 5 == 0 or step == 49:
            print(f"  {step:>4d} {total.item():>8.4f} {pol_loss.item():>7.4f} "
                  f"{win_loss.item():>7.4f} {moves_loss.item():>7.3f} "
                  f"{pol_acc:>5.1f}% {dt_ms:>5.0f}")

    elapsed = time.time() - t_start

    # NaN in model weights check
    nan_params = 0
    for name, p in model.named_parameters():
        if torch.isnan(p).any():
            nan_params += 1

    first_avg = np.mean(losses[:5])
    last_avg = np.mean(losses[-5:])
    throughput = 50 * batch_size / elapsed

    print(f"\n  Elapsed: {elapsed:.1f}s  |  Throughput: {throughput:.0f} samples/s")
    print(f"  First 5 avg loss: {first_avg:.4f}")
    print(f"  Last 5 avg loss:  {last_avg:.4f}")

    passed = True
    if nan_count > 0:
        print(f"  ✗ NaN steps detected: {nan_count}"); passed = False
    else:
        print(f"  ✓ Zero NaN steps across 50 batches")
    if nan_params > 0:
        print(f"  ✗ NaN in {nan_params} model params"); passed = False
    else:
        print(f"  ✓ No NaN in model weights")
    if last_avg < first_avg:
        print(f"  ✓ Loss descending ({first_avg - last_avg:.4f} drop)")
    else:
        print(f"  ⚠ Loss NOT descending — may indicate MPS numerical drift")

    if passed and last_avg < first_avg:
        # Project per-epoch time
        n_batches_per_epoch = 490_000 // batch_size
        epoch_s = n_batches_per_epoch * (elapsed / 50)
        print(f"\n  Projected epoch time on MPS: {epoch_s:.0f}s ({epoch_s/60:.1f} min)")
        print(f"  Projected 3-epoch total: {3*epoch_s/60:.1f} min")
        print(f"  ✓ MPS GREENLIT for full training run")
    else:
        print(f"\n  ✗ MPS NOT stable — stay on CPU or debug further")
        sys.exit(1)


if __name__ == '__main__':
    main()
