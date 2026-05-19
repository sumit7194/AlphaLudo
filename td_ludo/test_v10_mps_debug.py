"""
Debug: why does MPS work for 50 standalone batches but NaN for every
batch in the real training loop? Mirror the exact training setup
(DataLoader, shuffle, non_blocking, full data) and inspect per-batch.
"""
import os, sys, glob
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from td_ludo.models.v10 import AlphaLudoV10

# Inline copy of InMemoryDataset (same as train_sl_v10.py)
from torch.utils.data import Dataset


class InMemoryDataset(Dataset):
    def __init__(self, chunk_paths, max_states=None):
        need = []
        running = 0
        for p in chunk_paths:
            d = np.load(p)
            n = len(d['states'])
            if max_states and running + n > max_states:
                n = max_states - running
                d.close()
                need.append((p, n))
                running += n
                break
            d.close()
            need.append((p, n))
            running += n
            if max_states and running >= max_states:
                break
        total = running
        self.states = np.empty((total, 28, 15, 15), dtype=np.float32)
        self.policies = np.empty((total, 4), dtype=np.float32)
        self.masks = np.empty((total, 4), dtype=np.float32)
        self.won = np.empty(total, dtype=np.float32)
        self.moves = np.empty(total, dtype=np.float32)
        idx = 0
        for p, n in need:
            d = np.load(p)
            self.states[idx:idx+n] = d['states'][:n]
            self.policies[idx:idx+n] = d['policies'][:n]
            self.masks[idx:idx+n] = d['legal_masks'][:n]
            self.won[idx:idx+n] = d['won'][:n].astype(np.float32)
            self.moves[idx:idx+n] = d['moves_remaining'][:n].astype(np.float32)
            d.close()
            idx += n

    def __len__(self): return len(self.states)
    def __getitem__(self, i):
        return (torch.from_numpy(self.states[i]),
                torch.from_numpy(self.policies[i]),
                torch.from_numpy(self.masks[i]),
                torch.tensor(self.won[i], dtype=torch.float32),
                torch.tensor(self.moves[i], dtype=torch.float32))


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('mps')
    print(f"Device: {device}  |  Using EXACT real training setup\n")

    chunks = sorted(glob.glob('checkpoints/sl_data_v10/chunk_*.npz'))
    train_chunks = chunks[:-2]
    train_set = InMemoryDataset(train_chunks, max_states=500000)
    print(f"Dataset: {len(train_set):,} samples")

    train_loader = DataLoader(train_set, batch_size=512, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)

    model = AlphaLudoV10(num_res_blocks=6, num_channels=96, in_channels=28).to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    W_POL, W_WIN, W_MOV = 1.0, 0.5, 0.003

    MAX_STEPS = 200  # extended to find NaN onset
    print(f"\nRunning first {MAX_STEPS} batches to hunt NaN source...")
    for step, batch in enumerate(train_loader):
        if step >= MAX_STEPS:
            break
        states, policies, masks, won, moves = batch
        states = states.to(device, non_blocking=True)
        policies = policies.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        won = won.to(device, non_blocking=True)
        moves = moves.to(device, non_blocking=True)

        # Raw input check
        for name, t in [('states', states), ('policies', policies),
                         ('masks', masks), ('won', won), ('moves', moves)]:
            if torch.isnan(t).any() or torch.isinf(t).any():
                print(f"  Step {step}: ✗ NaN/Inf IN INPUT {name}!")

        pol, win, mv = model(states, masks)
        # Model output check
        pol_nan = torch.isnan(pol).any().item()
        win_nan = torch.isnan(win).any().item()
        mv_nan = torch.isnan(mv).any().item()

        log_student = torch.log(pol + 1e-8)
        pol_loss = F.kl_div(log_student, policies, reduction='batchmean')
        win_loss = F.binary_cross_entropy(win.clamp(1e-6, 1-1e-6), won)
        moves_loss = F.smooth_l1_loss(mv, moves)
        total = W_POL*pol_loss + W_WIN*win_loss + W_MOV*moves_loss

        total_is_nan = torch.isnan(total).item()
        tot_val = total.item()

        if total_is_nan or pol_nan or win_nan or mv_nan:
            print(f"  Step {step:>2}: ✗ NaN | model_out(pol={pol_nan}, win={win_nan}, mv={mv_nan}) "
                  f"| loss(pol={pol_loss.item():.4f}, win={win_loss.item():.4f}, mv={moves_loss.item():.4f}) "
                  f"| total={tot_val}")
            # Dump sample values
            print(f"    states min={states.min().item():.3f} max={states.max().item():.3f}")
            print(f"    pol sample: {pol[0].detach().cpu().numpy()}")
            print(f"    win sample: {win[:3].detach().cpu().numpy()}")
            print(f"    mv sample: {mv[:3].detach().cpu().numpy()}")
            break

        optimizer.zero_grad()
        total.backward()

        # Check for NaN in gradients
        grad_nan = 0
        for n, p in model.named_parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                grad_nan += 1
        if grad_nan > 0:
            print(f"  Step {step:>2}: ✗ {grad_nan} params have NaN in gradient — investigating")
            for n, p in model.named_parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    print(f"    NaN grad: {n}  shape={tuple(p.grad.shape)}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Check weights after step
        weight_nan = 0
        for n, p in model.named_parameters():
            if torch.isnan(p).any():
                weight_nan += 1
        if weight_nan > 0:
            print(f"  Step {step:>2}: ✗ {weight_nan} params NaN AFTER optimizer.step()")
            break

        # Report every 20 steps to reduce noise
        if step % 20 == 0 or step < 5:
            print(f"  Step {step:>3}: ✓  total={tot_val:.4f}  pol={pol_loss.item():.4f}  "
                  f"win={win_loss.item():.4f}  mv={moves_loss.item():.4f}")


if __name__ == '__main__':
    main()
