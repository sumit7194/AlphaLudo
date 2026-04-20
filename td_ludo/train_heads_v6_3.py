"""
Train calibrated heads on top of V6.3's frozen backbone.

Adds two new heads:
  - win_prob:         Linear(128 -> 64 -> 1) + tanh → target {-1, +1} via MSE,
                       or Linear + sigmoid → target {0, 1} via BCE (used here)
  - moves_remaining:  Linear(128 -> 64 -> 1)         → MSE on own-turn count

The V6.3 backbone + policy head are frozen. Only the new heads train.

Output: `checkpoints/ac_v6_3_capture/model_heads.pt` — a state_dict with all
the V6.3 weights plus the new heads, loadable via AlphaLudoV63WithHeads.
"""

import argparse
import glob
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from td_ludo.models.v6_3 import AlphaLudoV63


class CalibratedHeads(nn.Module):
    """Two heads on top of a 128-d GAP feature vector: win_prob + moves_remaining."""

    def __init__(self, feature_dim=128, hidden_dim=64, max_moves=150):
        super().__init__()
        self.max_moves = max_moves
        self.win_prob_fc1 = nn.Linear(feature_dim, hidden_dim)
        self.win_prob_fc2 = nn.Linear(hidden_dim, 1)
        self.moves_fc1 = nn.Linear(feature_dim, hidden_dim)
        self.moves_fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        # win probability — sigmoid → [0, 1]
        w = F.relu(self.win_prob_fc1(features))
        win_logit = self.win_prob_fc2(w).squeeze(-1)
        # moves_remaining — we predict in log-space so positive outputs scale
        m = F.relu(self.moves_fc1(features))
        moves = F.softplus(self.moves_fc2(m)).squeeze(-1)  # >= 0
        return win_logit, moves


class HeadsDataset(Dataset):
    """Loads NPZ chunks with states, won, moves_remaining. States kept in RAM."""

    def __init__(self, chunk_paths, max_states=None):
        all_states, all_won, all_mr = [], [], []
        total = 0
        for path in chunk_paths:
            d = np.load(path)
            n = len(d['states'])
            if max_states and total + n > max_states:
                n = max_states - total
            all_states.append(d['states'][:n].astype(np.float32))
            all_won.append(d['won'][:n].astype(np.float32))
            all_mr.append(d['moves_remaining'][:n].astype(np.float32))
            d.close()
            total += n
            if max_states and total >= max_states:
                break

        self.states = np.concatenate(all_states)
        self.won = np.concatenate(all_won)
        self.moves_remaining = np.concatenate(all_mr)
        assert self.states.shape[1] == 27
        ram_gb = self.states.nbytes / 1e9
        print(f"[Dataset] {len(self.states):,} samples loaded ({ram_gb:.2f} GB). "
              f"Win rate: {self.won.mean():.3f}, "
              f"avg moves_remaining: {self.moves_remaining.mean():.1f}")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.states[idx]),
            torch.tensor(self.won[idx], dtype=torch.float32),
            torch.tensor(self.moves_remaining[idx], dtype=torch.float32),
        )


def extract_backbone_features(model, states, device, batch_size=512):
    """Run states through V6.3 backbone (frozen) and return GAP features."""
    model.eval()
    features = []
    with torch.no_grad():
        for i in range(0, len(states), batch_size):
            batch = states[i:i+batch_size].to(device)
            feat = model._backbone(batch)
            features.append(feat.cpu())
    return torch.cat(features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='checkpoints/ac_v6_3_capture/model_latest.pt')
    parser.add_argument('--data-dir', default='checkpoints/heads_data_v6_3')
    parser.add_argument('--output', default='checkpoints/ac_v6_3_capture/model_heads.pt')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max-states', type=int, default=200000)
    parser.add_argument('--moves-weight', type=float, default=0.02,
                        help='Weight on moves_remaining loss '
                             '(small because scale is ~30x bigger than BCE)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train Heads] Device: {device}")

    # Load V6.3 backbone (frozen)
    backbone = AlphaLudoV63(num_res_blocks=10, num_channels=128, in_channels=27)
    ckpt = torch.load(args.backbone, map_location='cpu', weights_only=False)
    backbone.load_state_dict(ckpt['model_state_dict'])
    backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    print(f"[Train Heads] Backbone loaded & frozen: {args.backbone}")

    # Load data
    chunks = sorted(glob.glob(os.path.join(args.data_dir, 'chunk_*.npz')))
    if not chunks:
        raise FileNotFoundError(f"No chunks in {args.data_dir}")
    # Reserve last 2 chunks for val
    val_chunks = chunks[-2:]
    train_chunks = chunks[:-2]
    train_set = HeadsDataset(train_chunks, max_states=args.max_states)
    val_set = HeadsDataset(val_chunks)

    # Pre-compute backbone features ONCE (no backprop through backbone)
    print(f"[Train Heads] Precomputing backbone features (train + val)...")
    t0 = time.time()
    states_t_train = torch.from_numpy(train_set.states)
    train_features = extract_backbone_features(backbone, states_t_train, device)
    states_t_val = torch.from_numpy(val_set.states)
    val_features = extract_backbone_features(backbone, states_t_val, device)
    print(f"[Train Heads] Features: train {train_features.shape} | val {val_features.shape} "
          f"({time.time()-t0:.0f}s)")

    train_won = torch.from_numpy(train_set.won)
    train_mr = torch.from_numpy(train_set.moves_remaining)
    val_won = torch.from_numpy(val_set.won)
    val_mr = torch.from_numpy(val_set.moves_remaining)

    # Normalize moves_remaining for stable loss (will un-scale at inference)
    mr_mean = float(train_mr.mean())
    mr_std = float(train_mr.std() + 1e-6)
    print(f"[Train Heads] moves_remaining stats: mean={mr_mean:.2f} std={mr_std:.2f}")

    # Heads
    heads = CalibratedHeads(feature_dim=128, hidden_dim=64).to(device)
    optimizer = optim.AdamW(heads.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    n_train = len(train_features)
    best_val_combined = float('inf')
    print(f"\n[Train Heads] Starting {args.epochs} epochs (n_train={n_train:,})")

    for epoch in range(args.epochs):
        heads.train()
        idx = torch.randperm(n_train)
        tr_bce = tr_mse = n_batches = 0
        tr_correct = 0
        for start in range(0, n_train, args.batch_size):
            batch_idx = idx[start:start+args.batch_size]
            feat = train_features[batch_idx].to(device)
            y_won = train_won[batch_idx].to(device)
            y_mr = train_mr[batch_idx].to(device)

            win_logit, moves_pred = heads(feat)
            bce = F.binary_cross_entropy_with_logits(win_logit, y_won)
            mse = F.mse_loss(moves_pred, y_mr)
            loss = bce + args.moves_weight * mse

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
            optimizer.step()

            tr_bce += bce.item()
            tr_mse += mse.item()
            n_batches += 1
            with torch.no_grad():
                pred = (torch.sigmoid(win_logit) > 0.5).float()
                tr_correct += (pred == y_won).sum().item()

        scheduler.step()

        # Validation
        heads.eval()
        with torch.no_grad():
            vf = val_features.to(device)
            vw = val_won.to(device)
            vm = val_mr.to(device)
            win_logit, moves_pred = heads(vf)
            val_bce = F.binary_cross_entropy_with_logits(win_logit, vw).item()
            val_mse = F.mse_loss(moves_pred, vm).item()
            val_acc = ((torch.sigmoid(win_logit) > 0.5).float() == vw).float().mean().item()
            val_mae = (moves_pred - vm).abs().mean().item()

        tr_acc = tr_correct / n_train
        avg_bce = tr_bce / n_batches
        avg_mse = tr_mse / n_batches
        combined = val_bce + args.moves_weight * val_mse

        is_best = combined < best_val_combined
        if is_best:
            best_val_combined = combined

        print(f"  Epoch {epoch+1:>3}/{args.epochs}  "
              f"tr_bce={avg_bce:.4f} tr_acc={tr_acc*100:.1f}%  "
              f"val_bce={val_bce:.4f} val_acc={val_acc*100:.1f}% val_mae={val_mae:.2f}  "
              f"lr={scheduler.get_last_lr()[0]:.5f}"
              f"{'  * best' if is_best else ''}", flush=True)

        if is_best:
            # Save combined checkpoint: backbone + original heads + new heads
            save_dict = {
                'backbone_state_dict': ckpt['model_state_dict'],
                'heads_state_dict': heads.state_dict(),
                'mr_mean': mr_mean,
                'mr_std': mr_std,
                'epoch': epoch + 1,
                'val_bce': val_bce,
                'val_acc': val_acc,
                'val_mae': val_mae,
            }
            torch.save(save_dict, args.output)

    print(f"\n[Train Heads] Done. Best val BCE: {best_val_combined:.4f}")
    print(f"  Saved to: {args.output}")


if __name__ == '__main__':
    main()
