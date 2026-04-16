"""
SL Training for V6.3 — Policy Distillation from V6.1 Teacher

Trains AlphaLudoV63 (random init, 27 channels) to match V6.1's policy
distribution via KL divergence + value regression on actual game outcomes.

Key differences from train_sl_v6_1.py:
  - Student: AlphaLudoV63 (in_channels=27, has aux head that we ignore here)
  - Value target: actual game outcome (+1/-1), not teacher's value prediction
  - Output: checkpoint compatible with train_v6_3.py --resume

Usage:
  python3 train_sl_v6_3.py --epochs 10 --lr 1e-3
"""

import os
import sys
import time
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from td_ludo.models.v6_3 import AlphaLudoV63


class InMemorySLDataset(Dataset):
    """Loads a subset of NPZ chunks fully into RAM as float32.

    With 14GB VM RAM, ~500K samples fit (~6GB at float32). Uses all chunks
    if the total fits under max_ram_gb, otherwise subsamples.
    """

    def __init__(self, chunk_paths, max_states=500000):
        # First pass: determine which chunks to use and how much from each
        chunks_to_use = []
        running = 0
        for path in chunk_paths:
            d = np.load(path)
            n = len(d['states'])
            d.close()
            if running + n > max_states:
                remaining = max_states - running
                if remaining > 0:
                    chunks_to_use.append((path, remaining))
                    running += remaining
                break
            chunks_to_use.append((path, n))
            running += n

        total = running
        print(f"[SL Dataset] Preallocating for {total:,} states "
              f"({(total*27*15*15*4)/1e9:.2f} GB for states)...", flush=True)

        # Preallocate to avoid 2x memory peak during concat
        self.states = np.empty((total, 27, 15, 15), dtype=np.float32)
        self.policies = np.empty((total, 4), dtype=np.float32)
        self.values = np.empty(total, dtype=np.float32)
        self.masks = np.empty((total, 4), dtype=np.float32)

        idx = 0
        for i, (path, take_n) in enumerate(chunks_to_use):
            d = np.load(path)
            self.states[idx:idx+take_n] = d['states'][:take_n]
            self.policies[idx:idx+take_n] = d['policies'][:take_n]
            self.values[idx:idx+take_n] = d['values'][:take_n]
            self.masks[idx:idx+take_n] = d['legal_masks'][:take_n]
            d.close()
            idx += take_n
            if (i + 1) % 10 == 0 or i == len(chunks_to_use) - 1:
                print(f"  ... loaded {idx:,}/{total:,} states", flush=True)

        assert self.states.shape[1] == 27
        ram_gb = (self.states.nbytes + self.policies.nbytes + self.values.nbytes
                  + self.masks.nbytes) / 1e9
        print(f"[SL Dataset] Loaded {len(self.states):,} states "
              f"from {len(chunks_to_use)} chunks ({ram_gb:.2f} GB RAM)", flush=True)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.states[idx]),
            torch.from_numpy(self.policies[idx]),
            torch.tensor(self.values[idx], dtype=torch.float32),
            torch.from_numpy(self.masks[idx]),
        )


def train_sl(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[SL Train] Device: {device}")

    # Load data — chunk-level train/val split, in-RAM for speed
    chunks = sorted(glob.glob(os.path.join(args.data_dir, "chunk_*.npz")))
    if not chunks:
        raise FileNotFoundError(f"No chunks found in {args.data_dir}")
    # Reserve last 2 chunks (~20K states) for validation
    val_chunks = chunks[-2:]
    train_chunks = chunks[:-2]

    train_set = InMemorySLDataset(train_chunks, max_states=args.max_train_states)
    val_set = InMemorySLDataset(val_chunks, max_states=20000)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    n_train = len(train_set)
    n_val = len(val_set)
    print(f"[SL Train] Train: {n_train:,} | Val: {n_val:,}")

    # Create V6.3 student model (random init)
    model = AlphaLudoV63(num_res_blocks=10, num_channels=128, in_channels=27)
    model.to(device)
    print(f"[SL Train] Student: AlphaLudoV63(10 blocks, 128ch, 27in) = "
          f"{model.count_parameters():,} params")

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader)
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float('inf')
    best_path = os.path.join(args.output_dir, "model_sl.pt")
    latest_path = os.path.join(args.output_dir, "model_sl_latest.pt")

    print(f"\n[SL Train] Starting {args.epochs} epochs...")
    print(f"  Policy loss: KL divergence (soft labels from V6.1)")
    print(f"  Value loss:  SmoothL1 on game outcome (weight={args.value_weight})")
    print(f"  Output: {args.output_dir}/\n")

    for epoch in range(args.epochs):
        # === Train ===
        model.train()
        train_pl_sum = 0.0
        train_vl_sum = 0.0
        train_correct = 0
        train_total = 0
        t_epoch = time.time()

        for states, teacher_policies, outcomes, masks in train_loader:
            states = states.to(device, non_blocking=True)
            teacher_policies = teacher_policies.to(device, non_blocking=True)
            outcomes = outcomes.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # V6.3 model returns (policy, value, aux) — ignore aux for SL
            student_policy, student_value, _ = model(states, masks)
            student_value = student_value.squeeze(-1)

            # KL(teacher || student) = sum(teacher * log(teacher/student))
            log_student = torch.log(student_policy + 1e-8)
            policy_loss = F.kl_div(
                log_student, teacher_policies,
                reduction='batchmean', log_target=False
            )

            # Value regression on outcome
            value_loss = F.smooth_l1_loss(student_value, outcomes)

            loss = policy_loss + args.value_weight * value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_pl_sum += policy_loss.item() * states.size(0)
            train_vl_sum += value_loss.item() * states.size(0)

            pred_actions = student_policy.argmax(dim=1)
            teacher_actions = teacher_policies.argmax(dim=1)
            train_correct += (pred_actions == teacher_actions).sum().item()
            train_total += states.size(0)

        # === Validate ===
        model.eval()
        val_pl_sum = 0.0
        val_vl_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for states, teacher_policies, outcomes, masks in val_loader:
                states = states.to(device, non_blocking=True)
                teacher_policies = teacher_policies.to(device, non_blocking=True)
                outcomes = outcomes.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                student_policy, student_value, _ = model(states, masks)
                student_value = student_value.squeeze(-1)

                log_student = torch.log(student_policy + 1e-8)
                policy_loss = F.kl_div(
                    log_student, teacher_policies,
                    reduction='batchmean', log_target=False
                )
                value_loss = F.smooth_l1_loss(student_value, outcomes)

                val_pl_sum += policy_loss.item() * states.size(0)
                val_vl_sum += value_loss.item() * states.size(0)

                pred_actions = student_policy.argmax(dim=1)
                teacher_actions = teacher_policies.argmax(dim=1)
                val_correct += (pred_actions == teacher_actions).sum().item()
                val_total += states.size(0)

        train_pl = train_pl_sum / train_total
        train_vl = train_vl_sum / train_total
        train_acc = train_correct / train_total * 100
        val_pl = val_pl_sum / val_total
        val_vl = val_vl_sum / val_total
        val_acc = val_correct / val_total * 100
        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t_epoch

        is_best = val_pl < best_val_loss
        if is_best:
            best_val_loss = val_pl

        print(
            f"  Epoch {epoch+1:>3}/{args.epochs} [{elapsed:.0f}s]  "
            f"tr_kl={train_pl:.4f} tr_acc={train_acc:.1f}%  "
            f"val_kl={val_pl:.4f} val_acc={val_acc:.1f}%  "
            f"val_vl={val_vl:.4f}  lr={lr:.6f}"
            f"{'  * best' if is_best else ''}",
            flush=True,
        )

        # Save checkpoint — format compatible with train_v6_3.py --resume
        # Must include return stats for RL resume to work cleanly
        save_dict = {
            'model_state_dict': model.state_dict(),
            'total_games': 0,
            'total_updates': 0,
            'best_win_rate': 0.0,
            'return_running_mean': 0.0,
            'return_running_std': 1.0,
            'epoch': epoch + 1,
            'val_policy_loss': val_pl,
            'val_accuracy': val_acc,
            'train_accuracy': train_acc,
        }
        torch.save(save_dict, latest_path)
        if is_best:
            torch.save(save_dict, best_path)

    print(f"\n[SL Train] Done. Best val KL: {best_val_loss:.4f}")
    print(f"  Best SL model: {best_path}  <- use this as RL starting point")
    print(f"  Latest: {latest_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='V6.3 SL Training (Policy Distillation)')
    parser.add_argument('--data-dir', default='checkpoints/sl_data_v6_3')
    parser.add_argument('--output-dir', default='checkpoints/ac_v6_3_capture')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--value-weight', type=float, default=0.5)
    parser.add_argument('--max-train-states', type=int, default=500000,
                        help='Cap training samples (controls RAM: ~12MB per 1K samples at fp32)')
    args = parser.parse_args()
    train_sl(args)
