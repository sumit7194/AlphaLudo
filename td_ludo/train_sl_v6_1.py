"""
SL Training for V6.1 — Knowledge Distillation from V6 Teacher

Trains AlphaLudoV5(in_channels=24) to match V6's soft policy distribution.

Loss: KL divergence between teacher's policy and student's policy
      + MSE on value head (optional, helps with RL transition)

Usage:
  ./td_env/bin/python3 train_sl_v6_1.py
  ./td_env/bin/python3 train_sl_v6_1.py --epochs 50 --lr 0.001
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

from src.model import AlphaLudoV5


class SLDataset(Dataset):
    """Loads all NPZ chunks from the SL data directory."""

    def __init__(self, data_dir):
        chunks = sorted(glob.glob(os.path.join(data_dir, "chunk_*.npz")))
        if not chunks:
            raise FileNotFoundError(f"No chunks found in {data_dir}")

        all_states = []
        all_policies = []
        all_values = []
        all_masks = []

        for path in chunks:
            data = np.load(path)
            all_states.append(data['states'])
            all_policies.append(data['policies'])
            all_values.append(data['values'])
            all_masks.append(data['legal_masks'])

        self.states = np.concatenate(all_states)
        self.policies = np.concatenate(all_policies)
        self.values = np.concatenate(all_values)
        self.masks = np.concatenate(all_masks)

        print(f"[SL Dataset] Loaded {len(self.states):,} states from {len(chunks)} chunks")

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
    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[SL Train] Device: {device}")

    # Load data
    dataset = SLDataset(args.data_dir)
    n_total = len(dataset)
    n_val = min(5000, int(n_total * 0.1))
    n_train = n_total - n_val

    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    print(f"[SL Train] Train: {n_train:,} | Val: {n_val:,}")

    # Create V6.1 student model
    model = AlphaLudoV5(num_res_blocks=10, num_channels=128, in_channels=24)
    model.to(device)
    print(f"[SL Train] Student: AlphaLudoV5(10 blocks, 128ch, 24in) = {model.count_parameters():,} params")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Output path
    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float('inf')
    best_path = os.path.join(args.output_dir, "model_sl_v6_1_best.pt")
    latest_path = os.path.join(args.output_dir, "model_sl_v6_1.pt")

    print(f"\n[SL Train] Starting {args.epochs} epochs...")
    print(f"  Policy loss: KL divergence (soft labels)")
    print(f"  Value loss:  MSE (weight={args.value_weight})")
    print(f"  Output: {args.output_dir}/\n")

    for epoch in range(args.epochs):
        # === Train ===
        model.train()
        train_policy_loss = 0
        train_value_loss = 0
        train_correct = 0
        train_total = 0

        for states, teacher_policies, teacher_values, masks in train_loader:
            states = states.to(device)
            teacher_policies = teacher_policies.to(device)
            teacher_values = teacher_values.to(device)
            masks = masks.to(device)

            student_policy, student_value = model(states, masks)
            student_value = student_value.squeeze(-1)

            # KL divergence: teacher || student
            # KL(P || Q) = sum(P * log(P/Q))
            log_student = torch.log(student_policy + 1e-8)
            log_teacher = torch.log(teacher_policies + 1e-8)
            policy_loss = F.kl_div(log_student, teacher_policies, reduction='batchmean',
                                    log_target=False)

            # Value MSE
            value_loss = F.mse_loss(student_value, teacher_values)

            loss = policy_loss + args.value_weight * value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_policy_loss += policy_loss.item() * states.size(0)
            train_value_loss += value_loss.item() * states.size(0)

            # Action accuracy (argmax match)
            pred_actions = student_policy.argmax(dim=1)
            teacher_actions = teacher_policies.argmax(dim=1)
            train_correct += (pred_actions == teacher_actions).sum().item()
            train_total += states.size(0)

        scheduler.step()

        # === Validate ===
        model.eval()
        val_policy_loss = 0
        val_value_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for states, teacher_policies, teacher_values, masks in val_loader:
                states = states.to(device)
                teacher_policies = teacher_policies.to(device)
                teacher_values = teacher_values.to(device)
                masks = masks.to(device)

                student_policy, student_value = model(states, masks)
                student_value = student_value.squeeze(-1)

                log_student = torch.log(student_policy + 1e-8)
                policy_loss = F.kl_div(log_student, teacher_policies, reduction='batchmean',
                                        log_target=False)
                value_loss = F.mse_loss(student_value, teacher_values)

                val_policy_loss += policy_loss.item() * states.size(0)
                val_value_loss += value_loss.item() * states.size(0)

                pred_actions = student_policy.argmax(dim=1)
                teacher_actions = teacher_policies.argmax(dim=1)
                val_correct += (pred_actions == teacher_actions).sum().item()
                val_total += states.size(0)

        # === Log ===
        train_pl = train_policy_loss / n_train
        train_vl = train_value_loss / n_train
        train_acc = train_correct / train_total * 100
        val_pl = val_policy_loss / n_val
        val_vl = val_value_loss / n_val
        val_acc = val_correct / val_total * 100
        lr = scheduler.get_last_lr()[0]

        is_best = val_pl < best_val_loss
        if is_best:
            best_val_loss = val_pl

        print(f"  Epoch {epoch+1:>3}/{args.epochs}  "
              f"tr_kl={train_pl:.4f} tr_acc={train_acc:.1f}%  "
              f"val_kl={val_pl:.4f} val_acc={val_acc:.1f}%  "
              f"val_vl={val_vl:.4f}  lr={lr:.6f}"
              f"{'  ★ best' if is_best else ''}", flush=True)

        # Save
        save_dict = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'val_policy_loss': val_pl,
            'val_accuracy': val_acc,
            'train_accuracy': train_acc,
        }
        torch.save(save_dict, latest_path)
        if is_best:
            torch.save(save_dict, best_path)

    print(f"\n[SL Train] Done. Best val KL: {best_val_loss:.4f}")
    print(f"  Best model: {best_path}")
    print(f"  Latest model: {latest_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='V6.1 SL Training (Knowledge Distillation)')
    parser.add_argument('--data-dir', default='checkpoints/sl_data_v6_1')
    parser.add_argument('--output-dir', default='checkpoints/ac_v6_1_strategic')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--value-weight', type=float, default=0.5)
    args = parser.parse_args()
    train_sl(args)
