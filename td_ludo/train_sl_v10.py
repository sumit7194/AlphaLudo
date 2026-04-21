"""
Joint SL training for V10 — policy + win_prob + moves_remaining heads
trained together from scratch on V6.1-teacher data.

Multi-task loss forces backbone to learn features useful for all three tasks,
unlike V6.3 where PPO-trained backbone couldn't support a retrofitted win_prob.
"""

import argparse, glob, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from td_ludo.models.v10 import AlphaLudoV10


class InMemoryDataset(Dataset):
    """Load chunks up to max_states into RAM as float32. Keeps all 5 arrays."""

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
        print(f"[Dataset] Preallocating for {total:,} samples "
              f"({(total * 28 * 15 * 15 * 4) / 1e9:.2f} GB for states)...", flush=True)
        self.states = np.empty((total, 28, 15, 15), dtype=np.float32)
        self.policies = np.empty((total, 4), dtype=np.float32)
        self.masks = np.empty((total, 4), dtype=np.float32)
        self.won = np.empty(total, dtype=np.float32)
        self.moves = np.empty(total, dtype=np.float32)

        idx = 0
        for i, (p, n) in enumerate(need):
            d = np.load(p)
            self.states[idx:idx+n] = d['states'][:n]
            self.policies[idx:idx+n] = d['policies'][:n]
            self.masks[idx:idx+n] = d['legal_masks'][:n]
            self.won[idx:idx+n] = d['won'][:n].astype(np.float32)
            self.moves[idx:idx+n] = d['moves_remaining'][:n].astype(np.float32)
            d.close()
            idx += n
            if (i + 1) % 10 == 0 or i == len(need) - 1:
                print(f"  loaded {idx:,}/{total:,}", flush=True)

        print(f"[Dataset] Loaded {len(self.states):,} samples | "
              f"win rate {self.won.mean():.3f} | avg moves_remaining {self.moves.mean():.1f}",
              flush=True)

    def __len__(self): return len(self.states)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.states[idx]),
            torch.from_numpy(self.policies[idx]),
            torch.from_numpy(self.masks[idx]),
            torch.tensor(self.won[idx], dtype=torch.float32),
            torch.tensor(self.moves[idx], dtype=torch.float32),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='checkpoints/sl_data_v10')
    parser.add_argument('--output', default='checkpoints/ac_v10/model_sl.pt')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max-states', type=int, default=250000)
    parser.add_argument('--policy-weight', type=float, default=1.0)
    parser.add_argument('--win-weight', type=float, default=0.5)
    parser.add_argument('--moves-weight', type=float, default=0.003,
                        help='SmoothL1 moves loss is raw O(10-30). At 0.003, '
                             'weighted contribution is ~0.1 — comparable to policy KL.')
    parser.add_argument('--num-res-blocks', type=int, default=6,
                        help='Residual blocks (slim V10 default 6; V6.3-size 10)')
    parser.add_argument('--num-channels', type=int, default=96,
                        help='CNN channel width (slim V10 default 96; V6.3-size 128)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing --output checkpoint (loads model+optimizer+scheduler+epoch)')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    # CUDA > CPU. MPS disabled: 2026-04-21 iter 3 showed NaN divergence at
    # LR=1e-3 on MPS (loss -> 1e26 on epoch 1) with same code+data that
    # trained cleanly on CPU. PyTorch MPS has known instability with
    # torch.log(small) + F.kl_div; not worth the ~6x speedup if it breaks.
    # If re-enabling: drop LR to 1e-4, add NaN detection, start from CPU checkpoint.
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"[V10 Train] Device: {device}")

    chunks = sorted(glob.glob(os.path.join(args.data_dir, 'chunk_*.npz')))
    if not chunks:
        raise FileNotFoundError(f"No chunks in {args.data_dir}")
    val_chunks = chunks[-2:]
    train_chunks = chunks[:-2]
    train_set = InMemoryDataset(train_chunks, max_states=args.max_states)
    val_set = InMemoryDataset(val_chunks, max_states=20000)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    model = AlphaLudoV10(num_res_blocks=args.num_res_blocks,
                          num_channels=args.num_channels,
                          in_channels=28).to(device)
    print(f"[V10 Train] Model: {model.count_parameters():,} params "
          f"({args.num_res_blocks} blocks × {args.num_channels}ch × 28in)")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader)
    )

    best_val = float('inf')
    start_epoch = 0
    if args.resume and os.path.exists(args.output):
        print(f"[V10 Train] Resuming from {args.output}")
        ckpt = torch.load(args.output, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_val = ckpt.get('best_val', float('inf'))
        print(f"[V10 Train] Resumed at epoch {start_epoch}/{args.epochs} "
              f"(best_val={best_val:.4f})")
    elif args.resume:
        print(f"[V10 Train] --resume set but {args.output} not found; starting fresh")

    n_train = len(train_set)
    print(f"\n[V10 Train] Starting epoch {start_epoch+1}/{args.epochs} "
          f"(n_train={n_train:,})")

    for epoch in range(start_epoch, args.epochs):
        # Train
        model.train()
        tr_pol = tr_win = tr_moves = 0.0
        tr_pol_acc = tr_win_acc = 0
        nb = 0
        t_ep = time.time()

        for states, policies, masks, won, moves in train_loader:
            states = states.to(device, non_blocking=True)
            policies = policies.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            won = won.to(device, non_blocking=True)
            moves = moves.to(device, non_blocking=True)

            student_policy, student_win, student_moves = model(states, masks)

            # Policy: KL divergence teacher || student
            log_student = torch.log(student_policy + 1e-8)
            pol_loss = F.kl_div(log_student, policies, reduction='batchmean', log_target=False)

            # Win: BCE (student_win is already sigmoid)
            win_loss = F.binary_cross_entropy(student_win.clamp(1e-6, 1-1e-6), won)

            # Moves: SmoothL1 (Huber @ beta=1). MSE was gradient-dominating
            # the multi-task loss because raw errors of 10-20 moves → MSE 100-400
            # crushed the policy KL at 0.4. SmoothL1 grows linearly past threshold
            # so an error of 10 contributes ~10 (not 100). Matches the pattern
            # V6.3 used for its value head.
            moves_loss = F.smooth_l1_loss(student_moves, moves)

            total = (args.policy_weight * pol_loss
                     + args.win_weight * win_loss
                     + args.moves_weight * moves_loss)

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            tr_pol += pol_loss.item()
            tr_win += win_loss.item()
            tr_moves += moves_loss.item()
            nb += 1
            with torch.no_grad():
                tr_pol_acc += (student_policy.argmax(dim=1) == policies.argmax(dim=1)).sum().item()
                tr_win_acc += ((student_win > 0.5).float() == won).sum().item()

        # Validate
        model.eval()
        vp = vw = vm = 0.0
        vp_acc = vw_acc = vmae = 0.0
        vn = 0
        with torch.no_grad():
            for states, policies, masks, won, moves in val_loader:
                states = states.to(device, non_blocking=True)
                policies = policies.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                won = won.to(device, non_blocking=True)
                moves = moves.to(device, non_blocking=True)
                sp, sw, sm = model(states, masks)
                log_student = torch.log(sp + 1e-8)
                pl = F.kl_div(log_student, policies, reduction='batchmean').item()
                wl = F.binary_cross_entropy(sw.clamp(1e-6, 1-1e-6), won).item()
                ml = F.smooth_l1_loss(sm, moves).item()  # matches training loss
                batch_n = states.size(0)
                vp += pl * batch_n
                vw += wl * batch_n
                vm += ml * batch_n
                vp_acc += (sp.argmax(dim=1) == policies.argmax(dim=1)).sum().item()
                vw_acc += ((sw > 0.5).float() == won).sum().item()
                vmae += (sm - moves).abs().sum().item()
                vn += batch_n

        tr_pol /= nb
        tr_win /= nb
        tr_moves /= nb
        tr_pol_acc_pct = tr_pol_acc / n_train * 100
        tr_win_acc_pct = tr_win_acc / n_train * 100
        v_pol = vp / vn
        v_win = vw / vn
        v_moves = vm / vn
        v_pol_acc = vp_acc / vn * 100
        v_win_acc = vw_acc / vn * 100
        v_mae = vmae / vn
        lr = scheduler.get_last_lr()[0]
        dt = time.time() - t_ep

        combined = v_pol + args.win_weight * v_win + args.moves_weight * v_moves
        is_best = combined < best_val
        if is_best: best_val = combined

        print(
            f"  E{epoch+1:>2}/{args.epochs} [{dt:.0f}s] "
            f"tr: pol={tr_pol:.3f} acc={tr_pol_acc_pct:.1f}%  "
            f"win={tr_win:.3f} acc={tr_win_acc_pct:.1f}%  mse={tr_moves:.1f}  |  "
            f"val: pol_acc={v_pol_acc:.1f}%  win_acc={v_win_acc:.1f}%  "
            f"mae={v_mae:.1f}  lr={lr:.5f}"
            + ("  * best" if is_best else ""),
            flush=True,
        )

        save = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch + 1,
            'best_val': best_val,
            'val_pol_acc': v_pol_acc,
            'val_win_acc': v_win_acc,
            'val_moves_mae': v_mae,
            'arch': {'num_res_blocks': args.num_res_blocks,
                     'num_channels': args.num_channels,
                     'in_channels': 28},
        }
        torch.save(save, args.output)

    print(f"\n[V10 Train] Done. Best val combined: {best_val:.4f}  | saved to {args.output}")


if __name__ == '__main__':
    main()
