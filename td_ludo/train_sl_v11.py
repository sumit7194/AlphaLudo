"""
Joint SL training for V11 (ResTNet) — policy + win_prob + moves_remaining.

Mirrors train_sl_v10.py exactly except for the model class. Re-uses V10's
existing 500K mixed-teacher dataset (`checkpoints/sl_data_v10/`) since
V11 has the same input encoding (28 channels) and same head outputs.

Parity gate: target ≥73% WR vs bot mix (matching V10 SL iter 8).
- If V11 SL >= V10 SL by >=2pp → attention is helping at the SL stage
- If V11 SL ~= V10 SL → architecture is neutral; still test in RL phase
- If V11 SL << V10 SL → bug or attention is hurting; debug before RL
"""

import argparse, glob, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from td_ludo.models.v11 import AlphaLudoV11


class InMemoryDataset(Dataset):
    """Load chunks up to max_states into RAM as float32. Same as V10."""

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
    parser.add_argument('--data-dir', default='checkpoints/sl_data_v10',
                        help='Re-uses V10 SL data (28ch encoding identical)')
    parser.add_argument('--output', default='checkpoints/ac_v11/model_sl.pt')
    parser.add_argument('--epochs', type=int, default=5,
                        help='V10 iter 8 used 3 epochs; V11 starts with 5 since '
                             'transformer needs slightly more training, and V11 SL is '
                             'where we test parity vs V10.')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr-warmup-frac', type=float, default=0.05,
                        help='Fraction of total steps to spend on linear LR warmup '
                             '(transformers benefit; pure CNN did not need this).')
    parser.add_argument('--max-states', type=int, default=500000,
                        help='V10 iter 8 used 500K; matching for parity.')
    parser.add_argument('--policy-weight', type=float, default=1.0)
    parser.add_argument('--win-weight', type=float, default=0.5)
    parser.add_argument('--moves-weight', type=float, default=0.003)

    # V11 architecture knobs (defaults match the model class)
    parser.add_argument('--num-res-blocks', type=int, default=4,
                        help='V11: 4 ResBlocks (vs V10\'s 6) — 2 freed for attention')
    parser.add_argument('--num-channels', type=int, default=96)
    parser.add_argument('--num-attn-layers', type=int, default=2)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--ffn-ratio', type=int, default=4)
    parser.add_argument('--attn-dim', type=int, default=None,
                        help='Inner attention dim (Q/K/V/FFN width). If None, '
                             'matches num_channels (V11). Set <num_channels> '
                             '(e.g. 64) to add Linear projection in/out around '
                             'transformer for memory savings (V11.1).')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='SL training: 0.1 dropout in attention/FFN '
                             '(set 0 for RL to keep PPO importance ratios valid)')

    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing --output checkpoint')
    parser.add_argument('--init-from-v10', action='store_true',
                        help='Warm-start V11 backbone from V10 SL weights '
                             '(stem + first 4 ResBlocks + heads). Transformer '
                             'and pos_embed stay random-init. Faster convergence '
                             'but biases toward V10 features.')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # CUDA > MPS > CPU. MPS confirmed working for V11 attention (smoke test 2026-04-25).
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"[V11 SL] Device: {device}")

    chunks = sorted(glob.glob(os.path.join(args.data_dir, 'chunk_*.npz')))
    if not chunks:
        raise FileNotFoundError(f"No chunks in {args.data_dir}")
    val_chunks = chunks[-2:]
    train_chunks = chunks[:-2]
    train_set = InMemoryDataset(train_chunks, max_states=args.max_states)
    val_set = InMemoryDataset(val_chunks, max_states=20000)

    # pin_memory disabled: not supported on MPS and combining it with
    # non_blocking transfers caused stale/garbage reads on MPS (V11 NaN bug
    # 2026-04-25 — see git log).
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)

    model = AlphaLudoV11(
        num_res_blocks=args.num_res_blocks,
        num_channels=args.num_channels,
        num_attn_layers=args.num_attn_layers,
        num_heads=args.num_heads,
        ffn_ratio=args.ffn_ratio,
        dropout=args.dropout,
        in_channels=28,
        attn_dim=args.attn_dim,
    ).to(device)
    print(f"[V11 SL] Model: {model.count_parameters():,} params")
    print(f"[V11 SL]   {args.num_res_blocks} ResBlocks × {args.num_channels}ch")
    print(f"[V11 SL]   {args.num_attn_layers} Attn layers × {args.num_heads} heads "
          f"× FFN {args.num_channels * args.ffn_ratio}, dropout={args.dropout}")

    # Optional V10 backbone warm-start
    if args.init_from_v10:
        v10_path = 'checkpoints/ac_v10/model_sl.pt'
        if not os.path.exists(v10_path):
            print(f"[V11 SL] WARNING: --init-from-v10 set but {v10_path} not found")
        else:
            ckpt = torch.load(v10_path, map_location='cpu', weights_only=False)
            sd = ckpt.get('model_state_dict', ckpt)
            report = model.load_v10_backbone(sd)
            print(f"[V11 SL] V10 warm-start: copied {len(report['copied'])} tensors, "
                  f"skipped {len(report['skipped'])}, kept random for "
                  f"{len(report['missing'])} V11-specific params")
            model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # LR schedule: linear warmup + cosine decay (transformer best practice).
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.lr_warmup_frac * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        # Cosine decay from 1.0 down to 0.0 over remaining steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"[V11 SL] LR schedule: warmup {warmup_steps} steps "
          f"({args.lr_warmup_frac*100:.0f}%) → cosine decay over {total_steps-warmup_steps} steps")

    best_val = float('inf')
    start_epoch = 0
    if args.resume and os.path.exists(args.output):
        print(f"[V11 SL] Resuming from {args.output}")
        ckpt = torch.load(args.output, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_val = ckpt.get('best_val', float('inf'))
        print(f"[V11 SL] Resumed at epoch {start_epoch}/{args.epochs} (best_val={best_val:.4f})")
    elif args.resume:
        print(f"[V11 SL] --resume set but {args.output} not found; starting fresh")

    n_train = len(train_set)
    print(f"\n[V11 SL] Starting epoch {start_epoch+1}/{args.epochs} (n_train={n_train:,})")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        tr_pol = tr_win = tr_moves = 0.0
        tr_pol_acc = tr_win_acc = 0
        nb = 0
        t_ep = time.time()

        for states, policies, masks, won, moves in train_loader:
            states = states.to(device, non_blocking=False)
            policies = policies.to(device, non_blocking=False)
            masks = masks.to(device, non_blocking=False)
            won = won.to(device, non_blocking=False)
            moves = moves.to(device, non_blocking=False)

            student_policy, student_win, student_moves = model(states, masks)

            log_student = torch.log(student_policy + 1e-8)
            pol_loss = F.kl_div(log_student, policies, reduction='batchmean', log_target=False)

            win_loss = F.binary_cross_entropy(student_win.clamp(1e-6, 1-1e-6), won)
            moves_loss = F.smooth_l1_loss(student_moves, moves)

            total = (args.policy_weight * pol_loss
                     + args.win_weight * win_loss
                     + args.moves_weight * moves_loss)

            if torch.isnan(total) or torch.isinf(total):
                optimizer.zero_grad()
                continue

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
                states = states.to(device, non_blocking=False)
                policies = policies.to(device, non_blocking=False)
                masks = masks.to(device, non_blocking=False)
                won = won.to(device, non_blocking=False)
                moves = moves.to(device, non_blocking=False)
                sp, sw, sm = model(states, masks)
                log_student = torch.log(sp + 1e-8)
                pl = F.kl_div(log_student, policies, reduction='batchmean').item()
                wl = F.binary_cross_entropy(sw.clamp(1e-6, 1-1e-6), won).item()
                ml = F.smooth_l1_loss(sm, moves).item()
                batch_n = states.size(0)
                vp += pl * batch_n
                vw += wl * batch_n
                vm += ml * batch_n
                vp_acc += (sp.argmax(dim=1) == policies.argmax(dim=1)).sum().item()
                vw_acc += ((sw > 0.5).float() == won).sum().item()
                vmae += (sm - moves).abs().sum().item()
                vn += batch_n

        tr_pol /= max(1, nb)
        tr_win /= max(1, nb)
        tr_moves /= max(1, nb)
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
        if is_best:
            best_val = combined

        print(
            f"  E{epoch+1:>2}/{args.epochs} [{dt:.0f}s] "
            f"tr: pol={tr_pol:.3f} acc={tr_pol_acc_pct:.1f}%  "
            f"win={tr_win:.3f} acc={tr_win_acc_pct:.1f}%  smooth={tr_moves:.2f}  |  "
            f"val: pol_acc={v_pol_acc:.1f}%  win_acc={v_win_acc:.1f}%  "
            f"mae={v_mae:.1f}  lr={lr:.5f}"
            + ("  ★ best" if is_best else ""),
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
            'arch': {
                'num_res_blocks': args.num_res_blocks,
                'num_channels': args.num_channels,
                'num_attn_layers': args.num_attn_layers,
                'num_heads': args.num_heads,
                'ffn_ratio': args.ffn_ratio,
                'dropout': args.dropout,
                'in_channels': 28,
                'attn_dim': args.attn_dim,
            },
        }
        torch.save(save, args.output)

    print(f"\n[V11 SL] Done. Best val combined: {best_val:.4f}  | saved to {args.output}")
    print(f"[V11 SL] V10 SL parity gate: ≥73% WR vs bots (run eval_v10_sl.py "
          f"on this checkpoint with --num-res-blocks {args.num_res_blocks} etc.)")


if __name__ == '__main__':
    main()
