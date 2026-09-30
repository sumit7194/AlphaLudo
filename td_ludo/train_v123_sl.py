"""V12.3 — Supervised Learning from bot-game dataset.

V12.3 = V12-family CNN with **engineered V17 (17-channel) input**:
  V14 minimal channels (own/opp token planes + dice 6-hot) + 3 static
  channels (safe squares, my-home, opp-home) = 17 total channels.
  All hand-crafted features that earlier models found useful.

Arch: MinimalCNN14 with `in_channels=17`, `num_res_blocks=8`,
  `num_channels=96` → ~1.4M params. Token-indexed 4-token policy.

Trained on `checkpoints/sl_dataset_v1/`. Target is the token-ID (0..3)
of the winning side's chosen token.

Usage:
    PYTHONPATH=. ./td_env/bin/python train_v123_sl.py \
        --shard-dir checkpoints/sl_dataset_v1 \
        --out-dir checkpoints/v123_sl \
        --epochs 3 --batch-size 256 --lr 3e-4 --device mps
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sl_dataset import BotGamesDataset, make_train_val_split
from experiments.distillation_14ch.model_14ch import MinimalCNN14


# ── CLI ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shard-dir", type=Path,
                   default=HERE / "checkpoints" / "sl_dataset_v1")
    p.add_argument("--out-dir", type=Path,
                   default=HERE / "checkpoints" / "v123_sl")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-end", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    # V12.3 architecture
    p.add_argument("--num-res-blocks", type=int, default=8)
    p.add_argument("--num-channels", type=int, default=96)
    p.add_argument("--in-channels", type=int, default=17,
                   help="V17 encoder is 17 channels (V14 minimal 14 + 3 static).")
    # Loss balance
    p.add_argument("--value-coeff", type=float, default=0.1)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--device", default="auto",
                   choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--resume", action="store_true",
                   help="Resume from <out_dir>/model_latest.pt (restores "
                        "weights, optimizer, step counter, LR scheduler). "
                        "Run exits at total_steps regardless of epoch.")
    return p.parse_args()


def pick_device(name):
    if name in ("cpu", "cuda", "mps"):
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Eval ────────────────────────────────────────────────────────────────

def evaluate(model, loader, device, max_batches=200):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            t = batch["target"].to(device)
            lm = batch["legal_mask"].to(device)
            out = model(x, lm)
            policy = out[0] if isinstance(out, tuple) else out
            log_p = torch.log(policy.clamp_min(1e-12))
            pred = policy.argmax(dim=-1)
            correct += (pred == t).sum().item()
            total += t.size(0)
            total_loss += -(log_p.gather(1, t.unsqueeze(1)).squeeze(1).mean()).item()
            n_batches += 1
            if n_batches >= max_batches:
                break
    model.train()
    return correct / max(1, total), total_loss / max(1, n_batches)


def main():
    args = parse_args()
    device = pick_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"=== V12.3 SL training ===")
    print(f"  device:      {device}")
    print(f"  shard_dir:   {args.shard_dir}")
    print(f"  out_dir:     {args.out_dir}")
    print(f"  arch:        MinimalCNN14  res={args.num_res_blocks}  "
          f"ch={args.num_channels}  in={args.in_channels}(V17 engineered)")
    print(f"  epochs={args.epochs}  bs={args.batch_size}  lr={args.lr}→{args.lr_end}")

    train_shards, val_shards = make_train_val_split(
        args.shard_dir, val_fraction=args.val_fraction, seed=args.seed)
    print(f"  shards:      train={len(train_shards)}, val={len(val_shards)}")

    train_ds = BotGamesDataset(args.shard_dir, "v12_3",
                                shard_indices=train_shards,
                                max_rows=args.max_rows)
    val_ds = BotGamesDataset(args.shard_dir, "v12_3",
                              shard_indices=val_shards)
    print(f"  rows:        train={len(train_ds):,}  val={len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        persistent_workers=args.num_workers > 0,
    )

    model = MinimalCNN14(
        num_res_blocks=args.num_res_blocks,
        num_channels=args.num_channels,
        in_channels=args.in_channels,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model:       {n_params:,} params")
    model.to(device).train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * (len(train_ds) // args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr_end)

    best_val_acc = 0.0
    step = 0
    log_path = args.out_dir / "train.log"
    log_f = open(log_path, "a")
    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n"); log_f.flush()
    log(f"start: total_steps={total_steps:,}")

    # ── Resume ──────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        ckpt_path = args.out_dir / "model_latest.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            step = int(ckpt.get("step", 0))
            start_epoch = int(ckpt.get("epoch", 0))
            best_val_acc = float(ckpt.get("val_acc", 0.0))
            for _ in range(step):
                scheduler.step()
            log(f"[resume] loaded step={step} epoch={start_epoch} "
                f"val_acc={best_val_acc*100:.2f}% "
                f"lr={optimizer.param_groups[0]['lr']:.2e}")
        else:
            log(f"[resume] no checkpoint at {ckpt_path} — starting fresh")

    t_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        log(f"━━ epoch {epoch+1}/{args.epochs} ━━")
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        t_epoch = time.time()

        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            t = batch["target"].to(device, non_blocking=True)
            lm = batch["legal_mask"].to(device, non_blocking=True)

            out = model(x, lm)
            # MinimalCNN14 returns (policy, win_prob, moves_remaining)
            policy = out[0]   # post-softmax, post-legal-mask
            value = out[1]    # sigmoid, [0..1]
            log_p = torch.log(policy.clamp_min(1e-12))
            gold = log_p.gather(1, t.unsqueeze(1)).squeeze(1)
            if args.label_smoothing > 0:
                denom = lm.sum(dim=-1).clamp(min=1.0)
                smooth_term = (lm * log_p).sum(dim=-1) / denom
                loss_pol = -((1 - args.label_smoothing) * gold
                             + args.label_smoothing * smooth_term).mean()
            else:
                loss_pol = -(gold).mean()
            loss_val = F.binary_cross_entropy(value, torch.ones_like(value))
            loss = loss_pol + args.value_coeff * loss_val

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss_pol.item() * t.size(0)
            pred = policy.argmax(dim=-1)
            running_correct += (pred == t).sum().item()
            running_total += t.size(0)
            step += 1

            if step % args.log_every == 0:
                acc = running_correct / max(1, running_total)
                avg_loss = running_loss / max(1, running_total)
                elapsed = time.time() - t_start
                lr_cur = optimizer.param_groups[0]["lr"]
                log(f"step {step:>6} | loss {avg_loss:.4f}  acc {acc*100:.1f}%  "
                    f"lr {lr_cur:.2e}  ({running_total/elapsed:.0f} samples/sec)")
                running_loss = 0.0
                running_correct = 0
                running_total = 0

            if step % args.save_every == 0:
                val_acc, val_loss = evaluate(model, val_loader, device)
                log(f"  [eval @ step {step}] val_acc={val_acc*100:.2f}%  val_loss={val_loss:.4f}")
                ckpt = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step, "epoch": epoch,
                    "val_acc": val_acc, "val_loss": val_loss,
                    "args": {k: (str(v) if isinstance(v, Path) else v)
                             for k, v in vars(args).items()},
                }
                torch.save(ckpt, args.out_dir / "model_latest.pt")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(ckpt, args.out_dir / "model_best.pt")
                    log(f"  [save] new best val_acc={val_acc*100:.2f}% → model_best.pt")

            if step >= total_steps:
                log(f"[exit] step {step} >= total_steps {total_steps} — stopping")
                break

        log(f"epoch {epoch+1} done in {(time.time()-t_epoch)/60:.1f} min")
        if step >= total_steps:
            break

    val_acc, val_loss = evaluate(model, val_loader, device, max_batches=10**6)
    log(f"FINAL  val_acc={val_acc*100:.2f}%  val_loss={val_loss:.4f}")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step, "val_acc": val_acc, "val_loss": val_loss,
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
    }, args.out_dir / "model_sl.pt")
    log(f"saved final → {args.out_dir / 'model_sl.pt'}")
    log_f.close()


if __name__ == "__main__":
    main()
