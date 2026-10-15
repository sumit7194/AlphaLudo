"""Train a dedicated VALUE network for search (AlphaGo-style policy/value split).

Why: the v15.2 champion's value head cannot rank sibling moves — choosing by
1-ply exact-dice expectation of its value scores 34.5–35% against its own
greedy policy (2026-10-02). Its policy stays the move proposer; this script
trains a separate value net (same V15 architecture, initialised from the
champion, every weight trainable, value loss only) on champion self-play
outcomes, so V approximates the champion policy's own E[z | s].

Data: format-v2 shards from gen_selfplay.py. Positions within a game are
highly correlated, so each epoch samples --per-game positions per game on
average (uniform over records with probability per_game / game_length).
Validation: the first shard (held out by game). Metric that matters is
downstream — `value1ply` strength vs the champion (net_farm.py h2h with
--value-model) — but log-loss / AUC are tracked here.

Resumable: <out>/value_latest.pt holds model, optimizer, step, epoch.
`touch <out>/stop` stops cleanly.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
for p in (str(ROOT_DIR / "td_ludo"), str(ROOT_DIR / "python"), str(ROOT_DIR / "td_ludo_v15"), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from net_farm import CHAMPION_PATH, pick_device
from sl_dataset_v153 import V153TeacherDataset, open_shard
from td_ludo_v15.models.v15 import V15GraphTransformer
from train_v153_sl import save_atomic, write_json_atomic


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    npos = labels.sum()
    return float((ranks[labels == 1].sum() - npos * (npos + 1) / 2) / max(1, npos * (len(labels) - npos)))


@torch.no_grad()
def validate(model, loader, device):
    was = model.training
    model.train(False)
    vs, zs = [], []
    for b in loader:
        _, vl = model.forward_logits(b["x"].to(device), b["legal_mask"].to(device))
        vs.append(torch.sigmoid(vl).float().cpu().numpy())
        zs.append(b["target_value"].numpy())
    model.train(was)
    v = np.concatenate(vs)
    z = np.concatenate(zs)
    keep = z != 0
    v, y = v[keep], (z[keep] > 0).astype(np.float64)
    ll = float(-np.mean(y * np.log(np.clip(v, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - v, 1e-6, 1))))
    return {"val_logloss": ll, "val_auc": auc(v, y), "val_n": int(keep.sum())}


def per_game_sample(ds: V153TeacherDataset, per_game: float, rng) -> np.ndarray:
    """Indices into ds: each record kept with prob per_game / its game's length."""
    keep = []
    for si, recs in enumerate(ds.shards):
        items, offs = ds.shard_items(si)
        gid = recs["game_id"][offs]
        _, inv, counts = np.unique(gid, return_inverse=True, return_counts=True)
        p = np.minimum(1.0, per_game / counts[inv])
        keep.append(items[rng.random(len(offs)) < p])
    return np.concatenate(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/selfplay_v152",
                    help="shard dir(s), comma-separated; validation = first shard of the FIRST dir")
    ap.add_argument("--out", default="checkpoints/value_v1")
    ap.add_argument("--init", default=str(CHAMPION_PATH))
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--per-game", type=float, default=20.0)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--micro-batch", type=int, default=0,
                    help="split each batch into micro-batches and accumulate gradients (identical update, "
                         "lower peak memory; 0 = no split)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-records", type=int, default=50_000)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--save-every-min", type=float, default=10.0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = pick_device(a.device)
    stop = [out / "stop", ROOT_DIR / "stop"]
    dirs = [Path(d) for d in a.data.split(",") if d]
    first = sorted(dirs[0].glob("shard_*.bin"))
    if len(first) < 2:
        raise SystemExit(f"need >= 2 shards in {dirs[0]}")
    # The held-out shard stays fixed (first dir, shard 0) so validation numbers
    # are comparable across rounds that add more data.
    shards = first + [s for d in dirs[1:] for s in sorted(d.glob("shard_*.bin"))]
    val_ds = V153TeacherDataset(shard_files=shards[:1], movers="all", skip_forced=False, policy_target="action")
    rng = np.random.default_rng(0)
    vpick = np.sort(rng.choice(len(val_ds), min(a.val_records, len(val_ds)), replace=False))
    val_loader = DataLoader(Subset(val_ds, vpick.tolist()), batch_size=a.batch_size, num_workers=a.num_workers)
    train_ds = V153TeacherDataset(shard_files=shards[1:], movers="all", skip_forced=False, policy_target="action")
    n_games = sum(len(np.unique(open_shard(s)[1]["game_id"])) for s in shards[1:])
    samples_per_epoch = int(n_games * a.per_game)
    total_steps = int(a.epochs * samples_per_epoch / a.batch_size)
    print(f"[data] {len(shards) - 1} train shards, {n_games:,} games, {len(train_ds):,} records; "
          f"{samples_per_epoch:,} samples/epoch, {total_steps:,} steps | device {dev}", flush=True)

    model = V15GraphTransformer(d_model=128, n_heads=4, n_layers=4, ffn_dim=256, history_len=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    step, epoch, best_ll = 0, 0, float("inf")
    epoch_step = 0  # batches already consumed in the current epoch
    latest = out / "value_latest.pt"
    if latest.exists():
        ck = torch.load(latest, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["optimizer_state_dict"])
        step, epoch, best_ll = ck["step"], ck["epoch"], ck.get("best_ll", float("inf"))
        # Older checkpoints lack epoch_step; every step before them was in
        # epoch 0 or later epochs of known size, so derive it from `step`.
        epoch_step = ck.get("epoch_step", step - sum(ck.get("epoch_sizes", [])))
        print(f"[resume] step {step}, epoch {epoch}, {epoch_step} batches of this epoch already used", flush=True)
    else:
        ck = torch.load(a.init, map_location=dev, weights_only=False)
        model.load_state_dict(ck.get("model_state_dict", ck))
        v0 = validate(model, val_loader, dev)
        print(f"[init] champion value head: {v0}", flush=True)
        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps({"step": 0, **v0}) + "\n")

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        t = min(1.0, (s - a.warmup) / max(1, total_steps - a.warmup))
        return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * t))

    def save(path, extra=None):
        save_atomic({"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(),
                     "step": step, "epoch": epoch, "epoch_step": epoch_step, "epoch_sizes": epoch_sizes,
                     "best_ll": best_ll, "args": vars(a), **(extra or {})}, path)

    epoch_sizes = ck.get("epoch_sizes", []) if latest.exists() else []
    model.train(True)
    last_save = time.time()
    done = False
    while step < total_steps and not done:
        idx = per_game_sample(train_ds, a.per_game, np.random.default_rng(1000 + epoch))
        rng_e = np.random.default_rng(2000 + epoch)
        rng_e.shuffle(idx)
        n_keep = len(idx) // a.batch_size * a.batch_size
        # Resume mid-epoch: the epoch's sample order is fixed by its seed, so
        # skip exactly the batches already trained on (no repeats, no gaps).
        start = min(epoch_step, n_keep // a.batch_size) * a.batch_size
        loader = DataLoader(Subset(train_ds, idx[start:n_keep]), batch_size=a.batch_size, shuffle=False,
                            num_workers=a.num_workers, drop_last=True)
        t0, agg, nb = time.time(), 0.0, 0
        for b in loader:
            if any(p.exists() for p in stop):
                done = True
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            z_all = b["target_value"]
            # Normalise by the FULL batch's finished-game count, so summing the
            # micro-batch gradients gives exactly the full-batch gradient.
            denom = (z_all != 0).sum().clamp(min=1).to(dev).float()
            mb = a.micro_batch if 0 < a.micro_batch < len(z_all) else len(z_all)
            opt.zero_grad(set_to_none=True)
            loss = 0.0
            for lo in range(0, len(z_all), mb):
                z = z_all[lo:lo + mb].to(dev)
                fin = z != 0
                _, vl = model.forward_logits(b["x"][lo:lo + mb].to(dev), b["legal_mask"][lo:lo + mb].to(dev))
                part = (F.binary_cross_entropy_with_logits(vl, (z + 1) * 0.5, reduction="none") * fin).sum() / denom
                part.backward()
                loss = loss + part.detach()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            epoch_step += 1
            agg += loss.detach()
            nb += 1
            if step % 100 == 0:
                print(f"step {step:6d}/{total_steps} | epoch {epoch} | lr {lr_at(step):.1e} | train BCE {float(agg) / nb:.4f} "
                      f"| {nb * a.batch_size / (time.time() - t0):,.0f} samples/s", flush=True)
                agg, nb, t0 = 0.0, 0, time.time()
            if step % a.eval_every == 0:
                m = validate(model, val_loader, dev)
                m["step"] = step
                print(f"[val] {m}", flush=True)
                with open(out / "metrics.jsonl", "a") as f:
                    f.write(json.dumps(m) + "\n")
                if m["val_logloss"] < best_ll:
                    best_ll = m["val_logloss"]
                    save(out / "value_best.pt", {"val": m})
                save(latest)
                last_save = time.time()
                if dev.type == "mps":
                    torch.mps.empty_cache()
            if time.time() - last_save > a.save_every_min * 60:
                save(latest)
                last_save = time.time()
            if step >= total_steps:
                break
        else:
            epoch_sizes.append(epoch_step)
            epoch += 1
            epoch_step = 0
    m = validate(model, val_loader, dev)
    m["step"] = step
    print(f"[val final] {m}", flush=True)
    with open(out / "metrics.jsonl", "a") as f:
        f.write(json.dumps(m) + "\n")
    if m["val_logloss"] < best_ll:
        best_ll = m["val_logloss"]
        save(out / "value_best.pt", {"val": m})
    save(latest)
    print("[exit] clean", flush=True)


if __name__ == "__main__":
    main()
