"""Champion self-play data for value learning (format v2 shards).

Both seats play the network greedily through alphaludo_rs.NetFarm (Rust games
and records, batched inference here). Every decision is recorded with the
network's policy as pi and, at game end, the outcome z. Shards of
--shard-games games are written atomically; the run resumes by counting
existing shards and stops cleanly on `touch <out>/stop` (or ./stop).

  python python/alphaludo/gen_selfplay.py --out data/selfplay_v152 --games 1000000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
for p in (str(ROOT_DIR / "td_ludo"), str(ROOT_DIR / "python"), str(ROOT_DIR / "td_ludo_v15"), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

import alphaludo_rs
from net_farm import CHAMPION_PATH, NetServer, load_v15, pick_device

HEADER = b"ALSL" + (2).to_bytes(2, "little") + (32).to_bytes(2, "little")


def write_shard(path: Path, records: bytes) -> int:
    n = len(records) // 32
    tmp = path.with_suffix(".bin.tmp")
    with open(tmp, "wb") as f:
        f.write(HEADER + n.to_bytes(4, "little") + (0).to_bytes(4, "little"))
        f.write(records)
    os.replace(tmp, path)
    return n


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def claim_next(out: Path, n_shards: int):
    for i in range(n_shards):
        if (out / f"shard_{i:05d}.bin").exists():
            continue
        claim = out / f"shard_{i:05d}.claim"
        for _ in range(2):
            try:
                fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return i
            except FileExistsError:
                try:
                    owner = int(claim.read_text() or "0")
                except (OSError, ValueError):
                    owner = 0
                if owner and _pid_alive(owner) and owner != os.getpid():
                    break          # someone is working on it
                claim.unlink(missing_ok=True)  # stale claim (dead worker): retry once
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(CHAMPION_PATH))
    ap.add_argument("--out", default="data/selfplay_v152")
    ap.add_argument("--games", type=int, default=1_000_000)
    ap.add_argument("--shard-games", type=int, default=5000)
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--cpu-frac", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=152)
    ap.add_argument("--backend", default="torch", choices=("torch", "mlx"),
                    help="mlx = MlxV15 fp32 on the GPU (exact); torch = PyTorch on --device")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap.add_argument("--tag", default="", help="label for this worker in the log")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # Backend / worker layout is NOT part of the data contract: every backend
    # used is verified equivalent to the fp32 reference (argmax-identical).
    manifest = {"format_version": 2, "source": "champion greedy self-play", "model": a.model,
                "shard_games": a.shard_games, "seed": a.seed}
    mpath = out / "manifest.json"
    if mpath.exists() and json.loads(mpath.read_text()) != manifest:
        raise SystemExit(f"{mpath} has a different config; use a new --out")
    mpath.write_text(json.dumps(manifest, indent=2))

    if a.threads > 0:
        torch.set_num_threads(a.threads)
    if a.backend == "mlx":
        dev = torch.device("cpu")
        models = {0: load_v15(a.model, dev, backend="mlx")}
        cpu_models = None
    else:
        dev = pick_device(a.device)
        models = {0: load_v15(a.model, dev)}
        cpu_models = {0: load_v15(a.model, torch.device("cpu"))} if a.cpu_frac > 0 and dev.type != "cpu" else None
    server = NetServer(models, dev, cpu_models=cpu_models, cpu_frac=a.cpu_frac if cpu_models else 0.0)
    n_shards = (a.games + a.shard_games - 1) // a.shard_games
    seat = dict(kind="greedy", model=0, record=True)
    stop = [out / "stop", ROOT_DIR / "stop"]
    t0 = time.time()
    total = 0

    # Continuous farm: `concurrency` games are ALWAYS in flight (a finished
    # game is immediately replaced), so batches never drain at a shard
    # boundary. Every time >= shard_games games have finished, their records
    # (complete games only) go to the next free shard index. Each worker draws
    # its own random seed, so workers produce independent games.
    def shards_on_disk():
        return len(list(out.glob("shard_*.bin")))

    seed = int.from_bytes(os.urandom(6), "little")
    farm = alphaludo_rs.NetFarm(seat, seat, 2_000_000_000, a.concurrency, seed)
    written_games = 0
    ts = time.time()
    last_log = ts
    while shards_on_disk() < n_shards and not any(p.exists() for p in stop):
        farm.collect()
        x, mask, ids = farm.batch()
        p, v = server.answer(x, mask, ids)
        farm.submit(p, v)
        finished = farm.games_finished()
        if time.time() - last_log > 30:
            last_log = time.time()
            print(f"{a.tag}progress: {finished:,} games finished this session "
                  f"({finished / (last_log - t0) * 60:.0f} games/min)", flush=True)
        if finished - written_games >= a.shard_games:
            recs = farm.take_records()
            idx = claim_next(out, n_shards)
            if idx is None:
                break
            n_rec = write_shard(out / f"shard_{idx:05d}.bin", recs)
            (out / f"shard_{idx:05d}.claim").unlink(missing_ok=True)
            n_games = finished - written_games
            written_games = finished
            total += n_games
            dt = time.time() - ts
            ts = time.time()
            print(f"{a.tag}shard {idx:05d}: {n_games} games, {n_rec:,} records in {dt:.0f}s "
                  f"({n_games / dt * 60:.0f} games/min) | session {total:,} games, {(time.time() - t0) / 60:.1f} min",
                  flush=True)
    print("[exit] clean", flush=True)


if __name__ == "__main__":
    main()
