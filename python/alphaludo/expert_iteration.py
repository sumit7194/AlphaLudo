"""Expert iteration / AlphaZero loop for 2-player Ludo, starting from v15.2.

Each iteration:
  1. SELF-PLAY  — the current network plays itself through network-guided
     Expecti-MCTS (alphaludo_rs.NetFarm, Rust trees, batched inference here).
     Every searched decision is recorded with its visit distribution pi and,
     at game end, the outcome z. Root Dirichlet noise + sampled play for the
     first few decisions give diversity; labels are always the argmax/pi.
  2. TRAIN      — the network is trained (continuing from itself) on a replay
     window of recent self-play shards: soft cross-entropy to pi + BCE value
     to z (forward_logits; same losses as train_v153_sl).
  3. GATE       — candidate (greedy) vs current best (greedy), common random
     numbers. Promoted if it wins >= --gate. Also scored vs the fixed v15.2
     champion every iteration, so progress is anchored to an external
     reference rather than to itself.

Everything is resumable: run state lives in <run>/state.json, self-play
shards in <run>/data, networks in <run>/nets. `touch <run>/stop` (or
./stop at the repo root) stops cleanly at the next safe point; re-running the
same command resumes. Designed to run detached (nohup) — see
launch_expert_iteration.sh.
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
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

import alphaludo_rs
from fast_v15 import FastV15
from net_farm import CHAMPION_PATH, NetServer, pick_device, summarize
from sl_dataset_v153 import V153TeacherDataset
from td_ludo_v15.models.v15 import V15GraphTransformer
from train_v153_sl import compute_losses, save_atomic, write_json_atomic

HEADER = b"ALSL" + (2).to_bytes(2, "little") + (32).to_bytes(2, "little")


def new_net(device):
    return V15GraphTransformer(d_model=128, n_heads=4, n_layers=4, ffn_dim=256, history_len=1).to(device)


def load_net(path, device):
    m = new_net(device)
    ck = torch.load(path, map_location=device, weights_only=False)
    m.load_state_dict(ck.get("model_state_dict", ck))
    return m


def write_shard(path: Path, records: bytes):
    n = len(records) // 32
    tmp = path.with_suffix(".bin.tmp")
    with open(tmp, "wb") as f:
        f.write(HEADER + n.to_bytes(4, "little") + (0).to_bytes(4, "little"))
        f.write(records)
    os.replace(tmp, path)
    return n


class Run:
    def __init__(self, args):
        self.a = args
        self.dir = Path(args.run_dir)
        (self.dir / "data").mkdir(parents=True, exist_ok=True)
        (self.dir / "nets").mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.log_path = self.dir / "metrics.jsonl"
        self.dev = pick_device(args.device)
        self.cpu = torch.device("cpu")
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
            print(f"[resume] iteration {self.state['iteration']} phase {self.state['phase']}", flush=True)
        else:
            best = self.dir / "nets" / "best.pt"
            init = load_net(args.init, self.cpu)
            save_atomic({"model_state_dict": init.state_dict(), "source": str(args.init)}, best)
            self.state = {"iteration": 0, "phase": "selfplay", "best": str(best), "promotions": 0,
                          "games_total": 0, "records_total": 0, "train_steps": 0, "args": vars(args)}
            self.save_state()

    def save_state(self):
        write_json_atomic(self.state_path, self.state)

    def stop_requested(self):
        return (self.dir / "stop").exists() or (ROOT_DIR / "stop").exists()

    def log(self, entry):
        entry["time"] = time.time()
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print("[metrics] " + json.dumps(entry), flush=True)

    def server_for(self, nets):
        gpu = {i: FastV15(n.to(self.dev).train(False)) for i, n in nets.items()}
        cpu_m = None
        if self.a.cpu_frac > 0 and self.dev.type != "cpu":
            cpu_m = {i: FastV15(load_copy(n, self.cpu)) for i, n in nets.items()}
        return NetServer(gpu, self.dev, self.a.pad, cpu_models=cpu_m, cpu_frac=self.a.cpu_frac)

    # ── 1. self-play ─────────────────────────────────────────────────────────
    def selfplay(self, it):
        a = self.a
        best = load_net(self.state["best"], self.dev)
        mc = dict(kind="mcts", sims=a.sims, c_puct=a.c_puct, fpu_reduction=a.fpu, leaves_per_round=a.leaves,
                  dirichlet_eps=a.dirichlet, model=0, record=True, explore_moves=a.explore_moves)
        farm = alphaludo_rs.NetFarm(mc, mc, a.games_per_iter, a.concurrency, a.seed + 7919 * it)
        server = self.server_for({0: best})
        t0 = time.time()
        last = t0
        stopped = False
        while True:
            n = farm.collect()
            if n == 0:
                break
            x, mask, ids = farm.batch()
            p, v = server.answer(x, mask, ids)
            farm.submit(p, v)
            if time.time() - last > 60:
                last = time.time()
                print(f"  [selfplay it{it}] {farm.games_finished()}/{a.games_per_iter} games | "
                      f"{server.evals / (last - t0):,.0f} evals/s", flush=True)
                if self.stop_requested():
                    stopped = True
                    break
        recs = farm.take_records()
        shard = self.dir / "data" / f"shard_{it:05d}.bin"
        n_rec = write_shard(shard, recs) if recs else 0
        res = farm.results()
        self.state["games_total"] += len(res)
        self.state["records_total"] += n_rec
        self.log({"iteration": it, "phase": "selfplay", "games": len(res), "records": n_rec,
                  "seconds": time.time() - t0, "evals": server.evals, "stopped_early": stopped,
                  "avg_moves": float(np.mean([r[3] for r in res])) if res else 0.0})
        return not stopped

    # ── 2. train ─────────────────────────────────────────────────────────────
    def train(self, it):
        a = self.a
        shards = sorted((self.dir / "data").glob("shard_*.bin"))[-a.window:]
        ds = V153TeacherDataset(shard_files=shards, movers="all", skip_forced=True, policy_target="pi")
        n_steps = max(1, int(a.epochs * len(ds) / a.batch_size))
        model = load_net(self.state["best"], self.dev)
        model.train(True)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        sampler = RandomSampler(ds, replacement=True, num_samples=n_steps * a.batch_size)
        loader = DataLoader(ds, batch_size=a.batch_size, sampler=sampler, num_workers=a.num_workers, drop_last=True)
        t0 = time.time()
        agg = {"policy_loss": 0.0, "value_loss": 0.0, "correct": 0, "n_multi": 0}
        for step, batch in enumerate(loader, 1):
            loss, m = compute_losses(model, batch, self.dev, a.value_coef)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k in agg:
                agg[k] += m[k]
        if self.dev.type == "mps":
            torch.mps.empty_cache()
        cand = self.dir / "nets" / f"cand_{it:05d}.pt"
        save_atomic({"model_state_dict": model.state_dict(), "iteration": it}, cand)
        self.state["train_steps"] += n_steps
        self.log({"iteration": it, "phase": "train", "records": len(ds), "shards": len(shards), "steps": n_steps,
                  "policy_ce": float(agg["policy_loss"]) / n_steps, "value_bce": float(agg["value_loss"]) / n_steps,
                  "top1": int(agg["correct"]) / max(1, int(agg["n_multi"])), "seconds": time.time() - t0})
        return str(cand)

    # ── 3. gate ──────────────────────────────────────────────────────────────
    def h2h_greedy(self, a_net, b_net, games, seed):
        farm = alphaludo_rs.NetFarm(dict(kind="greedy", model=0), dict(kind="greedy", model=1), games,
                                    self.a.concurrency, seed)
        server = self.server_for({0: a_net, 1: b_net})
        while True:
            n = farm.collect()
            if n == 0:
                break
            x, mask, ids = farm.batch()
            p, v = server.answer(x, mask, ids)
            farm.submit(p, v)
        return summarize(farm.results())

    def gate(self, it, cand_path):
        a = self.a
        cand = load_net(cand_path, self.dev)
        best = load_net(self.state["best"], self.dev)
        champ = load_net(CHAMPION_PATH, self.dev)
        t0 = time.time()
        vs_best = self.h2h_greedy(cand, best, a.gate_games, seed=1000)  # fixed seed: common random numbers
        vs_champ = self.h2h_greedy(cand, champ, a.gate_games, seed=2000)
        promoted = vs_best["win_rate"] >= a.gate
        if promoted:
            best_path = self.dir / "nets" / "best.pt"
            save_atomic({"model_state_dict": cand.state_dict(), "iteration": it,
                         "vs_champion": vs_champ, "vs_prev_best": vs_best}, best_path)
            save_atomic({"model_state_dict": cand.state_dict(), "iteration": it, "vs_champion": vs_champ},
                        self.dir / "nets" / f"best_{it:05d}.pt")
            self.state["best"] = str(best_path)
            self.state["promotions"] += 1
        self.log({"iteration": it, "phase": "gate", "vs_best": vs_best["win_rate"], "vs_best_se": vs_best["stderr"],
                  "vs_champion": vs_champ["win_rate"], "vs_champion_se": vs_champ["stderr"],
                  "promoted": promoted, "promotions": self.state["promotions"], "seconds": time.time() - t0})

    def loop(self):
        while self.state["iteration"] < self.a.iterations:
            it = self.state["iteration"]
            if self.stop_requested():
                print("[stop] stop file present — exiting cleanly", flush=True)
                return
            ph = self.state["phase"]
            if ph == "selfplay":
                complete = self.selfplay(it)
                if not complete:
                    # Partial shard kept (finished games only); repeat the phase on resume.
                    self.save_state()
                    print("[stop] during self-play — resume will continue this iteration", flush=True)
                    return
                self.state["phase"] = "train"
                self.save_state()
            elif ph == "train":
                self.state["candidate"] = self.train(it)
                self.state["phase"] = "gate"
                self.save_state()
            elif ph == "gate":
                self.gate(it, self.state["candidate"])
                self.state["iteration"] = it + 1
                self.state["phase"] = "selfplay"
                self.save_state()


def load_copy(net, device):
    m = new_net(device)
    m.load_state_dict({k: v.detach().to(device) for k, v in net.state_dict().items()})
    m.train(False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="checkpoints/expit_v1")
    ap.add_argument("--init", default=str(CHAMPION_PATH))
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--games-per-iter", type=int, default=1000)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--fpu", type=float, default=0.2)
    ap.add_argument("--leaves", type=int, default=8)
    ap.add_argument("--dirichlet", type=float, default=0.25)
    ap.add_argument("--explore-moves", type=int, default=6)
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--window", type=int, default=8, help="replay window in self-play shards (iterations)")
    ap.add_argument("--epochs", type=float, default=1.0, help="passes over the window per iteration")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--gate", type=float, default=0.5)
    ap.add_argument("--gate-games", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=31337)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--pad", type=int, default=256)
    ap.add_argument("--cpu-frac", type=float, default=0.3)
    args = ap.parse_args()
    Run(args).loop()


if __name__ == "__main__":
    main()
