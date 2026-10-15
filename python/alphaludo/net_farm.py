"""Drive `alphaludo_rs.NetFarm`: network-guided MCTS with batched inference.

Rust owns the games, dice, search trees and training records; this module
only answers evaluation requests with V15 GraphTransformer forward passes.
Every forward pass uses the same padded batch shape (variable shapes grow the
MPS graph cache without bound).

  python python/alphaludo/net_farm.py h2h --sims 200 --games 400
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
for p in (str(ROOT_DIR / "td_ludo"), str(ROOT_DIR / "python"), str(ROOT_DIR / "td_ludo_v15"), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch

import alphaludo_rs

CHAMPION_PATH = ROOT_DIR / "td_ludo" / "play" / "model_weights" / "v15_2" / "model_best.pt"


def load_v15(path, device, fast: bool = True, backend: str = "torch"):
    """Loads a V15 (history_len=1) checkpoint.
    backend="torch": `fast` wraps it in FastV15 (equivalent within 2e-6).
    backend="mlx":   MlxV15 float32 on the Apple GPU (equivalent within 5e-6,
                     argmax-identical; ~23% faster than PyTorch-MPS on M4)."""
    if backend == "mlx":
        from mlx_v15 import MlxV15
        return MlxV15(load_v15(path, torch.device("cpu"), fast=False))
    from td_ludo_v15.models.v15 import V15GraphTransformer
    from fast_v15 import FastV15
    m = V15GraphTransformer(d_model=128, n_heads=4, n_layers=4, ffn_dim=256, history_len=1).to(device)
    ck = torch.load(path, map_location=device, weights_only=False)
    m.load_state_dict(ck.get("model_state_dict", ck))
    m.train(False)
    return FastV15(m) if fast else m


class PolicyValuePair(torch.nn.Module):
    """Policy from one network, value from another (AlphaGo-style split)."""

    def __init__(self, policy_net, value_net):
        super().__init__()
        self.p = policy_net
        self.v = value_net

    @torch.no_grad()
    def forward(self, x, mask):
        pol, _ = self.p(x, mask)
        _, val = self.v(x, mask)
        return pol, val


def pick_device(name: str) -> torch.device:
    if name == "mlx":
        return torch.device("cpu")  # tensors stay host-side; MLX owns the GPU
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class NetServer:
    """Answers farm batches; `models` maps model id -> network.

    With `cpu_models` (same ids, copies on CPU) and cpu_frac > 0, every batch
    is split: the CPU share runs in a worker thread while the GPU share runs
    in the caller (torch releases the GIL inside ops), so both devices work.
    """

    def __init__(self, models: Dict[int, torch.nn.Module], device: torch.device, pad: int = 256,
                 cpu_models: Dict[int, torch.nn.Module] | None = None, cpu_frac: float = 0.0, cpu_pad: int = 32):
        self.models = models
        self.device = device
        self.pad = pad
        self.cpu_models = cpu_models or {}
        self.cpu_frac = cpu_frac if self.cpu_models else 0.0
        self.cpu_pad = cpu_pad
        self.evals = 0
        self.seconds = 0.0
        self._pool = None
        if self.cpu_frac > 0:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(max_workers=1)

    @staticmethod
    @torch.no_grad()
    def _run(model, x: np.ndarray, mask: np.ndarray, device, pad: int):
        n = len(x)
        pol = np.empty((n, 225), dtype=np.float32)
        val = np.empty(n, dtype=np.float32)
        is_mlx = not isinstance(model, torch.nn.Module)
        for lo in range(0, n, pad):
            k = min(pad, n - lo)
            xb = np.zeros((pad, 675), dtype=np.int8)
            mb = np.zeros((pad, 225), dtype=np.float32)
            mb[:, 0] = 1.0
            xb[:k] = x[lo:lo + k]
            mb[:k] = mask[lo:lo + k]
            if is_mlx:  # numpy in / numpy out, fixed shape (no recompiles)
                p, v = model(xb, mb)
                pol[lo:lo + k] = p[:k]
                val[lo:lo + k] = v[:k]
                continue
            xt = torch.from_numpy(xb).to(device).float().view(pad, 1, 15, 15, 3)
            mt = torch.from_numpy(mb).to(device)
            p, v = model(xt, mt)
            out = torch.cat([p[:k], v[:k, None]], dim=1).float().cpu().numpy()  # one transfer
            pol[lo:lo + k] = out[:, :225]
            val[lo:lo + k] = out[:, 225]
        return pol, val

    def answer(self, x, mask, ids):
        t0 = time.time()
        n = len(ids)
        pol = np.empty((n, 225), dtype=np.float32)
        val = np.empty(n, dtype=np.float32)
        for mid in np.unique(ids):
            sel = np.nonzero(ids == mid)[0]
            # GPU gets whole padded batches (fixed shape: no MPS graph-cache
            # growth); the CPU takes the rest concurrently. (The first version
            # rounded the CPU share DOWN to whole 128-blocks, so with ~256
            # requests per round the CPU never ran — 2026-10-02.)
            if self.cpu_frac > 0 and len(sel) > self.pad:
                n_gpu = max(self.pad, int(len(sel) * (1 - self.cpu_frac)) // self.pad * self.pad)
            else:
                n_gpu = len(sel)
            gs, cs = sel[:n_gpu], sel[n_gpu:]
            fut = None
            if len(cs):
                fut = self._pool.submit(self._run, self.cpu_models[int(mid)], x[cs], mask[cs], torch.device("cpu"), self.cpu_pad)
            if len(gs):
                p, v = self._run(self.models[int(mid)], x[gs], mask[gs], self.device, self.pad)
                pol[gs] = p
                val[gs] = v
            if fut is not None:
                p, v = fut.result()
                pol[cs] = p
                val[cs] = v
        self.evals += n
        self.seconds += time.time() - t0
        return pol, val


def run_farm(farm, server: NetServer, log_every: float = 30.0, on_progress=None):
    t0 = time.time()
    last = t0
    batches = 0
    while True:
        n = farm.collect()
        if n == 0:
            break
        x, mask, ids = farm.batch()
        pol, val = server.answer(x, mask, ids)
        farm.submit(pol, val)
        batches += 1
        if time.time() - last > log_every:
            last = time.time()
            el = last - t0
            msg = (f"  {farm.games_finished()} games | {server.evals:,} evals ({server.evals / el:,.0f}/s, "
                   f"net {100 * server.seconds / el:.0f}% of time) | {batches} batches | {el:.0f}s")
            print(msg, flush=True)
            if on_progress:
                on_progress(farm)
    return time.time() - t0


def summarize(results):
    """Win rate of config A (seat-alternated)."""
    n = len(results)
    wins = sum(1 for (_, a, w, _) in results if w == a)
    unfinished = sum(1 for r in results if r[2] < 0)
    p = wins / max(1, n)
    s0 = [r for r in results if r[1] == 0]
    s2 = [r for r in results if r[1] == 2]
    return {
        "win_rate": p, "stderr": math.sqrt(max(p * (1 - p), 1e-9) / max(1, n)), "wins": wins, "games": n,
        "seat0_wr": sum(1 for r in s0 if r[2] == 0) / max(1, len(s0)),
        "seat2_wr": sum(1 for r in s2 if r[2] == 2) / max(1, len(s2)),
        "unfinished": unfinished, "avg_moves": sum(r[3] for r in results) / max(1, n),
    }


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("h2h", help="seat A (net-MCTS or greedy) vs seat B, both answered by networks")
    h.add_argument("--a-model", default=str(CHAMPION_PATH))
    h.add_argument("--b-model", default=str(CHAMPION_PATH))
    h.add_argument("--a-kind", default="mcts", choices=("mcts", "expecti", "greedy"))
    h.add_argument("--b-kind", default="greedy", choices=("mcts", "expecti", "greedy"))
    h.add_argument("--max-evals", type=int, default=100, help="expecti: network evals per decision")
    h.add_argument("--expansions", type=int, default=4, help="expecti: expansions per round")
    h.add_argument("--by-visits", action="store_true", help="expecti: play the most-visited move")
    h.add_argument("--value-model", default=None, help="seat A takes values from this net (policy from --a-model)")
    h.add_argument("--sims", type=int, default=200)
    h.add_argument("--b-sims", type=int, default=200)
    h.add_argument("--c-puct", type=float, default=1.5)
    h.add_argument("--fpu", type=float, default=0.2)
    h.add_argument("--leaves", type=int, default=8)
    h.add_argument("--games", type=int, default=400)
    h.add_argument("--concurrency", type=int, default=128)
    h.add_argument("--seed", type=int, default=2026)
    h.add_argument("--device", default="mps")
    h.add_argument("--pad", type=int, default=256)
    h.add_argument("--cpu-frac", type=float, default=0.0, help="share of each batch run on CPU concurrently")
    a = ap.parse_args()

    dev = pick_device(a.device)
    same = Path(a.a_model).resolve() == Path(a.b_model).resolve()
    models = {0: load_v15(a.a_model, dev)}
    models[1] = models[0] if same else load_v15(a.b_model, dev)
    if a.value_model:
        models[0] = PolicyValuePair(load_v15(a.a_model, dev), load_v15(a.value_model, dev))
    common = dict(c_puct=a.c_puct, fpu_reduction=a.fpu, leaves_per_round=a.leaves, max_evals=a.max_evals,
                  expansions_per_round=a.expansions, pick_by_visits=a.by_visits)
    seat_a = dict(kind=a.a_kind, sims=a.sims, model=0, **common)
    seat_b = dict(kind=a.b_kind, sims=a.b_sims, model=1, **common)
    desc = {"greedy": "", "mcts": f"(sims={a.sims}, c={a.c_puct}, fpu={a.fpu})",
            "expecti": f"(max_evals={a.max_evals}, c={a.c_puct}, fpu={a.fpu}, by_visits={a.by_visits})"}[a.a_kind]
    print(f"[h2h] A={a.a_kind}{desc} "
          f"vs B={a.b_kind} | {a.games} games | {dev}", flush=True)
    farm = alphaludo_rs.NetFarm(seat_a, seat_b, a.games, a.concurrency, a.seed)
    cpu_models = None
    if a.cpu_frac > 0 and dev.type != "cpu":
        cpu = torch.device("cpu")
        cpu_models = {0: load_v15(a.a_model, cpu)}
        cpu_models[1] = cpu_models[0] if same else load_v15(a.b_model, cpu)
        if a.value_model:
            cpu_models[0] = PolicyValuePair(load_v15(a.a_model, cpu), load_v15(a.value_model, cpu))
    server = NetServer(models, dev, a.pad, cpu_models=cpu_models, cpu_frac=a.cpu_frac)
    secs = run_farm(farm, server)
    r = summarize(farm.results())
    print(f"[h2h] A win rate {r['win_rate']:.3f} ±{r['stderr']:.3f} ({r['wins']}/{r['games']}) "
          f"seat0 {r['seat0_wr']:.3f} seat2 {r['seat2_wr']:.3f} unfinished {r['unfinished']} "
          f"avg moves {r['avg_moves']:.0f} | {server.evals:,} evals in {secs:.0f}s", flush=True)


if __name__ == "__main__":
    main()
