"""Search headroom: does network-guided Max^n search on top of a V17 network beat
the bare network? (2026-10-15; engine: crates/ludo_search/src/net_maxn.rs)

Each match: role 0 = the model (bare / with N simulations), other roles = an
opponent; seats shuffled per game, fixed seeds per opponent so different N face
the same dice streams. 2P fair share 50%, 4P (one model seat vs 3 copies) 25%.

  python -m v17.search_headroom checkpoints/v17_4p/rl_latest.pt --sims 0,64,200 --games 400
"""
from __future__ import annotations

import argparse
import json
import math
import time

import alphaludo_rs

from .evaluate import reference_spec
from .inference import V17Player, load_v17


def play(player, roles, games, seed, concurrency=128):
    f = alphaludo_rs.SearchFarm(roles, n_models=1, games=games, concurrency=concurrency, seed=seed)
    t = time.time()
    while f.collect():
        for m in f.models_pending():
            x, mask = f.batch(m)
            p, v = player.answer_full(x, mask)
            f.submit(m, p, v)
    res = f.take_results()
    w = sum(1 for (_, ros, win, _) in res if win >= 0 and ros[win] == 0)
    p = w / max(1, len(res))
    se_, ge_, _ = f.stats()
    return {"win_rate": p, "stderr": math.sqrt(p * (1 - p) / max(1, len(res))), "games": len(res),
            "seconds": time.time() - t, "search_evals": se_, "greedy_evals": ge_}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--sims", default="0,64,200")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--opponents", default="2p:mcts3000,2p:self,4p:teacher,4p:self")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    player = V17Player(load_v17(a.checkpoint), backend="mlx")
    out = {}
    for i, opp in enumerate(a.opponents.split(",")):
        mode, name = opp.split(":")
        n = 2 if mode == "2p" else 4
        o = {"greedy": 0} if name == "self" else (reference_spec(name if mode == "2p" else "4p:" + name))
        for sims in [int(v) for v in a.sims.split(",")]:
            me = {"greedy": 0} if sims == 0 else {"search": 0, "sims": sims, "c": a.c}
            r = play(player, [me] + [o] * (n - 1), a.games, seed=9000 + 100 * i)
            out[f"{opp}@{sims}"] = r
            print(f"{opp:14s} sims {sims:5d}: {100 * r['win_rate']:5.1f}% ±{100 * r['stderr']:.1f} "
                  f"(n={r['games']}, {r['seconds']:.0f}s, {r['search_evals'] / max(1, r['games']):.0f} evals/game)",
                  flush=True)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
