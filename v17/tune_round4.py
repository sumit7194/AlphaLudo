"""SPSA tuning of the 4P round-search utility for PLAYING STRENGTH.

The leaf of the exact full-round search (crates/ludo_search/src/round4.rs) is a
linear "me vs the others" utility on the eval4 per-seat features:
    V_k = w · phi_k − sum_{j≠k} u · phi_j
(score_position is one point of this family). Outcome-fitted weights rank moves
poorly (greedy on calibrated win shares: 18% vs 3x Expectimax), so w and u are
tuned directly on win rate. All games run in Rust (rayon); Python only proposes
parameters. Common random numbers: both sides of a perturbation use the same farm
seed (same dice streams per game index).

Objective: mean win rate of ONE round-bot seat vs 3x each opponent in --pool
(fair share 25%). Every --check-every iterations the current point is re-measured
with --check-games per opponent and the best measured point is kept.

  python -m v17.tune_round4 --iters 150 --log checkpoints/round4_tune.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

import alphaludo_rs

N = 16
# score_position mapped onto the eval4 features (progress, base, scored, exposed)
W0 = [2.0, -0.357, 2.14, 0, -0.536, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
U0 = [1.0, -0.179, 1.07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def bot(theta, rolls):
    return {"bot": "round", "leaf": "lin", "weights": [float(v) for v in theta[:N]],
            "opp": [float(v) for v in theta[N:]], "rolls": rolls}


def winrate(theta, opp: str, games: int, seed: int, rolls: int) -> float:
    f = alphaludo_rs.Farm(["unified"], concurrency=256, seed=seed)
    o = {"bot": opp}
    f.set_matchups([{"players": [bot(theta, rolls), o, o, o], "weight": 1.0}])
    f.set_game_limit(games)
    while f.collect():
        pass
    res = f.take_results()
    return sum(1 for (_, _, ros, w, _) in res if w >= 0 and ros[w] == 0) / max(1, len(res))


def objective(theta, pool, games, seed, rolls):
    return float(np.mean([winrate(theta, o, games, seed + 101 * i, rolls) for i, o in enumerate(pool)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--games", type=int, default=1000, help="games per opponent per evaluation")
    ap.add_argument("--pool", default="expectimax,depth2")
    ap.add_argument("--rolls", type=int, default=3)
    ap.add_argument("--a", type=float, default=1.5, help="SPSA step size")
    ap.add_argument("--c", type=float, default=0.15, help="SPSA perturbation (normalised units)")
    ap.add_argument("--check-every", type=int, default=10)
    ap.add_argument("--check-games", type=int, default=3000)
    ap.add_argument("--init", default=None, help="JSON with 'theta' to start from")
    ap.add_argument("--log", default="checkpoints/round4_tune.jsonl")
    ap.add_argument("--seed", type=int, default=31)
    a = ap.parse_args()
    pool = a.pool.split(",")
    theta0 = np.array(W0 + U0, np.float64)
    if a.init:
        theta0 = np.array(json.load(open(a.init))["theta"], np.float64)
    scale = np.maximum(np.abs(theta0), 0.5)
    x = np.zeros_like(theta0)
    rng = np.random.default_rng(a.seed)
    best = {"score": -1.0}
    t0 = time.time()
    for k in range(1, a.iters + 1):
        ak = a.a / (k + 10) ** 0.602 * 11 ** 0.602
        ck = a.c / k ** 0.101
        d = rng.choice([-1.0, 1.0], size=x.shape)
        seed = int(rng.integers(1 << 30))
        fp = objective(theta0 + scale * (x + ck * d), pool, a.games, seed, a.rolls)
        fm = objective(theta0 + scale * (x - ck * d), pool, a.games, seed, a.rolls)
        g = (fp - fm) / (2 * ck) * d
        x = x + ak * g
        rec = {"iter": k, "f_plus": fp, "f_minus": fm, "step": ak, "elapsed": time.time() - t0}
        if k % a.check_every == 0 or k == a.iters:
            th = theta0 + scale * x
            sc = objective(th, pool, a.check_games, 777_000 + k, a.rolls)
            rec["check"] = sc
            rec["theta"] = th.tolist()
            if sc > best["score"]:
                best = {"score": sc, "iter": k, "theta": th.tolist()}
                with open(a.log.replace(".jsonl", "_best.json"), "w") as f:
                    json.dump(best, f, indent=1)
            print(f"[iter {k}] check {sc * 100:.2f}% (best {best['score'] * 100:.2f}% @ {best['iter']}) "
                  f"| {time.time() - t0:.0f}s", flush=True)
        with open(a.log, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
