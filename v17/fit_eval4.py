"""Fit the 4-player leaf evaluator (crates/ludo_search/src/eval4.rs).

Conditional logit: P(seat k wins | state) = softmax_k(w · phi_k) over active
seats, phi_k = `seat_features` computed in Rust (the search uses the very same
code). Data: bot-vs-bot 4P games generated in Rust (rayon), pre-roll states
sampled with keep_prob, split by GAME into train / validation (states within a
game are correlated).

Equal-strength tables (4x depth2, 4x expectimax) dominate so that a position's
value is not confounded by which bot sits in it; a mixed table adds variety.

  python -m v17.fit_eval4 --games 40000 --out checkpoints/eval4_fit.json

Prints the Rust constant to paste into eval4.rs (E4_WEIGHTS) and the validation
log-loss of uniform / legacy (2P-fitted) / fitted evaluators.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

import alphaludo_rs

TABLES = {
    "4xdepth2": [{"bot": "depth2"}] * 4,
    "4xexpectimax": [{"bot": "expectimax"}] * 4,
    "mixed": [{"bot": "depth2"}, {"bot": "expectimax"}, {"bot": "heuristic"}, {"bot": "expert"}],
}


def generate(games: int, keep: float, seed: int, shares=(0.45, 0.35, 0.20)) -> dict:
    parts = []
    for i, ((name, seats), sh) in enumerate(zip(TABLES.items(), shares)):
        t = time.time()
        d = alphaludo_rs.eval4_dataset(seats, int(games * sh), keep_prob=keep, seed=seed + 7919 * i)
        d["game"] = d["game"].astype(np.int64) + (i + 1) * 10_000_000
        d["table"] = np.full(len(d["winner"]), i, np.int8)
        print(f"[data] {name}: {int(games * sh):,} games -> {len(d['winner']):,} states ({time.time() - t:.1f}s)",
              flush=True)
        parts.append(d)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def nll(logits: torch.Tensor, active: torch.Tensor, winner: torch.Tensor) -> torch.Tensor:
    z = logits.masked_fill(~active, -1e9)
    return torch.nn.functional.cross_entropy(z, winner, reduction="mean")


def fit(d: dict, val_frac: float = 0.1, l2: float = 1e-4, seed: int = 0):
    games = np.unique(d["game"])
    rng = np.random.default_rng(seed)
    val_games = set(rng.choice(games, int(len(games) * val_frac), replace=False).tolist())
    is_val = np.array([g in val_games for g in d["game"]])
    X = torch.from_numpy(d["feats"]).double()
    A = torch.from_numpy(d["active"])
    Y = torch.from_numpy(d["winner"].astype(np.int64))
    tr, va = torch.from_numpy(~is_val), torch.from_numpy(is_val)
    w = torch.zeros(X.shape[-1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=1.0, max_iter=500, tolerance_grad=1e-10, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = nll(X[tr] @ w, A[tr], Y[tr]) + l2 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        out = {"w": w.detach().numpy(), "is_val": is_val,
               "train_nll": float(nll(X[tr] @ w, A[tr], Y[tr])), "val_nll": float(nll(X[va] @ w, A[va], Y[va]))}
    return out


def report(d: dict, w: np.ndarray, is_val: np.ndarray, names):
    X, A, Y = d["feats"][is_val], d["active"][is_val], d["winner"][is_val].astype(np.int64)
    z = np.where(A, X @ w, -np.inf)
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    leg = d["legacy"][is_val]
    n = len(Y)
    ll = lambda P: float(-np.mean(np.log(np.clip(P[np.arange(n), Y], 1e-12, 1))))
    uni = np.where(A, 1.0 / A.sum(1, keepdims=True), 0.0)
    acc = lambda P: float(np.mean(P.argmax(1) == Y))
    print(f"[val] {n:,} states | log-loss uniform {ll(uni):.4f}  legacy {ll(leg):.4f}  eval4 {ll(p):.4f} | "
          f"top-1 legacy {acc(leg):.3f}  eval4 {acc(p):.3f}")
    t = d["t"][is_val]
    for lo, hi in ((0, .25), (.25, .5), (.5, .75), (.75, 1.01)):
        m = (t >= lo) & (t < hi)
        sub = lambda P: float(-np.mean(np.log(np.clip(P[m][np.arange(m.sum()), Y[m]], 1e-12, 1))))
        print(f"   game {lo:.2f}-{min(hi, 1):.2f}: uniform {sub(uni):.4f}  legacy {sub(leg):.4f}  eval4 {sub(p):.4f}")
    # calibration of the predicted share of the eventual winner seat, per decile
    pf, yf = p[A], (np.arange(4)[None, :] == Y[:, None])[A]
    bins = np.clip((pf * 10).astype(int), 0, 9)
    cal = [(f"{b / 10:.1f}", float(pf[bins == b].mean()), float(yf[bins == b].mean()), int((bins == b).sum()))
           for b in range(10) if (bins == b).sum() > 200]
    print("   calibration (bin, mean predicted, observed, n): " +
          "  ".join(f"{b}:{mp:.3f}/{ob:.3f}" for b, mp, ob, _ in cal))
    for nm, wi in zip(names, w):
        print(f"   {nm:18s} {wi:+.4f}")
    return {"val_ll_uniform": ll(uni), "val_ll_legacy": ll(leg), "val_ll_eval4": ll(p), "calibration": cal}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40_000)
    ap.add_argument("--keep", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=4004)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--out", default="checkpoints/eval4_fit.json")
    a = ap.parse_args()
    names, _ = alphaludo_rs.eval4_info()
    d = generate(a.games, a.keep, a.seed)
    t = time.time()
    r = fit(d, l2=a.l2)
    print(f"[fit] train NLL {r['train_nll']:.4f}  val NLL {r['val_nll']:.4f}  ({time.time() - t:.1f}s)")
    rep = report(d, r["w"], r["is_val"], names)
    w32 = [float(np.float32(x)) for x in r["w"]]
    print("pub const E4_WEIGHTS: [f32; N_E4] = [" + ", ".join(f"{x:.6}" for x in w32) + "];")
    with open(a.out, "w") as f:
        json.dump({"names": names, "weights": w32, "games": a.games, "states": int(len(d["winner"])),
                   "train_nll": r["train_nll"], **rep}, f, indent=1)


if __name__ == "__main__":
    main()
