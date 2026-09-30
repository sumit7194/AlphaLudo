#!/usr/bin/env python3
"""Generations tournament for the v136_rl_parity run (2026-06-13).

Round-robin H2H between snapshots of the SAME model taken across the whole
run (all V135ProductionAdapter 6x96 head_hidden=64), to see whether later
training produced a genuinely stronger policy or just churned.

Design:
- Greedy argmax policy (deterministic given state); dice supply the variance.
- Mirrored seeds: each pair plays N/2 games with A as P0 and N/2 with A as
  P2, sharing the dice sequence across the mirror so seat/dice luck cancels.
- CPU-only, torch single-threaded per worker, multiprocessing pool sized to
  leave cores for the live trainer. nice'd at launch. Zero GPU contention.

Usage:
    python3 gen_tournament.py --manifest gens.json --games 1000 \
        --workers 5 --out gen_tournament_results.json
manifest = {"label": "abs/path/to/ckpt.pt", ...} (insertion order = ladder
display order; intended chronological by game count).
"""
import argparse
import collections
import itertools
import json
import os
import sys

import numpy as np

# Single-threaded torch BEFORE import so each pool worker uses one core.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch  # noqa: E402
torch.set_num_threads(1)

import td_ludo_cpp as cpp  # noqa: E402
from td_ludo.models.v13_5_production import V135ProductionAdapter  # noqa: E402
from td_ludo.game.encoder_v18_production import encode_state_v18_production  # noqa: E402

MAX_MOVES_PER_GAME = 400


def _strip(sd):
    return {k.replace("_orig_mod.", "").replace("inner.inner.", "inner."): v
            for k, v in sd.items()}


def load_model(path):
    """Load any V135ProductionAdapter checkpoint — probes arch (channels,
    res-block count) from the state dict so 6x96 (V13.6) and 10x128 (V13.5)
    both work, and normalizes the key prefix (bare V135Symmetric dumps lack
    the adapter's `inner.` prefix; adapter dumps have it)."""
    import re
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    # Normalize to bare V135Symmetric keys (strip any inner. prefix), then
    # the adapter's load gets inner.-prefixed back below.
    bare = {k[len("inner."):] if k.startswith("inner.") else k: v
            for k, v in sd.items()}
    # Probe arch from bare keys.
    ci = [k for k in bare if k.endswith("conv_input.weight")]
    nc = int(bare[ci[0]].shape[0]) if ci else 96
    blocks = set()
    for k in bare:
        m = re.search(r"res_blocks\.(\d+)\.", k)
        if m:
            blocks.add(int(m.group(1)))
    nb = (max(blocks) + 1) if blocks else 6
    model = V135ProductionAdapter(num_res_blocks=nb, num_channels=nc, head_hidden=64)
    # Adapter expects inner.-prefixed keys.
    prefixed = {("inner." + k): v for k, v in bare.items()}
    missing, unexpected = model.load_state_dict(prefixed, strict=False)
    if missing or unexpected:
        print(f"[load] {os.path.basename(path)} ({nb}x{nc}): "
              f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    else:
        print(f"[load] {os.path.basename(path)} ({nb}x{nc}): clean", flush=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def pick(model, state, legal):
    if len(legal) == 1:
        return legal[0]
    enc = encode_state_v18_production(state).astype(np.float32)
    tl = np.zeros(4, dtype=np.float32)
    for a in legal:
        tl[a] = 1.0
    with torch.no_grad():
        out = model(torch.from_numpy(enc).unsqueeze(0),
                    torch.from_numpy(tl).unsqueeze(0))
        policy = out[0] if isinstance(out, tuple) else out
        a = int(policy.argmax(dim=1).item())
    return a if a in legal else legal[0]


def play_one(pick_p0, pick_p2, seed):
    import random
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    state = cpp.create_initial_state_2p()
    csix = [0, 0, 0, 0]
    mc = 0
    while not state.is_terminal and mc < MAX_MOVES_PER_GAME:
        cp = int(state.current_player)
        if not state.active_players[cp]:
            n = (cp + 1) % 4
            while not state.active_players[n]:
                n = (n + 1) % 4
            state.current_player = n
            continue
        if state.current_dice_roll == 0:
            d = random.randint(1, 6)
            if d == 6:
                csix[cp] += 1
                if csix[cp] >= 3:
                    csix[cp] = 0
                    n = (cp + 1) % 4
                    while not state.active_players[n]:
                        n = (n + 1) % 4
                    state.current_player = n
                    state.current_dice_roll = 0
                    continue
            else:
                csix[cp] = 0
            state.current_dice_roll = d
        legal = cpp.get_legal_moves(state)
        if not legal:
            n = (cp + 1) % 4
            while not state.active_players[n]:
                n = (n + 1) % 4
            state.current_player = n
            state.current_dice_roll = 0
            continue
        fn = pick_p0 if cp == 0 else pick_p2
        action = fn(state, list(legal))
        state = cpp.apply_move(state, int(action))
        mc += 1
    return int(cpp.get_winner(state)) if state.is_terminal else -1


# Worker-global model cache (one process plays one pair → loads 2 models once)
_CACHE = {}


def _get(label, path):
    if label not in _CACHE:
        _CACHE[label] = load_model(path)
    return _CACHE[label]


def run_pair(task):
    la, pa, lb, pb, games, seed_base = task
    ma, mb = _get(la, pa), _get(lb, pb)
    fa = lambda s, l: pick(ma, s, l)
    fb = lambda s, l: pick(mb, s, l)
    a_wins = b_wins = draws = 0
    half = games // 2
    for g in range(half * 2):
        a_p0 = (g % 2 == 0)
        seed = seed_base + (g // 2)
        if a_p0:
            w = play_one(fa, fb, seed); ap, bp = 0, 2
        else:
            w = play_one(fb, fa, seed); ap, bp = 2, 0
        if w == ap:
            a_wins += 1
        elif w == bp:
            b_wins += 1
        else:
            draws += 1
    _CACHE.clear()  # free models before next task
    return (la, lb, a_wins, b_wins, draws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default="gen_tournament_results.json")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))  # ordered dict (label -> path)
    labels = list(manifest.keys())
    print(f"[gen-tourney] {len(labels)} snapshots, "
          f"{len(labels)*(len(labels)-1)//2} pairs, {args.games} games/pair, "
          f"{args.workers} workers", flush=True)

    tasks = [(la, manifest[la], lb, manifest[lb], args.games, 1000 + 7 * i)
             for i, (la, lb) in enumerate(itertools.combinations(labels, 2))]

    import multiprocessing as mp
    with mp.Pool(args.workers) as pool:
        results = []
        for r in pool.imap_unordered(run_pair, tasks):
            la, lb, aw, bw, dr = r
            tot = aw + bw + dr
            print(f"  {la} vs {lb}: {la} {100*aw/tot:.1f}% "
                  f"({aw}-{bw}, draws {dr})", flush=True)
            results.append(r)

    # Win matrix + overall WR ladder (draws excluded from WR denominator)
    matrix = {la: {} for la in labels}
    wins = collections.Counter()
    decisive = collections.Counter()
    for la, lb, aw, bw, dr in results:
        dec = aw + bw
        matrix[la][lb] = round(100.0 * aw / dec, 1) if dec else None
        matrix[lb][la] = round(100.0 * bw / dec, 1) if dec else None
        wins[la] += aw; wins[lb] += bw
        decisive[la] += dec; decisive[lb] += dec
    ladder = sorted(labels, key=lambda l: wins[l] / max(1, decisive[l]), reverse=True)

    out = {
        "games_per_pair": args.games,
        "snapshots": manifest,
        "matrix_row_vs_col_winpct": matrix,
        "overall_winpct": {l: round(100.0 * wins[l] / max(1, decisive[l]), 1)
                           for l in labels},
        "ladder": ladder,
        "raw": [{"a": la, "b": lb, "a_wins": aw, "b_wins": bw, "draws": dr}
                for la, lb, aw, bw, dr in results],
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print("\n=== LADDER (overall win%, decisive games) ===", flush=True)
    for l in ladder:
        print(f"  {l:>14}: {out['overall_winpct'][l]:.1f}%", flush=True)
    print(f"\n[gen-tourney] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
