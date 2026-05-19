"""V15.1-era H2H tournament — 4-way round-robin.

Contestants (latest of each generation):
  1. V13.5      — pre-shaping-experiment RL latest (v135_prod_rl_local/model_latest.pt).
                  V135ProductionAdapter arch.
  2. V13.5_exp  — post-shaping-experiment RL latest (v135_shaping_exp/model_latest.pt).
                  Same arch.
  3. V15        — V15 RL (rich phase L) latest (v15_rich_phase_l/model_latest.pt).
                  V15GraphTransformer d=256, n_layers=8, history=8.
  4. V15.1      — V15.1 RL local-trained latest (v151_rl/model_latest.pt).
                  V15GraphTransformer d=128, n_layers=4, history=2 (~588K params).

Each pair plays N games as P0 and N as P2 (mirrored sides). Each V15-family
model carries its own (d_model, n_heads, n_layers, ffn_dim, history_len) so
the small V15.1 arch loads without shape errors.

Usage:
    python3 h2h_v151_tournament.py --games-per-orientation 100
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
V15_ROOT = HERE.parent / "td_ludo_v15"
sys.path.insert(0, str(V15_ROOT))

import td_ludo_cpp as cpp
from td_ludo.game.encoder_v18_production import encode_state_v18_production  # noqa: F401
from td_ludo.models.v13_5_production import V135ProductionAdapter

from td_ludo_v15.game.cells import (
    NUM_BOARD_CELLS, cell_to_index, position_to_cell_in_pov,
)
from td_ludo_v15.game.encoder import encode_frame
from td_ludo_v15.models.v15 import V15GraphTransformer
import td_ludo_v15_cpp as v15_cpp


_BASE_POS = v15_cpp.BASE_POS
MAX_MOVES_PER_GAME = 400
# Per-player history deque size = max of any V15-family contestant's history_len.
# Allocated once below; each V15 picker reads only its tail-T slice.
GLOBAL_HISTORY_MAX = 8


# ─── Loaders ──────────────────────────────────────────────────────────────
def _strip_prefixes(sd):
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    return sd


def load_v135_prod(path, device):
    """V135ProductionAdapter — V13.5 family RL ckpts."""
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    sd = _strip_prefixes(sd)
    model = V135ProductionAdapter(num_res_blocks=10, num_channels=128)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_v15_any(path, device, *, d_model, n_heads, n_layers, ffn_dim, history_len):
    """V15GraphTransformer with explicit arch — works for V15 (256/8/8/512/8)
    AND V15.1 (128/4/4/256/2). The student's input_mlp in_features is
    derived from history_len*3 inside the model constructor."""
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    sd = _strip_prefixes(sd)
    model = V15GraphTransformer(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        ffn_dim=ffn_dim, history_len=history_len,
    )
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ─── Pickers ──────────────────────────────────────────────────────────────
def pick_v135(model, device, state, legal, history=None):
    if len(legal) == 1:
        return legal[0]
    enc = encode_state_v18_production(state).astype(np.float32)
    token_legal = np.zeros(4, dtype=np.float32)
    for a in legal:
        token_legal[a] = 1.0
    with torch.no_grad():
        x = torch.from_numpy(enc).unsqueeze(0).to(device)
        lmt = torch.from_numpy(token_legal).unsqueeze(0).to(device)
        out = model(x, lmt)
        policy = out[0] if isinstance(out, tuple) else out
        action = int(policy.argmax(dim=1).item())
    return action if action in legal else legal[0]


def make_pick_v15(model, device, *, history_len):
    """Returns a closure that picks with a model-specific history depth.
    The closure expects `history` to be a deque of past states (newest last).
    Only the tail (history_len - 1) past frames are used, plus current state."""
    total_frames = history_len
    past_needed = history_len - 1  # number of pre-current frames to stack

    def pick(state, legal, history):
        if len(legal) == 1:
            return legal[0]
        cp = int(state.current_player)
        past = list(history) if history else []
        # Trim to the last `past_needed` past frames; pad with None at the front.
        if past_needed == 0:
            real_past = []
        else:
            past = past[-past_needed:]
            real_past = [None] * (past_needed - len(past)) + past
        real_frames = real_past + [state]
        assert len(real_frames) == total_frames, \
            f"history mismatch: got {len(real_frames)} frames, expected {total_frames}"
        v15_x = np.zeros((total_frames, 15, 15, 3), dtype=np.float32)
        for t_idx, st in enumerate(real_frames):
            if st is None:
                continue
            v15_x[t_idx] = encode_frame(st, pov_player=cp)
        v15_legal = np.zeros(NUM_BOARD_CELLS, dtype=np.float32)
        legal_cells = []
        for t in legal:
            pos = int(state.player_positions[cp][t])
            c = position_to_cell_in_pov(_BASE_POS if pos == _BASE_POS else pos, cp, cp)
            v15_legal[cell_to_index(*c)] = 1.0
            legal_cells.append((t, c))
        with torch.no_grad():
            xt = torch.from_numpy(v15_x).unsqueeze(0).to(device)
            mt = torch.from_numpy(v15_legal).unsqueeze(0).to(device)
            policy, _ = model(xt, mt)
            chosen_idx = int(policy.argmax(dim=-1).item())
        chosen_cell = divmod(chosen_idx, 15)
        for t, c in legal_cells:
            if c == chosen_cell:
                return t
        return legal[0]
    return pick


# ─── Game runner ──────────────────────────────────────────────────────────
def play_one(picks, seed):
    """picks: {player_id: callable(state, legal, history) → token_id}
    Returns: winner (0..3) or -1 (truncation)."""
    random.seed(seed)
    np.random.seed(seed)
    state = cpp.create_initial_state_2p()
    csix = [0, 0, 0, 0]
    mc = 0
    history = {p: collections.deque(maxlen=GLOBAL_HISTORY_MAX) for p in range(4)}

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

        pick_fn = picks.get(cp)
        if pick_fn is None:
            action = legal[0]
        else:
            action = pick_fn(state, list(legal), history.get(cp))
        for p in range(4):
            history[p].append(state)
        state = cpp.apply_move(state, int(action))
        mc += 1

    if state.is_terminal:
        return int(cpp.get_winner(state))
    return -1


def run_pair(name_a, picker_a, name_b, picker_b,
             games_per_orientation, seed_base=42, verbose=True):
    """Mirrored-sides pair: A plays as P0 half the time, P2 the other half."""
    a_wins = b_wins = draws = 0
    for g in range(games_per_orientation * 2):
        if g % 2 == 0:
            picks = {0: picker_a, 2: picker_b}
            a_player, b_player = 0, 2
        else:
            picks = {0: picker_b, 2: picker_a}
            a_player, b_player = 2, 0
        seed = seed_base + (g // 2)
        winner = play_one(picks, seed)
        if winner == a_player:
            a_wins += 1
        elif winner == b_player:
            b_wins += 1
        else:
            draws += 1
        if verbose and (g + 1) % 50 == 0:
            total = a_wins + b_wins + draws
            print(f"    [{g + 1:>4}/{games_per_orientation * 2}] "
                  f"{name_a} {100 * a_wins / total:.1f}% "
                  f"({a_wins}-{b_wins}, draws {draws})", flush=True)
    return a_wins, b_wins, draws


# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games-per-orientation", type=int, default=100,
                    help="Games each side plays as P0 (total per pair = 2N).")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--out", default="h2h_v151_results.json",
                    help="Where to dump JSON results")
    args = ap.parse_args()

    if args.device == "auto":
        device = (torch.device("cuda") if torch.cuda.is_available()
                  else torch.device("mps") if torch.backends.mps.is_available()
                  else torch.device("cpu"))
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    repo_root = HERE
    v15_root = HERE.parent / "td_ludo_v15"

    print("\nLoading models...")
    contestants = []

    # 1. V13.5 (pre-shaping-experiment latest)
    p = repo_root / "checkpoints" / "v135_prod_rl_local" / "model_latest.pt"
    print(f"  V13.5:     {p}")
    contestants.append(("V13.5", pick_v135, load_v135_prod(p, device)))
    # 2. V13.5_exp (post-shaping latest)
    p = repo_root / "checkpoints" / "v135_shaping_exp" / "model_latest.pt"
    print(f"  V13.5_exp: {p}")
    contestants.append(("V13.5_exp", pick_v135, load_v135_prod(p, device)))
    # 3. V15 RL (4.4M, history=8)
    p = v15_root / "checkpoints" / "v15_rich_phase_l" / "model_latest.pt"
    m = load_v15_any(p, device, d_model=256, n_heads=8, n_layers=8,
                     ffn_dim=512, history_len=8)
    print(f"  V15:       {p}")
    contestants.append(("V15", make_pick_v15(m, device, history_len=8), m))
    # 4. V15.1 RL (588K, history=2)
    p = v15_root / "checkpoints" / "v151_rl" / "model_latest.pt"
    m = load_v15_any(p, device, d_model=128, n_heads=4, n_layers=4,
                     ffn_dim=256, history_len=2)
    print(f"  V15.1:     {p}")
    contestants.append(("V15.1", make_pick_v15(m, device, history_len=2), m))

    # Convert to (name, picker_callable, _) — picker is bound to the model.
    # For V13.5 we wrap pick_v135 to bind the model.
    bound = []
    for name, picker, model in contestants:
        if picker is pick_v135:
            bound.append((name, (lambda s, l, h, _m=model: pick_v135(_m, device, s, l, h))))
        else:
            bound.append((name, picker))  # V15 pickers are already closures over model

    names = [b[0] for b in bound]
    n = len(bound)
    print(f"\nRound-robin: {n} models, "
          f"{args.games_per_orientation * 2} games per pair "
          f"({args.games_per_orientation} per orientation).")
    print(f"Total games: {n * (n - 1) // 2 * args.games_per_orientation * 2}")
    print()

    results = {a: {b: None for b in names} for a in names}
    t_start = time.time()

    for i in range(n):
        for j in range(i + 1, n):
            name_a, picker_a = bound[i]
            name_b, picker_b = bound[j]
            print(f"━━ {name_a} vs {name_b} ━━")
            t = time.time()
            aw, bw, dr = run_pair(name_a, picker_a, name_b, picker_b,
                                  args.games_per_orientation, args.seed_base)
            results[name_a][name_b] = (aw, bw, dr)
            results[name_b][name_a] = (bw, aw, dr)
            total = aw + bw + dr
            print(f"  FINAL: {name_a} {100 * aw / total:.1f}% "
                  f"vs {name_b} {100 * bw / total:.1f}% (draws {dr}) "
                  f"in {time.time() - t:.0f}s\n")

    elapsed_total = time.time() - t_start
    print("=" * 70)
    print(f"WR matrix (row vs column, % wins for row)  "
          f"[{args.games_per_orientation * 2}g/pair, {elapsed_total / 60:.1f} min]")
    print("=" * 70)
    print("            " + "".join(f"{n:>12}" for n in names))
    for name_a in names:
        row = f"{name_a:>11} "
        for name_b in names:
            if name_a == name_b:
                row += f"{'  --':>12}"
            elif results[name_a][name_b] is None:
                row += f"{'?':>12}"
            else:
                aw, bw, dr = results[name_a][name_b]
                row += f"{100 * aw / (aw + bw + dr):>11.1f}%"
        print(row)

    print("\nAggregate WR (vs all opponents):")
    aggs = []
    for name_a in names:
        tw = tg = 0
        for name_b in names:
            if name_a == name_b or results[name_a][name_b] is None:
                continue
            aw, bw, dr = results[name_a][name_b]
            tw += aw
            tg += aw + bw + dr
        aggs.append((name_a, 100 * tw / tg if tg else 0, tw, tg))
    aggs.sort(key=lambda x: -x[1])
    for rank, (nm, wr, w, g) in enumerate(aggs, 1):
        print(f"  {rank}. {nm:<12} {wr:>5.1f}%  ({w}/{g})")

    with open(args.out, "w") as f:
        json.dump({
            "names": names,
            "results": {a: {b: results[a][b] for b in names} for a in names},
            "aggregates": [{"name": n, "wr": wr, "wins": w, "games": g}
                           for n, wr, w, g in aggs],
            "games_per_orientation": args.games_per_orientation,
            "elapsed_min": elapsed_total / 60,
            "device": str(device),
        }, f, indent=2, default=str)
    print(f"\nJSON → {args.out}")


if __name__ == "__main__":
    main()
