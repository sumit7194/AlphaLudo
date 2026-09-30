"""H2H: SL models (V15.2, V12.3) vs V13.5 RL best.

V13.5 RL best (`play/model_weights/v13_5/model_best.pt`) is our prior
strongest model — symmetric CNN with rank-indexed 4-token policy,
trained with RL on the V13.x trainer. This is the "RL ceiling" we
want to know if pure-SL on new dataset can approach.

Two matchups, sequential, 500 games per orientation each (1000g per
matchup, ±2.2pp std error).

  1. V15.2 SL  vs  V13.5 RL best
  2. V12.3 SL  vs  V13.5 RL best

Usage:
    cd /Users/sumit/Github/AlphaLudo/td_ludo
    PYTHONPATH=.:../td_ludo_v15 \\
      ./td_env/bin/python -u h2h_sl_vs_v135best.py
"""
from __future__ import annotations

import collections
import json
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
from eval_v135_v15_v151_local import (
    load_v135, make_picker_v135,
    load_v15_any, make_picker_v15,
)

MAX_MOVES = 400


# ── V12.3 loader + picker (same as h2h_v152sl_vs_v123sl.py) ─────────────

def load_v123(path, device):
    from experiments.distillation_14ch.model_14ch import MinimalCNN14
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    args = ck.get("args", {}) if isinstance(ck, dict) else {}
    n_res = int(args.get("num_res_blocks", 8))
    n_ch  = int(args.get("num_channels", 96))
    n_in  = int(args.get("in_channels", 17))
    m = MinimalCNN14(num_res_blocks=n_res, num_channels=n_ch, in_channels=n_in)
    m.load_state_dict(sd, strict=False)
    m.to(device).eval()
    return m


def make_picker_v123(model, device):
    from td_ludo.game.encoder_v17 import encode_state_v17
    @torch.no_grad()
    def pick(state, legal, history):
        x = torch.from_numpy(encode_state_v17(state).astype(np.float32))
        x = x.unsqueeze(0).to(device)
        lm = torch.zeros(4, dtype=torch.float32, device=device)
        for a in legal:
            lm[a] = 1.0
        lm = lm.unsqueeze(0)
        out = model(x, lm)
        policy = out[0] if isinstance(out, tuple) else out
        return int(policy.argmax(dim=-1).item())
    return pick


# ── Game loop ────────────────────────────────────────────────────────────

def run_game(picker_a, picker_b, hist_a, hist_b, seed):
    random.seed(seed)
    state = cpp.create_initial_state_2p()
    ha = collections.deque(maxlen=hist_a) if hist_a > 0 else None
    hb = collections.deque(maxlen=hist_b) if hist_b > 0 else None
    csix = [0, 0, 0, 0]
    mc = 0
    while not state.is_terminal and mc < MAX_MOVES:
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
        if cp == 0:
            action = picker_a(state, list(legal), ha)
        else:
            action = picker_b(state, list(legal), hb)
        if ha is not None: ha.append(state)
        if hb is not None: hb.append(state)
        state = cpp.apply_move(state, int(action))
        mc += 1
    if state.is_terminal:
        w = cpp.get_winner(state)
        if w == 0: return "a", mc
        if w == 2: return "b", mc
    return "draw", mc


def run_h2h(name_a, pick_a, hist_a, name_b, pick_b, hist_b, *,
            games_per_orientation, label):
    aw = bw = dr = 0
    total_len = 0
    t0 = time.time()
    for i in range(games_per_orientation):
        r, mc = run_game(pick_a, pick_b, hist_a, hist_b, seed=i)
        total_len += mc
        if   r == "a": aw += 1
        elif r == "b": bw += 1
        else:          dr += 1
        if (i + 1) % 50 == 0:
            print(f"  [{label}] orient1 {i+1}/{games_per_orientation}  "
                  f"| {name_a}={aw} {name_b}={bw} draw={dr}  | "
                  f"{(i+1)/(time.time()-t0):.1f} g/s", flush=True)
    for i in range(games_per_orientation):
        r, mc = run_game(pick_b, pick_a, hist_b, hist_a, seed=10_000 + i)
        total_len += mc
        if   r == "a": bw += 1
        elif r == "b": aw += 1
        else:          dr += 1
        if (i + 1) % 50 == 0:
            print(f"  [{label}] orient2 {i+1}/{games_per_orientation}  "
                  f"| {name_a}={aw} {name_b}={bw} draw={dr}  | "
                  f"{(i+1+games_per_orientation)/(time.time()-t0):.1f} g/s",
                  flush=True)
    total = aw + bw + dr
    return {
        "a_wins": aw, "b_wins": bw, "draws": dr,
        "a_wr": 100.0 * aw / total,
        "b_wr": 100.0 * bw / total,
        "elapsed_sec": time.time() - t0,
        "avg_len": total_len / total,
        "games": total,
    }


def print_verdict(name_a, name_b, r):
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  {name_a:<25}  {r['a_wins']:>4} wins ({r['a_wr']:.1f}%)")
    print(f"  {name_b:<25}  {r['b_wins']:>4} wins ({r['b_wr']:.1f}%)")
    print(f"  draws: {r['draws']:>4}  ·  avg len: {r['avg_len']:.0f} moves")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    p = r['a_wr'] / 100.0; n = r['games']
    se = 100.0 * (p * (1 - p) / n) ** 0.5
    delta = r['a_wr'] - r['b_wr']
    print(f"  std error: ±{se:.2f}pp")
    if delta > 2 * se:
        v = f"{name_a} > {name_b}  (+{delta:.1f}pp, real)"
    elif delta < -2 * se:
        v = f"{name_a} < {name_b}  ({delta:.1f}pp, real)"
    else:
        v = f"~tie  ({delta:+.1f}pp, within ±{2*se:.1f}pp noise)"
    print(f"  verdict:   {v}\n")
    return se, delta


def main():
    device = torch.device("cpu")
    games_per_orientation = 500

    ckpt_dir = V15_ROOT / "checkpoints" / "h2h_compare"
    path_v152 = ckpt_dir / "v152_sl_final.pt"
    path_v123 = ckpt_dir / "v123_sl_final.pt"
    path_v135 = HERE / "play" / "model_weights" / "v13_5" / "model_best.pt"

    print(f"Device: {device}")
    print(f"games per orientation: {games_per_orientation} (total per matchup: {2*games_per_orientation})\n")

    print(f"Loading models...")
    m_v135 = load_v135(path_v135, device)
    pick_v135 = make_picker_v135(m_v135, device)
    print(f"  V13.5 RL best        : {sum(p.numel() for p in m_v135.parameters()):,} params  ← {path_v135.name}")

    m_v152 = load_v15_any(path_v152, device,
                          d_model=128, n_heads=4, n_layers=4, ffn_dim=256,
                          history_len=1)
    pick_v152 = make_picker_v15(m_v152, device, history_len=1)
    print(f"  V15.2 SL (h=1)       : {sum(p.numel() for p in m_v152.parameters()):,} params")

    m_v123 = load_v123(path_v123, device)
    pick_v123 = make_picker_v123(m_v123, device)
    print(f"  V12.3 SL (V17 CNN)   : {sum(p.numel() for p in m_v123.parameters()):,} params\n")

    results = {}

    # Matchup 1: V15.2 SL vs V13.5 RL
    print(f"━━━ Matchup 1: V15.2_SL vs V13.5_best  ({2*games_per_orientation} games) ━━━\n")
    t0 = time.time()
    r1 = run_h2h("V15.2_SL", pick_v152, 1, "V13.5_best", pick_v135, 8,
                 games_per_orientation=games_per_orientation,
                 label="V152SL_vs_V135")
    el1 = time.time() - t0
    print(f"\nMatchup 1 done in {el1/60:.1f} min ({r1['games']/el1:.1f} g/s)\n")
    se1, d1 = print_verdict("V15.2_SL", "V13.5_best", r1)
    results["v152sl_vs_v135"] = {"r": r1, "se": se1, "delta_pp": d1, "elapsed_sec": el1}

    # Matchup 2: V12.3 SL vs V13.5 RL
    print(f"━━━ Matchup 2: V12.3_SL vs V13.5_best  ({2*games_per_orientation} games) ━━━\n")
    t0 = time.time()
    r2 = run_h2h("V12.3_SL", pick_v123, 0, "V13.5_best", pick_v135, 8,
                 games_per_orientation=games_per_orientation,
                 label="V123SL_vs_V135")
    el2 = time.time() - t0
    print(f"\nMatchup 2 done in {el2/60:.1f} min ({r2['games']/el2:.1f} g/s)\n")
    se2, d2 = print_verdict("V12.3_SL", "V13.5_best", r2)
    results["v123sl_vs_v135"] = {"r": r2, "se": se2, "delta_pp": d2, "elapsed_sec": el2}

    # Save combined
    out = HERE / "h2h_sl_vs_v135best_results.json"
    with open(out, "w") as f:
        json.dump({
            "ckpt_v135_best": str(path_v135),
            "ckpt_v152_sl":   str(path_v152),
            "ckpt_v123_sl":   str(path_v123),
            "games_per_orientation": games_per_orientation,
            "matchups": results,
        }, f, indent=2)
    print(f"All results saved → {out}")

    print("━━━ Summary ━━━")
    print(f"  V15.2_SL vs V13.5_best:  {r1['a_wr']:.1f}% / {r1['b_wr']:.1f}%  (Δ {d1:+.1f}pp)")
    print(f"  V12.3_SL vs V13.5_best:  {r2['a_wr']:.1f}% / {r2['b_wr']:.1f}%  (Δ {d2:+.1f}pp)")


if __name__ == "__main__":
    main()
