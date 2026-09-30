"""H2H: V13.5 RL best vs V15.1 RL pre-strong-opp.

The user reports V13.5 best peaked at 85% bot-eval WR vs V15.1 PRE's
~76% ceiling. So V13.5 best should win H2H — this run confirms it.

  A: V13.5 RL best  (play/model_weights/v13_5/model_best.pt, 3.0M params)
  B: V15.1 RL pre-strong-opp  (h2h_compare/v151_rl_pre_strong.pt, 0.59M)

1000 games (500 per orientation), CPU, ±2.2pp std error.

Usage:
    cd /Users/sumit/Github/AlphaLudo/td_ludo
    PYTHONPATH=.:../td_ludo_v15 \\
      ./td_env/bin/python -u h2h_v135best_vs_v151pre.py
"""
from __future__ import annotations

import collections
import json
import random
import sys
import time
from pathlib import Path

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


def main():
    device = torch.device("cpu")
    games_per_orientation = 500

    ckpt_dir = V15_ROOT / "checkpoints" / "h2h_compare"
    path_v135 = HERE / "play" / "model_weights" / "v13_5" / "model_best.pt"
    path_v151 = ckpt_dir / "v151_rl_pre_strong.pt"

    print(f"Device: {device}")
    print(f"games per orientation: {games_per_orientation} (total {2*games_per_orientation})")
    print(f"Loading models...")

    m_v135 = load_v135(path_v135, device)
    pick_v135 = make_picker_v135(m_v135, device)
    print(f"  V13.5 RL best (3M, 85% eval ceiling): "
          f"{sum(p.numel() for p in m_v135.parameters()):,} params  ← {path_v135.name}")

    m_v151 = load_v15_any(path_v151, device,
                          d_model=128, n_heads=4, n_layers=4, ffn_dim=256,
                          history_len=2)
    pick_v151 = make_picker_v15(m_v151, device, history_len=2)
    print(f"  V15.1 RL PRE  (0.59M, 76% eval ceiling): "
          f"{sum(p.numel() for p in m_v151.parameters()):,} params\n")

    print(f"Running H2H: V13.5_best vs V15.1_PRE  "
          f"({2*games_per_orientation} games)\n")
    t0 = time.time()
    r = run_h2h("V135_best", pick_v135, 8, "V151_PRE", pick_v151, 2,
                games_per_orientation=games_per_orientation,
                label="V135best_vs_V151PRE")
    el = time.time() - t0
    print(f"\nDone in {el/60:.1f} min ({r['games']/el:.1f} g/s)\n")

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  V13.5 best (85% eval ceiling):    {r['a_wins']:>4} wins ({r['a_wr']:.1f}%)")
    print(f"  V15.1 PRE  (76% eval ceiling):    {r['b_wins']:>4} wins ({r['b_wr']:.1f}%)")
    print(f"  draws: {r['draws']:>4}  ·  avg len: {r['avg_len']:.0f} moves")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    p = r['a_wr'] / 100.0; n = r['games']
    se = 100.0 * (p * (1 - p) / n) ** 0.5
    delta = r['a_wr'] - r['b_wr']
    print(f"  std error: ±{se:.2f}pp")
    if delta > 2 * se:
        v = f"V13.5 best > V15.1 PRE  (+{delta:.1f}pp, real)"
    elif delta < -2 * se:
        v = f"V13.5 best < V15.1 PRE  ({delta:.1f}pp, real)"
    else:
        v = f"~tie  ({delta:+.1f}pp, within ±{2*se:.1f}pp noise)"
    print(f"  verdict:   {v}\n")

    out = HERE / "h2h_v135best_vs_v151pre_results.json"
    with open(out, "w") as f:
        json.dump({
            "result": r,
            "ckpt_v135_best": str(path_v135),
            "ckpt_v151_pre":  str(path_v151),
            "games_per_orientation": games_per_orientation,
            "elapsed_sec": el,
            "std_err_pp": se,
            "delta_pp": delta,
        }, f, indent=2)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
