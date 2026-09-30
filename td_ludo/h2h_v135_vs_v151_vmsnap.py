"""H2H: V13.5 (pre-experiment RL latest, G=535K) vs V15.1 (VM snapshot, G=151K).

V13.5 has plateaued at ~75.5% bot-WR; V15.1 just reached parity at ~75.9%
after 150K games of strong-opp training. Bot-WR is now equal — so the
question is: does V15.1's H2H advantage from the 6-way tournament
(~55-56% h2h) still hold for the latest snapshot, or did the strong-opp
training shift things?

1000 games (500 per orientation) for ±1.5pp confidence.
Reuses loaders + pickers from eval_v135_v15_v151_local.py.

Usage:
    cd /Users/sumit/Github/AlphaLudo/td_ludo
    PYTHONPATH=.:../td_ludo_v15 \
      ./td_env/bin/python -u h2h_v135_vs_v151_vmsnap.py
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
    load_v135, load_v15_any, make_picker_v135, make_picker_v15,
)
from td_ludo_v15.rich.v15_bot_eval import configure_history

MAX_MOVES = 400


def run_game(picker_a, picker_b, seed, history_maxlen=8):
    """picker_a plays p0, picker_b plays p2. Returns ('a','b','draw', mc)."""
    random.seed(seed)
    state = cpp.create_initial_state_2p()
    history = collections.deque(maxlen=history_maxlen)
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
            action = picker_a(state, list(legal), history)
        else:
            action = picker_b(state, list(legal), history)
        history.append(state)
        state = cpp.apply_move(state, int(action))
        mc += 1
    if state.is_terminal:
        w = cpp.get_winner(state)
        if w == 0: return "a", mc
        if w == 2: return "b", mc
    return "draw", mc


def run_h2h(name_a, pick_a, name_b, pick_b, *, games_per_orientation,
            label, history_maxlen=8):
    aw = bw = dr = 0
    total_len = 0
    t0 = time.time()
    for i in range(games_per_orientation):
        r, mc = run_game(pick_a, pick_b, seed=i, history_maxlen=history_maxlen)
        total_len += mc
        if   r == "a": aw += 1
        elif r == "b": bw += 1
        else:          dr += 1
        if (i + 1) % 100 == 0:
            print(f"  [{label}] orient1 {i+1}/{games_per_orientation}  "
                  f"| {name_a}={aw} {name_b}={bw} draw={dr}  | "
                  f"{(i+1)/(time.time()-t0):.1f} g/s", flush=True)
    for i in range(games_per_orientation):
        r, mc = run_game(pick_b, pick_a, seed=10_000 + i,
                         history_maxlen=history_maxlen)
        total_len += mc
        if   r == "a": bw += 1
        elif r == "b": aw += 1
        else:          dr += 1
        if (i + 1) % 100 == 0:
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
    games_per_orientation = 1000

    print(f"Device: {device}")
    print(f"Loading models...")

    v135 = load_v135(
        HERE / "play" / "model_weights" / "v13_5" / "model_latest.pt", device)
    pick_v135 = make_picker_v135(v135, device)
    print(f"  V13.5 (pre-experiment RL, G=535K): "
          f"{sum(p.numel() for p in v135.parameters()):,} params")

    v151 = load_v15_any(
        V15_ROOT / "checkpoints" / "v151_rl_strong_vm_snapshot" / "model_prev.pt",
        device, d_model=128, n_heads=4, n_layers=4, ffn_dim=256, history_len=2)
    pick_v151 = make_picker_v15(v151, device, history_len=2)
    print(f"  V15.1 (VM model_prev.pt snapshot, G=503K, post style-diverse training): "
          f"{sum(p.numel() for p in v151.parameters()):,} params\n")

    # configure_history matters for the V15 module's internal state
    # (it's not used by make_picker_v15 directly, but other places import it).
    configure_history(2)

    print(f"Running H2H: V13.5 vs V15.1  "
          f"({2*games_per_orientation} games, {games_per_orientation} per orientation)\n")
    t0 = time.time()
    r = run_h2h("V13.5", pick_v135, "V15.1", pick_v151,
                games_per_orientation=games_per_orientation,
                label="V13.5_vs_V15.1", history_maxlen=8)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min  ({r['games']/elapsed:.1f} g/s)\n")

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  V13.5:  {r['a_wins']:>4} wins  ({r['a_wr']:.1f}%)")
    print(f"  V15.1:  {r['b_wins']:>4} wins  ({r['b_wr']:.1f}%)")
    print(f"  draws:  {r['draws']:>4}")
    print(f"  avg game length: {r['avg_len']:.0f} moves")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # std error
    p = r['a_wr'] / 100.0
    n = r['games']
    se = 100.0 * (p * (1 - p) / n) ** 0.5
    print(f"\n  std error (Wilson, sample): ±{se:.1f}pp")

    # comparison to 6-way tournament historical numbers
    print(f"\n  Yesterday's 6-way H2H result: V15.1 won ~55-56% vs V13.5")
    if r['b_wr'] > 50:
        print(f"  This run:                      V15.1 wins {r['b_wr']:.1f}% "
              f"(+{r['b_wr']-50:.1f}pp edge)")
    else:
        print(f"  This run:                      V13.5 wins {r['a_wr']:.1f}% "
              f"(V15.1 lost {50-r['b_wr']:.1f}pp ground)")

    out = HERE / "h2h_v135_vs_v151_vmsnap_results.json"
    with open(out, "w") as f:
        json.dump({
            "result": r,
            "ckpt_v135": "play/model_weights/v13_5/model_latest.pt (G=535450)",
            "ckpt_v151": "checkpoints/v151_rl_strong_vm_snapshot/model_latest.pt (G=151257)",
            "games_per_orientation": games_per_orientation,
            "elapsed_sec": elapsed,
        }, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
