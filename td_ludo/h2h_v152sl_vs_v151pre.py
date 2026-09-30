"""H2H: V15.2 SL (final, 1 epoch, val_acc 93.5%) vs V15.1 RL pre-strong-opp.

Baseline check: how does the freshly-trained imitation-of-bots SL model
compare against the V15.1 RL model from before the strong-opp experiment?

Expectations:
  - V15.1 PRE had ~75K games of RL self-play + opp-mix tuning
  - V15.2 SL only imitates bot moves (Expectimax-family weighted heavily)
    with 1 epoch over 37M states
  - PRE should win, but the margin tells us how much value RL added

Architectures (both V15 GraphTransformer, 587K params each):
  V15.2 SL:  history_len=1, d_model=128, L4, H4, ffn=256, 225-cell output
  V15.1 RL:  history_len=2, ...same...                    225-cell output

Both use the same picker (make_picker_v15) — only history_len differs.

Usage:
    cd /Users/sumit/Github/AlphaLudo/td_ludo
    PYTHONPATH=.:../td_ludo_v15 \\
      ./td_env/bin/python -u h2h_v152sl_vs_v151pre.py
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
from eval_v135_v15_v151_local import load_v15_any, make_picker_v15
from td_ludo_v15.rich.v15_bot_eval import configure_history

MAX_MOVES = 400


def run_game(picker_a, picker_b, hist_a, hist_b, seed):
    """picker_a plays p0, picker_b plays p2.

    NOTE: Each picker may want a different history length (V15.2 uses 1,
    V15.1 uses 2). We maintain a separate deque per picker and pass the
    right one to each call.
    """
    random.seed(seed)
    state = cpp.create_initial_state_2p()
    ha = collections.deque(maxlen=hist_a)
    hb = collections.deque(maxlen=hist_b)
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
        ha.append(state)
        hb.append(state)
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
    path_v152 = ckpt_dir / "v152_sl_final.pt"
    path_pre  = ckpt_dir / "v151_rl_pre_strong.pt"

    print(f"Device: {device}")
    print(f"games per orientation: {games_per_orientation}  "
          f"(total {2*games_per_orientation})")
    print(f"Loading models...")

    # V15.2 SL: history_len=1
    m_v152 = load_v15_any(path_v152, device,
                          d_model=128, n_heads=4, n_layers=4, ffn_dim=256,
                          history_len=1)
    pick_v152 = make_picker_v15(m_v152, device, history_len=1)
    print(f"  V15.2 SL   (1 epoch, val_acc 93.5%, history=1): "
          f"{sum(p.numel() for p in m_v152.parameters()):,} params")

    # V15.1 RL PRE: history_len=2
    m_pre = load_v15_any(path_pre, device,
                         d_model=128, n_heads=4, n_layers=4, ffn_dim=256,
                         history_len=2)
    pick_pre = make_picker_v15(m_pre, device, history_len=2)
    print(f"  V15.1 RL PRE (pre-strong-opp, history=2):       "
          f"{sum(p.numel() for p in m_pre.parameters()):,} params\n")

    # configure_history is module-level for the v15 player; set the larger.
    configure_history(2)

    print(f"Running H2H: V15.2_SL vs V15.1_PRE  "
          f"({2*games_per_orientation} games)\n")
    t0 = time.time()
    r = run_h2h("V152SL", pick_v152, 1, "V151PRE", pick_pre, 2,
                games_per_orientation=games_per_orientation,
                label="V152SL_vs_V151PRE")
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed/60:.1f} min  ({r['games']/elapsed:.1f} g/s)\n")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  V15.2 SL  (1-epoch imitation):  {r['a_wins']:>4} wins ({r['a_wr']:.1f}%)")
    print(f"  V15.1 PRE (pre-strong-opp RL):  {r['b_wins']:>4} wins ({r['b_wr']:.1f}%)")
    print(f"  draws: {r['draws']:>4}  ·  avg len: {r['avg_len']:.0f} moves")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    p = r['a_wr'] / 100.0; n = r['games']
    se = 100.0 * (p * (1 - p) / n) ** 0.5
    delta = r['a_wr'] - r['b_wr']
    print(f"  std error: ±{se:.2f}pp")
    if delta > 2 * se:
        v = f"V15.2 SL > V15.1 PRE  (+{delta:.1f}pp, real)"
    elif delta < -2 * se:
        v = f"V15.2 SL < V15.1 PRE  ({delta:.1f}pp, real)"
    else:
        v = f"~tie  ({delta:+.1f}pp, within ±{2*se:.1f}pp noise)"
    print(f"  verdict:   {v}\n")

    out = HERE / "h2h_v152sl_vs_v151pre_results.json"
    with open(out, "w") as f:
        json.dump({
            "result": r,
            "ckpt_v152_sl": str(path_v152),
            "ckpt_v151_pre": str(path_pre),
            "games_per_orientation": games_per_orientation,
            "elapsed_sec": elapsed,
            "std_err_pp": se,
            "delta_pp": delta,
        }, f, indent=2)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
