"""H2H: V15.1 RL post-strong-opp (G=826K, current VM) vs V15.1 RL pre-strong-opp.

Sanity check on whether the strong-opp + dense-shaping training has
ACTUALLY made V15.1 stronger H2H, even though the bot-WR eval is
oscillating in the 71-77% range.

  A: v151_rl_strong/model_prev.pt  (G≈826K, post strong-opp + dense shaping)
  B: v151_rl/model_latest.pt       (pre strong-opp, baseline)

Both checkpoints were copied locally via gcloud scp into
td_ludo_v15/checkpoints/h2h_compare/.

Same arch (V15GT d=128/L4/H4/ffn=256/history=2) so loader is identical.

Usage:
    cd /Users/sumit/Github/AlphaLudo/td_ludo
    PYTHONPATH=.:../td_ludo_v15 \\
      ./td_env/bin/python -u h2h_v151_strong_vs_pre.py
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


def run_game(picker_a, picker_b, seed, history_maxlen=8):
    """picker_a plays p0, picker_b plays p2. Returns (winner, mc)."""
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


def run_h2h(name_a, pick_a, name_b, pick_b, *, games_per_orientation, label,
            history_maxlen=8):
    aw = bw = dr = 0
    total_len = 0
    t0 = time.time()
    for i in range(games_per_orientation):
        r, mc = run_game(pick_a, pick_b, seed=i, history_maxlen=history_maxlen)
        total_len += mc
        if   r == "a": aw += 1
        elif r == "b": bw += 1
        else:          dr += 1
        if (i + 1) % 50 == 0:
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
    games_per_orientation = 500   # 1000 games total → ±2.2pp std error

    ckpt_dir = V15_ROOT / "checkpoints" / "h2h_compare"
    path_strong = ckpt_dir / "v151_rl_strong_G826K.pt"
    path_pre    = ckpt_dir / "v151_rl_pre_strong.pt"

    print(f"Device: {device}")
    print(f"games per orientation: {games_per_orientation}  "
          f"(total {2*games_per_orientation})")
    print(f"Loading models...")

    arch = dict(d_model=128, n_heads=4, n_layers=4, ffn_dim=256, history_len=2)
    m_strong = load_v15_any(path_strong, device, **arch)
    pick_strong = make_picker_v15(m_strong, device, history_len=2)
    print(f"  STRONG (V15.1-RL post-strong-opp, G≈826K): "
          f"{sum(p.numel() for p in m_strong.parameters()):,} params  "
          f"← {path_strong.name}")

    m_pre = load_v15_any(path_pre, device, **arch)
    pick_pre = make_picker_v15(m_pre, device, history_len=2)
    print(f"  PRE    (V15.1-RL pre-strong-opp, baseline):  "
          f"{sum(p.numel() for p in m_pre.parameters()):,} params  "
          f"← {path_pre.name}\n")

    configure_history(2)

    print(f"Running H2H: STRONG vs PRE  "
          f"({2*games_per_orientation} games, "
          f"{games_per_orientation} per orientation)\n")
    t0 = time.time()
    r = run_h2h("STRONG", pick_strong, "PRE", pick_pre,
                games_per_orientation=games_per_orientation,
                label="STRONG_vs_PRE", history_maxlen=8)
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed/60:.1f} min  ({r['games']/elapsed:.1f} g/s)\n")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  STRONG (G=826K, dense shaping):   {r['a_wins']:>4} wins  ({r['a_wr']:.1f}%)")
    print(f"  PRE    (pre-strong-opp baseline): {r['b_wins']:>4} wins  ({r['b_wr']:.1f}%)")
    print(f"  draws:  {r['draws']:>4}")
    print(f"  avg game length: {r['avg_len']:.0f} moves")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    p = r['a_wr'] / 100.0
    n = r['games']
    se = 100.0 * (p * (1 - p) / n) ** 0.5
    print(f"\n  std error: ±{se:.2f}pp")

    delta = r['a_wr'] - r['b_wr']
    if delta > 2 * se:
        verdict = (f"STRONG > PRE  (+{delta:.1f}pp, "
                   f">{2*se:.1f}pp 2σ → real H2H gain)")
    elif delta < -2 * se:
        verdict = (f"STRONG < PRE  ({delta:.1f}pp, "
                   f"<-{2*se:.1f}pp 2σ → REAL regression)")
    else:
        verdict = (f"~tie  ({delta:+.1f}pp, within ±{2*se:.1f}pp noise)")
    print(f"  verdict:    {verdict}\n")

    out = HERE / "h2h_v151_strong_vs_pre_results.json"
    with open(out, "w") as f:
        json.dump({
            "result": r,
            "ckpt_strong": str(path_strong),
            "ckpt_pre":    str(path_pre),
            "games_per_orientation": games_per_orientation,
            "elapsed_sec": elapsed,
            "std_err_pp": se,
            "delta_pp": delta,
        }, f, indent=2)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
