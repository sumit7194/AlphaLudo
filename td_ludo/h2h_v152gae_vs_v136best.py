"""H2H: V15.2 gaeterminal BEST  vs  V13.6 gaeterminal BEST (champion).

Fair cross-architecture comparison — eval numbers aren't comparable across the
two training pools, so play them head-to-head in the same (v13) engine via the
shared adapters.

  - V15.2 best  = single-frame GraphTransformer (587K params), terminal-only+GAE,
                  init from the old v15.2 RL best, best eval ~0.845.
                  td_ludo_v15/checkpoints/h2h_compare/v152_gaeterminal_best.pt
  - V13.6 best  = gaeterminal 85.5% champion (V135 6x96 head_hidden=64, ~1.05M).
                  checkpoint_backups/v136_gaeterminal_2026-06-17/...BEST...pt

2000 games (1000 per orientation), CPU, ~±1.1pp std error.

Usage:
    cd /Users/sumit/Github/AlphaLudo/td_ludo
    PYTHONPATH=.:../td_ludo_v15 ./td_env/bin/python -u h2h_v152gae_vs_v136best.py
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
from td_ludo.models.v13_5_production import V135ProductionAdapter
from eval_v135_v15_v151_local import (
    _strip, load_v15_any, make_picker_v15, make_picker_v135,
)

MAX_MOVES = 400


def load_v136(path, device):
    """V13.6 = V135ProductionAdapter 6x96 head_hidden=64 (matches play server)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    sd = _strip(sd)
    m = V135ProductionAdapter(num_res_blocks=6, num_channels=96, head_hidden=64)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    # Sanity: only aux heads (progress/capture/risk) may be missing — never core.
    core_missing = [k for k in missing if not any(
        t in k for t in ("progress_fc", "capture_fc", "risk_fc"))]
    if core_missing:
        raise RuntimeError(f"V13.6 core weights missing: {core_missing[:6]}")
    m.eval().to(device)
    return m


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


def run_h2h(pick_a, hist_a, pick_b, hist_b, name_a, name_b, gpo):
    aw = bw = dr = 0
    t0 = time.time()
    for i in range(gpo):
        r, _ = run_game(pick_a, pick_b, hist_a, hist_b, seed=i)
        if r == "a": aw += 1
        elif r == "b": bw += 1
        else: dr += 1
        if (i + 1) % 100 == 0:
            print(f"  orient1 {i+1}/{gpo} | {name_a}={aw} {name_b}={bw} draw={dr} "
                  f"| {(i+1)/(time.time()-t0):.1f} g/s", flush=True)
    for i in range(gpo):
        r, _ = run_game(pick_b, pick_a, hist_b, hist_a, seed=10_000 + i)
        if r == "a": bw += 1
        elif r == "b": aw += 1
        else: dr += 1
        if (i + 1) % 100 == 0:
            print(f"  orient2 {i+1}/{gpo} | {name_a}={aw} {name_b}={bw} draw={dr} "
                  f"| {(i+1+gpo)/(time.time()-t0):.1f} g/s", flush=True)
    total = aw + bw + dr
    return {"a_wins": aw, "b_wins": bw, "draws": dr, "games": total,
            "a_wr": 100.0 * aw / total, "b_wr": 100.0 * bw / total,
            "elapsed_sec": time.time() - t0}


def main():
    device = torch.device("cpu")
    gpo = 1000  # 2000 total

    path_v152 = V15_ROOT / "checkpoints" / "h2h_compare" / "v152_gaeterminal_best.pt"
    path_v136 = HERE.parent / "checkpoint_backups" / "v136_gaeterminal_2026-06-17" / \
        "v136_gaeterminal_BEST_eval85.5pct_G200k_2026-06-16.pt"

    print(f"Loading models (device={device})...")
    m_v152 = load_v15_any(path_v152, device, d_model=128, n_heads=4, n_layers=4,
                          ffn_dim=256, history_len=1)
    pick_v152 = make_picker_v15(m_v152, device, history_len=1)
    print(f"  V15.2 best (single-frame GT): {sum(p.numel() for p in m_v152.parameters()):,} params")

    m_v136 = load_v136(path_v136, device)
    pick_v136 = make_picker_v135(m_v136, device)
    print(f"  V13.6 best (champion 85.5%):  {sum(p.numel() for p in m_v136.parameters()):,} params\n")

    print(f"Running H2H: V15.2 (a) vs V13.6 (b) — {2*gpo} games\n")
    res = run_h2h(pick_v152, 0, pick_v136, 0, "V15.2", "V13.6", gpo)

    se = (res["a_wr"] * res["b_wr"] / res["games"]) ** 0.5
    print("\n" + "=" * 56)
    print(f"  V15.2 : {res['a_wins']}  ({res['a_wr']:.1f}%)")
    print(f"  V13.6 : {res['b_wins']}  ({res['b_wr']:.1f}%)")
    print(f"  draws : {res['draws']}")
    print(f"  games : {res['games']}  | std err ~±{se:.1f}pp")
    print(f"  time  : {res['elapsed_sec']:.0f}s")
    verdict = ("V15.2 BEATS V13.6" if res["a_wr"] > res["b_wr"] + 2 * se else
               "V13.6 BEATS V15.2" if res["b_wr"] > res["a_wr"] + 2 * se else
               "STATISTICAL TIE")
    print(f"  VERDICT: {verdict}")
    print("=" * 56)

    out = HERE / "h2h_v152gae_vs_v136best_results.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
