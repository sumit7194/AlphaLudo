"""3-way H2H — after V13.5 strong-opp experiment is paused.

Tests:
  A. V13.5 strong-opp (G=31K, dense rewards, no KL anchor) vs V13.5 baseline
     → did 31K games of dense-reward + strong-opp training help?
  B. V13.5 strong-opp vs V15.1 strong (G=507K, final V15.1 backup)
     → cross-arch comparison
  C. V13.5 baseline vs V15.1 strong
     → reference (we already have this number, but redo for symmetry)

2000 games per pair (1000 per orientation) → ±1.1pp confidence.
Total: 6000 games, ETA ~45-60 min on Mac CPU.

Usage:
    cd /Users/sumit/Github/AlphaLudo/td_ludo
    PYTHONPATH=.:../td_ludo_v15 \
      ./td_env/bin/python -u h2h_3way_strongopp.py
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
# Production adapter (V13.5 baseline)
from td_ludo.models.v13_5_production import V135ProductionAdapter
from td_ludo.game.encoder_v18_production import encode_state_v18_production
# Bare symmetric (V13.5 strong-opp — trained inside train_v135_rl.py)
from td_ludo.models.v13_5 import V135Symmetric, compute_rank_masks
from td_ludo.game.encoder_v18_symmetric import encode_state_v18_symmetric
from td_ludo.game.rank_mapping import (
    state_to_rank_mapping, legal_mask_per_rank, rank_to_token_id,
)
# V15.1
from td_ludo_v15.models.v15 import V15GraphTransformer
from td_ludo_v15.game.encoder import encode_frame
from td_ludo_v15.game.cells import (
    NUM_BOARD_CELLS, cell_to_index, position_to_cell_in_pov,
)
import td_ludo_v15_cpp as v15_cpp

_BASE_POS = v15_cpp.BASE_POS
MAX_MOVES = 400


def _strip(sd):
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    return sd


# ─── Loaders ───────────────────────────────────────────────────────────

def load_v135_production_adapter(path, device):
    """V13.5 baseline checkpoint (G=535K, terminal-only RL, production adapter)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    sd = _strip(sd)
    m = V135ProductionAdapter(num_res_blocks=10, num_channels=128)
    m.load_state_dict(sd, strict=False)
    m.eval().to(device)
    return m


def load_v135_symmetric_bare(path, device):
    """V13.5 strong-opp checkpoint (G=31K, dense rewards, bare V135Symmetric).

    Trained inside train_v135_rl.py. State dict has no `inner.` prefix.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    sd = _strip(sd)
    # Defensive: handle both shapes (bare V135Sym OR wrapped in inner.).
    if any(k.startswith("inner.") for k in sd):
        sd = {k[len("inner."):]: v for k, v in sd.items() if k.startswith("inner.")}
    m = V135Symmetric(num_res_blocks=10, num_channels=128, in_channels=13)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys (likely progress head)")
    m.eval().to(device)
    return m


def load_v151(path, device):
    """V15.1 GraphTransformer (history_len=2)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    sd = _strip(sd)
    m = V15GraphTransformer(d_model=128, n_heads=4, n_layers=4,
                             ffn_dim=256, history_len=2)
    m.load_state_dict(sd, strict=False)
    m.eval().to(device)
    return m


# ─── Pickers ───────────────────────────────────────────────────────────

def make_picker_v135_prod(model, device):
    """V13.5 baseline: V18-production 21-channel input, token-indexed policy."""
    def pick(state, legal, _history):
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
    return pick


def make_picker_v135_sym(model, device):
    """V13.5 strong-opp: V18-symmetric 13-channel input, rank-indexed policy."""
    def pick(state, legal, _history):
        if len(legal) == 1:
            return legal[0]
        cp = int(state.current_player)
        pp = state.player_positions[cp]
        _, rank_tokens = state_to_rank_mapping(pp)
        rank_legal = legal_mask_per_rank(legal, rank_tokens)
        enc = encode_state_v18_symmetric(state).astype(np.float32)
        rm = compute_rank_masks(state).astype(np.float32)
        with torch.no_grad():
            x = torch.from_numpy(enc).unsqueeze(0).to(device)
            rmt = torch.from_numpy(rm).unsqueeze(0).to(device)
            lmt = torch.from_numpy(rank_legal.astype(np.float32)).unsqueeze(0).to(device)
            out = model(x, rmt, lmt)
            policy = out[0]  # rank-indexed, legal-masked, post-softmax
            rank = int(policy.argmax(dim=1).item())
        action = rank_to_token_id(rank, legal, rank_tokens)
        return action if action in legal else legal[0]
    return pick


def make_picker_v151(model, device):
    """V15.1 GraphTransformer (history_len=2)."""
    HISTORY_LEN = 1  # 1 prev + 1 current = 2 frames
    def pick(state, legal, history):
        if len(legal) == 1:
            return legal[0]
        cp = int(state.current_player)
        past = list(history) if history else []
        v15_x = np.zeros((2, 15, 15, 3), dtype=np.float32)
        if past:
            v15_x[0] = encode_frame(past[-1], pov_player=cp)
        v15_x[1] = encode_frame(state, pov_player=cp)
        v15_legal = np.zeros(NUM_BOARD_CELLS, dtype=np.float32)
        legal_cells = []
        for t in legal:
            pos = int(state.player_positions[cp][t])
            c = position_to_cell_in_pov(
                _BASE_POS if pos == _BASE_POS else pos, cp, cp)
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


# ─── Game engine ───────────────────────────────────────────────────────

def run_game(picker_a, picker_b, seed):
    """picker_a is p0, picker_b is p2. Returns 'a', 'b', or 'draw'."""
    random.seed(seed)
    state = cpp.create_initial_state_2p()
    history = collections.deque(maxlen=8)
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


def run_h2h(name_a, pick_a, name_b, pick_b, *, games_per_orientation, label):
    aw = bw = dr = 0
    total_len = 0
    t0 = time.time()
    for i in range(games_per_orientation):
        r, mc = run_game(pick_a, pick_b, seed=i)
        total_len += mc
        if   r == "a": aw += 1
        elif r == "b": bw += 1
        else:          dr += 1
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [{label}] orient1 {i+1}/{games_per_orientation}  "
                  f"| {name_a}={aw} {name_b}={bw} draw={dr}  | "
                  f"{(i+1)/elapsed:.1f} g/s", flush=True)
    for i in range(games_per_orientation):
        r, mc = run_game(pick_b, pick_a, seed=10_000 + i)
        total_len += mc
        if   r == "a": bw += 1
        elif r == "b": aw += 1
        else:          dr += 1
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [{label}] orient2 {i+1}/{games_per_orientation}  "
                  f"| {name_a}={aw} {name_b}={bw} draw={dr}  | "
                  f"{(i+1+games_per_orientation)/elapsed:.1f} g/s",
                  flush=True)
    total = aw + bw + dr
    return {
        "a_name": name_a, "b_name": name_b,
        "a_wins": aw, "b_wins": bw, "draws": dr,
        "a_wr": 100.0 * aw / total if total else 0.0,
        "b_wr": 100.0 * bw / total if total else 0.0,
        "elapsed_sec": time.time() - t0,
        "avg_len": total_len / total if total else 0.0,
        "games": total,
    }


def main():
    device = torch.device("cpu")
    games_per_orientation = 1000  # 2000g total per pair

    print(f"Device: {device}")
    print(f"Games per pair: {2 * games_per_orientation}\n")

    print("Loading 3 contestants...")
    v135_base = load_v135_production_adapter(
        HERE / "play" / "model_weights" / "v13_5" / "model_latest.pt", device)
    v135_strong = load_v135_symmetric_bare(
        HERE / "checkpoints" / "v135_prod_rl_strongopp_FINAL_G31K_backup" / "model_latest.pt",
        device)
    v151_strong = load_v151(
        V15_ROOT / "checkpoints" / "v151_rl_strong_FINAL_G507K_backup" / "model_latest.pt",
        device)

    pick_v135_base = make_picker_v135_prod(v135_base, device)
    pick_v135_strong = make_picker_v135_sym(v135_strong, device)
    pick_v151_strong = make_picker_v151(v151_strong, device)

    print(f"  V13.5 baseline:    {sum(p.numel() for p in v135_base.parameters()):,} params  "
          f"(G=535K canonical RL, V18-production 21ch, terminal rewards)")
    print(f"  V13.5 strong-opp:  {sum(p.numel() for p in v135_strong.parameters()):,} params  "
          f"(G=31K dense rewards, V18-symmetric 13ch, no KL anchor)")
    print(f"  V15.1 strong:      {sum(p.numel() for p in v151_strong.parameters()):,} params  "
          f"(G=507K strong-opp final backup)\n")

    matchups = [
        ("V13.5_strong", pick_v135_strong, "V13.5_base", pick_v135_base),
        ("V13.5_strong", pick_v135_strong, "V15.1_strong", pick_v151_strong),
        ("V13.5_base",   pick_v135_base,   "V15.1_strong", pick_v151_strong),
    ]
    results = {}
    t_all = time.time()
    for name_a, pa, name_b, pb in matchups:
        label = f"{name_a}_vs_{name_b}"
        print(f"\n━━━━━ {label} ({2*games_per_orientation} games) ━━━━━")
        r = run_h2h(name_a, pa, name_b, pb,
                    games_per_orientation=games_per_orientation, label=label)
        results[label] = r
        elapsed_m = r['elapsed_sec'] / 60.0
        print(f"   Result: {name_a}={r['a_wins']} ({r['a_wr']:.1f}%)  "
              f"{name_b}={r['b_wins']} ({r['b_wr']:.1f}%)  "
              f"draws={r['draws']}  "
              f"avg_len={r['avg_len']:.0f}  "
              f"{elapsed_m:.1f} min")

    total_min = (time.time() - t_all) / 60.0
    print(f"\n{'='*72}")
    print(f"TOTAL: {total_min:.1f} min\n")
    print(f"{'Match':<40} {'Winner':<22} {'WR':>8}  ±SE")
    print("-" * 72)
    for k, r in results.items():
        winner = r['a_name'] if r['a_wr'] > r['b_wr'] else r['b_name']
        wr = max(r['a_wr'], r['b_wr'])
        p = wr / 100.0
        se = 100.0 * (p * (1 - p) / r['games']) ** 0.5
        print(f"{k:<40} {winner:<22} {wr:>6.1f}%  ±{se:.1f}pp")

    out = HERE / "h2h_3way_strongopp_results.json"
    with open(out, "w") as f:
        json.dump({
            "matchups": results,
            "games_per_orientation": games_per_orientation,
            "elapsed_min": total_min,
        }, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
