"""5-way H2H tournament — V13.5_RL_pre, V13.5_exp, V15_SL, V15_RL, V13.2.

Round-robin, mirrored seeds (each pair plays each game twice with sides
swapped). Output: WR matrix.

Models tested
-------------
1. V13.5_RL_pre  — V13.5 RL latest BEFORE the shaping experiment
                   (checkpoint_backups/v135_prod_rl_G779k_20260514_140354).
                   V135ProductionAdapter arch, V18 21-channel encoder,
                   rank-indexed 4-way output.
2. V13.5_exp     — V13.5 latest AFTER the mixed-shaping experiment
                   (td_ludo/checkpoints/v135_shaping_exp/model_latest.pt).
                   Same arch as V13.5_RL_pre.
3. V15_SL        — V15 SL distillation result (3M params matched teacher).
                   V15GraphTransformer arch, per-cell triplet encoder,
                   source-cell 225-way output.
4. V15_RL        — V15 RL after our hyperparam-tuned PPO run on VM.
                   Same arch as V15_SL.
5. V13.2         — Legacy V13.2 (was V13.5's teacher source).
                   MinimalCNN14 arch, V17 17-channel encoder, per-token output.

Usage:
    python3 h2h_5way_tournament.py --games-per-orientation 200
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add repo roots for legacy + V15 imports
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
V15_ROOT = HERE.parent / "td_ludo_v15"
sys.path.insert(0, str(V15_ROOT))

# Legacy engine + encoders
import td_ludo_cpp as cpp
from td_ludo.game.encoder_v17 import encode_state_v17
from td_ludo.game.encoder_v18_symmetric import encode_state_v18_symmetric
from td_ludo.game.encoder_v18_production import encode_state_v18_production  # noqa: F401
from td_ludo.game.rank_mapping import (
    state_to_rank_mapping, legal_mask_per_rank, rank_to_token_id,
)
from td_ludo.models.v13_5 import V135Symmetric, compute_rank_masks
from td_ludo.models.v13_5_production import V135ProductionAdapter
from experiments.distillation_14ch.model_14ch import MinimalCNN14

# V15 imports
from td_ludo_v15.game.cells import (
    NUM_BOARD_CELLS, cell_to_index, position_to_cell_in_pov,
)
from td_ludo_v15.game.encoder import encode_frame
from td_ludo_v15.models.v15 import V15GraphTransformer
import td_ludo_v15_cpp as v15_cpp


_BASE_POS = v15_cpp.BASE_POS
HISTORY_LEN = 7
TOTAL_FRAMES = 8
MAX_MOVES_PER_GAME = 400


# ───── Model loaders ─────────────────────────────────────────────────────
def _strip_prefixes(sd):
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    return sd


def load_v135_prod(path, device):
    """Load V135ProductionAdapter (used by V13.5 RL via train_v12.py)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    sd = _strip_prefixes(sd)
    # V135ProductionAdapter wraps V135Symmetric with inner. prefix
    model = V135ProductionAdapter(num_res_blocks=10, num_channels=128)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_v132(path, device):
    """Load V13.2 MinimalCNN14."""
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    sd = _strip_prefixes(sd)
    # Probe arch from state dict
    cw = sd.get("conv_input.weight")
    nc = int(cw.shape[0])
    nrb_set = set()
    for k in sd:
        if k.startswith("res_blocks."):
            try:
                nrb_set.add(int(k.split(".")[1]))
            except Exception:
                pass
    nrb = max(nrb_set) + 1 if nrb_set else 10
    model = MinimalCNN14(num_res_blocks=nrb, num_channels=nc, in_channels=17)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_v15(path, device):
    """Load V15 GraphTransformer (used by both SL and RL ckpts)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    sd = _strip_prefixes(sd)
    model = V15GraphTransformer(d_model=256, n_heads=8, n_layers=8, ffn_dim=512)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ───── Action pickers (greedy argmax per arch) ──────────────────────────
def pick_v135(model, device, state, legal, history=None):
    """V135ProductionAdapter: V18-production encoder (21 ch) → TOKEN-ID-indexed
    policy. The adapter handles rank-token mapping internally; we pass
    token-id-indexed legal mask and read token-id-indexed argmax directly."""
    if len(legal) == 1:
        return legal[0]
    enc = encode_state_v18_production(state).astype(np.float32)
    # Token-id-indexed legal mask (NOT rank-indexed)
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


def pick_v132(model, device, state, legal, history=None):
    """V13.2 MinimalCNN14: V17 encoder → per-token output."""
    if len(legal) == 1:
        return legal[0]
    enc = encode_state_v17(state).astype(np.float32)
    mask = np.zeros(4, dtype=np.float32)
    for a in legal:
        mask[a] = 1.0
    with torch.no_grad():
        x = torch.from_numpy(enc).unsqueeze(0).to(device)
        m = torch.from_numpy(mask).unsqueeze(0).to(device)
        policy, _, _ = model(x, m)
        a = int(policy.argmax(dim=1).item())
    return a if a in legal else legal[0]


def pick_v15(model, device, state, legal, history):
    """V15: per-cell triplet + 8-frame history → source-cell → token-id."""
    if len(legal) == 1:
        return legal[0]
    cp = int(state.current_player)
    past = list(history) if history else []
    pad = HISTORY_LEN - len(past)
    v15_x = np.zeros((TOTAL_FRAMES, 15, 15, 3), dtype=np.float32)
    real_frames = [None] * pad + past + [state]
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


# ───── Single-game runner (handles dice + 3-six skip) ───────────────────
def play_one(picks, picker_needs_history, seed):
    """picks: {player_id: callable(state, legal, history) → token_id}
    picker_needs_history: {player_id: bool}
    Returns: winner (0..3) or -1 (truncation)."""
    random.seed(seed)
    np.random.seed(seed)
    state = cpp.create_initial_state_2p()
    csix = [0, 0, 0, 0]
    mc = 0
    # Per-player history (only used by V15 pickers)
    import collections
    history = {p: collections.deque(maxlen=HISTORY_LEN) for p in range(4)}

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
        # Push current state into history for V15-style pickers
        for p in range(4):
            history[p].append(state)
        state = cpp.apply_move(state, int(action))
        mc += 1

    if state.is_terminal:
        return int(cpp.get_winner(state))
    return -1


# ───── Tournament runner ────────────────────────────────────────────────
def run_pair(name_a, model_a, picker_a, name_b, model_b, picker_b,
             games_per_orientation, device, seed_base=42, verbose=True):
    """Run a single pair head-to-head with mirrored seeds.
    A plays as P0 for half, P2 for the other half. Returns (a_wins, b_wins, draws)."""
    a_wins = b_wins = draws = 0
    fa = lambda s, l, h, _m=model_a: picker_a(_m, device, s, l, h)
    fb = lambda s, l, h, _m=model_b: picker_b(_m, device, s, l, h)
    needs_hist = {0: True, 2: True}  # safe default — V15 needs it, others ignore
    for g in range(games_per_orientation * 2):
        a_is_p0 = (g % 2 == 0)
        if a_is_p0:
            picks = {0: fa, 2: fb}
            a_player, b_player = 0, 2
        else:
            picks = {0: fb, 2: fa}
            a_player, b_player = 2, 0
        seed = seed_base + (g // 2)
        winner = play_one(picks, needs_hist, seed)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games-per-orientation", type=int, default=200,
                    help="Games each side plays as P0 (so total games per pair = 2×N).")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--out", default="h2h_5way_results.json",
                    help="Where to dump the JSON results")
    args = ap.parse_args()

    if args.device == "auto":
        device = (torch.device("cuda") if torch.cuda.is_available()
                  else torch.device("mps") if torch.backends.mps.is_available()
                  else torch.device("cpu"))
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    repo_root = HERE
    bak_root = HERE.parent / "checkpoint_backups"
    models_root_v15 = HERE.parent / "td_ludo_v15"

    print("\nLoading models...")
    contestants = []
    # 1. V13.5 RL pre-experiment
    p = bak_root / "v135_prod_rl_G779k_20260514_140354" / "model_latest.pt"
    print(f"  V13.5_RL_pre: {p}")
    contestants.append(("V13.5_RL_pre", load_v135_prod(p, device), pick_v135))
    # 2. V13.5 experimental (mixed shaping result)
    p = repo_root / "checkpoints" / "v135_shaping_exp" / "model_latest.pt"
    print(f"  V13.5_exp:    {p}")
    contestants.append(("V13.5_exp", load_v135_prod(p, device), pick_v135))
    # 3. V15 SL
    p = models_root_v15 / "checkpoints" / "v15_sl_v2" / "model_sl.pt"
    print(f"  V15_SL:       {p}")
    contestants.append(("V15_SL", load_v15(p, device), pick_v15))
    # 4. V15 RL
    p = models_root_v15 / "checkpoints" / "v15_rich_phase_l" / "model_latest.pt"
    print(f"  V15_RL:       {p}")
    contestants.append(("V15_RL", load_v15(p, device), pick_v15))
    # 5. V13.2 anchor
    p = repo_root / "checkpoints" / "v132" / "model_latest.pt"
    print(f"  V13.2:        {p}")
    contestants.append(("V13.2", load_v132(p, device), pick_v132))

    print(f"\nRunning round-robin: {len(contestants)} models, "
          f"{args.games_per_orientation * 2} games per pair "
          f"({args.games_per_orientation} per orientation)")
    print(f"Total games: {len(contestants) * (len(contestants) - 1) // 2 * args.games_per_orientation * 2}")
    print()

    # WR matrix: results[name_a][name_b] = (a_wins, b_wins, draws)
    results = {c[0]: {c[0]: None for c in contestants} for c in contestants}
    t_start = time.time()

    for i, (name_a, model_a, picker_a) in enumerate(contestants):
        for j, (name_b, model_b, picker_b) in enumerate(contestants):
            if j <= i:
                continue
            pair_t = time.time()
            print(f"━━ {name_a} vs {name_b} ━━")
            aw, bw, dr = run_pair(
                name_a, model_a, picker_a, name_b, model_b, picker_b,
                args.games_per_orientation, device, args.seed_base,
            )
            results[name_a][name_b] = (aw, bw, dr)
            results[name_b][name_a] = (bw, aw, dr)
            elapsed = time.time() - pair_t
            total = aw + bw + dr
            print(f"  FINAL: {name_a} {100 * aw / total:.1f}% "
                  f"vs {name_b} {100 * bw / total:.1f}% (draws {dr}) "
                  f"in {elapsed:.0f}s\n")

    # ── Summary ────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    names = [c[0] for c in contestants]
    print("=" * 80)
    print(f"Final Win-Rate Matrix (row vs column, % wins for row)  "
          f"[{args.games_per_orientation * 2}g each, {elapsed_total / 60:.1f} min total]")
    print("=" * 80)
    header = "                  " + "  ".join(f"{n:>14}" for n in names)
    print(header)
    for name_a in names:
        row_str = f"{name_a:>16}  "
        for name_b in names:
            if name_a == name_b:
                cell = "    --"
            elif results[name_a][name_b] is None:
                cell = "     ?"
            else:
                aw, bw, dr = results[name_a][name_b]
                total = aw + bw + dr
                cell = f"{100 * aw / total:>5.1f}%"
            row_str += f"  {cell:>14}"
        print(row_str)

    print()
    print("Aggregate WR (average vs all opps):")
    aggs = []
    for name_a in names:
        total_w = total_g = 0
        for name_b in names:
            if name_a == name_b or results[name_a][name_b] is None:
                continue
            aw, bw, dr = results[name_a][name_b]
            total_w += aw
            total_g += aw + bw + dr
        wr = 100 * total_w / total_g if total_g > 0 else 0
        aggs.append((name_a, wr, total_w, total_g))
    aggs.sort(key=lambda x: -x[1])
    for rank, (name, wr, w, g) in enumerate(aggs, start=1):
        print(f"  {rank}. {name:<18}  {wr:>5.1f}%  ({w}/{g})")

    # JSON dump
    with open(args.out, "w") as f:
        json.dump({
            "results": {a: {b: results[a][b] for b in names} for a in names},
            "names": names,
            "games_per_orientation": args.games_per_orientation,
            "elapsed_min": elapsed_total / 60,
            "device": str(device),
        }, f, indent=2, default=str)
    print(f"\nJSON saved → {args.out}")


if __name__ == "__main__":
    main()
