#!/usr/bin/env python3
"""
Head-to-Head Tournament: V13.7 AlphaZero vs V15.2 GraphTransformer & V13.5 ResNet.
Evaluates pure zero-sum 2-player performance across mirrored seat orientations (P0 vs P2).
"""
import sys
import time
import random
import argparse
from pathlib import Path
import numpy as np
import torch

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
TD_LUDO_DIR = ROOT_DIR / "td_ludo"
PYTHON_DIR = ROOT_DIR / "python"
V15_ROOT = ROOT_DIR / "td_ludo_v15"
sys.path.insert(0, str(TD_LUDO_DIR))
sys.path.insert(0, str(PYTHON_DIR))
sys.path.insert(0, str(V15_ROOT))

import td_ludo_cpp as cpp
import td_ludo_v15_cpp as v15_cpp
from td_ludo.game.encoder_v17 import encode_state_v17
from td_ludo.game.encoder_v18_production import encode_state_v18_production
from alphaludo.model_v137 import AlphaLudoV137
from td_ludo_v15.models.v15 import V15GraphTransformer
from td_ludo.models.v13_5_production import V135ProductionAdapter
from td_ludo_v15.game.cells import NUM_BOARD_CELLS, cell_to_index, position_to_cell_in_pov
from td_ludo_v15.game.encoder import encode_frame

_BASE_POS = v15_cpp.BASE_POS


# ─────────────────────────────────────────────────────────────────────────────
# Pickers
# ─────────────────────────────────────────────────────────────────────────────
def pick_v137(model, device, state, legal):
    if len(legal) == 1:
        return legal[0]
    enc = encode_state_v17(state)
    mask = np.zeros(4, dtype=np.float32)
    for m in legal:
        mask[m] = 1.0
    with torch.no_grad():
        x = torch.from_numpy(enc).unsqueeze(0).to(device)
        m = torch.from_numpy(mask).unsqueeze(0).to(device)
        logits, _ = model(x, m)
        action = int(logits.argmax(dim=1).item())
    return action if action in legal else legal[0]


def pick_v152(model, device, state, legal):
    if len(legal) == 1:
        return legal[0]
    cp = int(state.current_player)
    v15_x = np.zeros((1, 15, 15, 3), dtype=np.float32)
    v15_x[0] = encode_frame(state, pov_player=cp)
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
        pol, _ = model(xt, mt)
        chosen_idx = int(pol.argmax(dim=-1).item())
    chosen_cell = divmod(chosen_idx, 15)
    for t, c in legal_cells:
        if c == chosen_cell:
            return t
    return legal[0]


def pick_v135(model, device, state, legal):
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


# ─────────────────────────────────────────────────────────────────────────────
# Match Engine
# ─────────────────────────────────────────────────────────────────────────────
def play_game(p0_picker, p2_picker, seed: int, max_moves: int = 350):
    random.seed(seed)
    np.random.seed(seed)
    state = cpp.create_initial_state_2p()
    moves = 0

    while not state.is_terminal and moves < max_moves:
        cp = int(state.current_player)
        d = random.randint(1, 6)
        state.current_dice_roll = d
        legal = cpp.get_legal_moves(state)
        if not legal:
            state.current_player = 2 if cp == 0 else 0
            state.current_dice_roll = 0
            continue

        if cp == 0:
            act = p0_picker(state, legal)
        else:
            act = p2_picker(state, legal)

        state = cpp.apply_move(state, int(act))
        moves += 1

    winner = cpp.get_winner(state) if state.is_terminal else -1
    return winner, moves


def run_head_to_head(name_a, picker_a, name_b, picker_b, games_per_side=50, base_seed=42):
    """Plays 2*games_per_side: half with A as P0 and half with A as P2."""
    a_wins = 0
    b_wins = 0
    draws = 0
    total_moves = []

    # Orientation 1: A as P0, B as P2
    for i in range(games_per_side):
        w, m = play_game(picker_a, picker_b, seed=base_seed + i)
        total_moves.append(m)
        if w == 0:
            a_wins += 1
        elif w == 2:
            b_wins += 1
        else:
            draws += 1

    # Orientation 2: B as P0, A as P2
    for i in range(games_per_side):
        w, m = play_game(picker_b, picker_a, seed=base_seed + 10000 + i)
        total_moves.append(m)
        if w == 2:
            a_wins += 1
        elif w == 0:
            b_wins += 1
        else:
            draws += 1

    total_games = a_wins + b_wins + draws
    wr_a = (a_wins / total_games) * 100 if total_games > 0 else 0.0
    avg_m = np.mean(total_moves) if total_moves else 0.0
    return {
        "contestant_a": name_a,
        "contestant_b": name_b,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "total_games": total_games,
        "win_rate_a": wr_a,
        "avg_moves": avg_m,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-per-side", type=int, default=50, help="Games per orientation (total = 2 * N)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"\n================================================================================")
    print(f"🏆 ALPHALUDO 2-PLAYER HEAD-TO-HEAD SHOWDOWN")
    print(f"   Device: {device} | Games Per Pair: {args.games_per_side * 2} ({args.games_per_side} as P0, {args.games_per_side} as P2)")
    print(f"================================================================================\n")

    # 1. Load V13.7 Latest
    v137_latest_ck = torch.load(ROOT_DIR / "checkpoints/v13_7/model_latest.pt", map_location=device, weights_only=False)
    m_v137_latest = AlphaLudoV137(17, 6, 128).to(device)
    m_v137_latest.load_state_dict(v137_latest_ck["model_state_dict"])
    m_v137_latest.eval()
    picker_v137_latest = lambda s, l: pick_v137(m_v137_latest, device, s, l)
    states_latest = v137_latest_ck.get("total_states", 0)
    print(f"✅ Loaded V13.7 Latest: {states_latest:,} states trained")

    # 2. Load V13.7 Best (Iter 650)
    v137_best_ck = torch.load(ROOT_DIR / "checkpoints/v13_7/model_best.pt", map_location=device, weights_only=False)
    m_v137_best = AlphaLudoV137(17, 6, 128).to(device)
    m_v137_best.load_state_dict(v137_best_ck["model_state_dict"])
    m_v137_best.eval()
    picker_v137_best = lambda s, l: pick_v137(m_v137_best, device, s, l)
    states_best = v137_best_ck.get("total_states", 0)
    print(f"✅ Loaded V13.7 Best (Peak Checkpoint): {states_best:,} states trained")

    # 3. Load V15.2 (GraphTransformer)
    v152_ck = torch.load(ROOT_DIR / "td_ludo/play/model_weights/v15_2/model_best.pt", map_location=device, weights_only=False)
    m_v152 = V15GraphTransformer(d_model=128, n_heads=4, n_layers=4, ffn_dim=256, history_len=1).to(device)
    m_v152.load_state_dict(v152_ck["model_state_dict"])
    m_v152.eval()
    picker_v152 = lambda s, l: pick_v152(m_v152, device, s, l)
    games_v152 = v152_ck.get("total_games", 0)
    print(f"✅ Loaded V15.2 GraphTransformer: {games_v152:,} games trained (WR 84.6%)")

    # 4. Load V13.5 Best (Production ResNet)
    v135_ck = torch.load(ROOT_DIR / "td_ludo/play/model_weights/v13_5/model_best.pt", map_location=device, weights_only=False)
    m_v135 = V135ProductionAdapter(num_res_blocks=10, num_channels=128).to(device)
    m_v135.load_state_dict(v135_ck.get("model_state_dict", v135_ck), strict=False)
    m_v135.eval()
    picker_v135 = lambda s, l: pick_v135(m_v135, device, s, l)
    print(f"✅ Loaded V13.5 Production ResNet: 10 blocks x 128ch")

    matches = [
        ("V13.7 (Latest, 16.8M)", picker_v137_latest, "V15.2 (GraphTransformer)", picker_v152),
        ("V13.7 (Best, 5.4M)", picker_v137_best, "V15.2 (GraphTransformer)", picker_v152),
        ("V13.7 (Latest, 16.8M)", picker_v137_latest, "V13.5 (Prod ResNet)", picker_v135),
        ("V15.2 (GraphTransformer)", picker_v152, "V13.5 (Prod ResNet)", picker_v135),
    ]

    print("\n" + "=" * 80)
    print("⚔️  COMMENCING HEAD-TO-HEAD MATCHES...")
    print("=" * 80 + "\n")

    results = []
    for name_a, pick_a, name_b, pick_b in matches:
        t0 = time.time()
        print(f"▶️  Matchup: {name_a} vs {name_b} ({args.games_per_side * 2} games)...", end="", flush=True)
        res = run_head_to_head(name_a, pick_a, name_b, pick_b, games_per_side=args.games_per_side)
        dt = time.time() - t0
        print(f" Done in {dt:.1f}s!")
        print(f"   Score: {res['contestant_a']} {res['a_wins']} - {res['b_wins']} {res['contestant_b']} (Draws: {res['draws']})")
        print(f"   Win Rate for {res['contestant_a']}: {res['win_rate_a']:.1f}% (Avg Game: {res['avg_moves']:.0f} moves)\n")
        results.append(res)

    print("=" * 80)
    print("📊 FINAL TOURNAMENT STANDINGS")
    print("=" * 80)
    for r in results:
        winner = r['contestant_a'] if r['win_rate_a'] > 50 else r['contestant_b'] if r['win_rate_a'] < 50 else 'Tied'
        print(f"• {r['contestant_a']} vs {r['contestant_b']}: {r['a_wins']}-{r['b_wins']} ({r['win_rate_a']:.1f}% WR)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
