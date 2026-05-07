"""Head-to-head: V12.2-bias (post-bias-penalty training) vs V12.2-pre-search.

Both models are AlphaLudoV12 (3 ResBlocks × 128ch, 33ch v11 encoder), so this
is a clean weight-only comparison. Each game has model A in one seat (P0 or
P2) and model B in the other; seats are alternated every game so neither side
gets the seat advantage.

Reports aggregate WR for model A (V12.2-bias) over total games.

Run from td_ludo/:
    td_env/bin/python experiments/search_validation/h2h_v122.py --games 1000
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TD_LUDO_DIR = HERE.parent.parent
sys.path.insert(0, str(TD_LUDO_DIR))

import td_ludo_cpp as ludo_cpp  # noqa: E402
from src.config import MAX_MOVES_PER_GAME  # noqa: E402
from td_ludo.models.v12 import AlphaLudoV12  # noqa: E402

# Default paths
NEW_CKPT = TD_LUDO_DIR.parent / 'checkpoint_backups'
OLD_CKPT = TD_LUDO_DIR / 'play' / 'model_weights' / 'v12_2' / 'model_latest_pre_exp24_20260430_0128.pt'


def build_v122():
    return AlphaLudoV12(
        num_res_blocks=3, num_channels=128,
        num_attn_layers=2, num_heads=4,
        ffn_ratio=4, dropout=0.0, in_channels=33,
    )


def load(weights, device):
    m = build_v122()
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and any(
        isinstance(k, str) and k.startswith('_orig_mod.') for k in ckpt.keys()
    ):
        ckpt = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()}
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
        sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        m.load_state_dict(sd)
    else:
        m.load_state_dict(ckpt)
    m.eval().to(device)
    return m


def model_argmax(model, state, legal_moves, device):
    state_tensor = ludo_cpp.encode_state_v11(state)
    legal_mask = np.zeros(4, dtype=np.float32)
    for m in legal_moves:
        legal_mask[m] = 1.0
    with torch.no_grad():
        s_t = torch.from_numpy(np.asarray(state_tensor)).unsqueeze(0).to(device, dtype=torch.float32)
        m_t = torch.from_numpy(legal_mask).unsqueeze(0).to(device, dtype=torch.float32)
        logits = model.forward_policy_only(s_t, m_t)
        a = int(logits.argmax(dim=1).item())
    return a if a in legal_moves else random.choice(legal_moves)


def play_game(model_a, model_b, device, a_player, seed=None):
    """a_player ∈ {0, 2}; b takes the other seat. Returns 1 if A wins, 0 else."""
    if seed is not None:
        random.seed(seed)
    b_player = 2 if a_player == 0 else 0
    state = ludo_cpp.create_initial_state_2p()
    csix = [0, 0, 0, 0]
    moves = 0
    while not state.is_terminal and moves < MAX_MOVES_PER_GAME:
        cp = int(state.current_player)
        if not state.active_players[cp]:
            n = (cp + 1) % 4
            while not state.active_players[n]:
                n = (n + 1) % 4
            state.current_player = n
            continue
        if state.current_dice_roll == 0:
            state.current_dice_roll = random.randint(1, 6)
            if state.current_dice_roll == 6:
                csix[cp] += 1
            else:
                csix[cp] = 0
            if csix[cp] >= 3:
                n = (cp + 1) % 4
                while not state.active_players[n]:
                    n = (n + 1) % 4
                state.current_player = n
                state.current_dice_roll = 0
                csix[cp] = 0
                continue
        legal = ludo_cpp.get_legal_moves(state)
        if not legal:
            n = (cp + 1) % 4
            while not state.active_players[n]:
                n = (n + 1) % 4
            state.current_player = n
            state.current_dice_roll = 0
            continue
        if len(legal) == 1:
            action = legal[0]
        else:
            picker = model_a if cp == a_player else model_b
            action = model_argmax(picker, state, list(legal), device)
        state = ludo_cpp.apply_move(state, int(action))
        moves += 1
    winner = ludo_cpp.get_winner(state) if state.is_terminal else -1
    return 1 if winner == a_player else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--new', type=str, required=True, help='New V12.2-bias weights')
    ap.add_argument('--old', type=str, default=str(OLD_CKPT), help='Old V12.2 pre-search weights')
    ap.add_argument('--games', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', type=str, default='cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f'Loading NEW: {args.new}')
    model_new = load(args.new, device)
    print(f'Loading OLD: {args.old}')
    model_old = load(args.old, device)

    print(f'\nPlaying {args.games} games (alternating seats every game)\n')
    new_wins = 0
    t0 = time.time()
    for i in range(args.games):
        # Alternate seats: even games new=P0, odd new=P2
        a_player = 0 if (i % 2 == 0) else 2
        won = play_game(model_new, model_old, device, a_player, seed=args.seed + i)
        if won:
            new_wins += 1
        if (i + 1) % 100 == 0:
            wr = 100 * new_wins / (i + 1)
            print(f'  [{i+1:>4}/{args.games}] NEW WR = {wr:.1f}%  ({time.time()-t0:.0f}s)')

    wr = 100 * new_wins / args.games
    # 95% CI for proportion
    ci = 1.96 * (wr * (100 - wr) / args.games) ** 0.5
    print(f'\n{"="*50}')
    print(f'NEW (V12.2-bias)  vs  OLD (V12.2-pre-search)')
    print(f'{"="*50}')
    print(f'NEW wins:  {new_wins}/{args.games} = {wr:.1f}%  (95% CI ±{ci:.1f}%)')
    print(f'OLD wins:  {args.games - new_wins}/{args.games} = {100-wr:.1f}%')
    if wr > 50 + ci:
        print(f'\n→ NEW is statistically stronger (margin = +{wr-50:.1f}pp > {ci:.1f}pp CI)')
    elif wr < 50 - ci:
        print(f'\n→ OLD is statistically stronger (margin = -{50-wr:.1f}pp > {ci:.1f}pp CI)')
    else:
        print(f'\n→ Statistically tied (margin = {wr-50:+.1f}pp within ±{ci:.1f}pp CI)')


if __name__ == '__main__':
    main()
