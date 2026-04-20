"""
Evaluate V6.3 with CALIBRATED HEADS doing 1-ply lookahead vs heuristic bots.

At each decision:
  for each legal move m:
      new_state = apply_move(state, m)
      feat = backbone(new_state)
      win_prob = sigmoid(win_head(feat))
      moves_rem = moves_head(feat)
      score = win_prob - λ * moves_rem / MAX_MOVES
  choose move with max score

Compares against:
  - pure policy argmax (baseline)
"""

import argparse
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import td_ludo_cpp as ludo_cpp
from td_ludo.models.v6_3 import AlphaLudoV63
from src.heuristic_bot import get_bot, BOT_REGISTRY
from src.config import MAX_MOVES_PER_GAME

from train_heads_v6_3 import CalibratedHeads


def load_model_with_heads(checkpoint_path, device):
    """Load V6.3 backbone + calibrated heads from a combined checkpoint."""
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    backbone = AlphaLudoV63(num_res_blocks=10, num_channels=128, in_channels=27)
    backbone.load_state_dict(ck['backbone_state_dict'])
    backbone.to(device).eval()
    heads = CalibratedHeads(feature_dim=128, hidden_dim=64)
    heads.load_state_dict(ck['heads_state_dict'])
    heads.to(device).eval()
    mr_mean = float(ck.get('mr_mean', 46.0))
    mr_std = float(ck.get('mr_std', 25.0))
    return backbone, heads, mr_mean, mr_std


def select_move_policy(backbone, device, state, legal, consec):
    if len(legal) == 1:
        return legal[0]
    cp = state.current_player
    t = torch.from_numpy(np.array(
        ludo_cpp.encode_state_v6_3(state, int(consec[cp])), dtype=np.float32
    )).unsqueeze(0).to(device)
    mask = np.zeros(4, dtype=np.float32)
    for m in legal:
        mask[m] = 1.0
    m_t = torch.from_numpy(mask).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = backbone.forward_policy_only(t, m_t)
        action = int(logits.argmax(dim=1).item())
    return action if action in legal else random.choice(legal)


def select_move_lookahead(backbone, heads, device, state, legal, consec, lam):
    """1-ply lookahead using calibrated heads. Pick move maximizing
    win_prob - lam * (moves_remaining / MAX_MOVES).

    Post-move value is interpreted from current-player's POV of new state:
      - If new current_player == us, head output is OUR win_prob → use as-is
      - If switched to opponent, use (1 - win_prob) as OUR win_prob
    For moves_remaining, smaller is better for us when winning, so:
      - If we're winning, minimize moves_remaining
      - If we're losing, we don't care much about moves (but still prefer slow)
    """
    if len(legal) == 1:
        return legal[0]
    me = state.current_player

    candidates = []
    for move in legal:
        new_state = ludo_cpp.apply_move(state, move)
        if new_state.is_terminal:
            winner = ludo_cpp.get_winner(new_state)
            score = 1e6 if winner == me else -1e6
            candidates.append((score, move))
            continue

        new_cp = new_state.current_player
        if new_cp == me and state.current_dice_roll == 6:
            new_consec = min(consec[me] + 1, 2)
        else:
            new_consec = 0

        t = torch.from_numpy(np.array(
            ludo_cpp.encode_state_v6_3(new_state, int(new_consec)), dtype=np.float32
        )).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = backbone._backbone(t)
            win_logit, moves_pred = heads(feat)
            win_prob = torch.sigmoid(win_logit).item()
            mr = max(0.0, moves_pred.item())

        # Translate to "our" win prob
        our_wp = win_prob if new_cp == me else 1.0 - win_prob
        # Score: maximize our_wp, minimize remaining moves (small penalty)
        score = our_wp - lam * mr / 150.0
        candidates.append((score, move))

    best_score, best_move = max(candidates, key=lambda x: x[0])
    return best_move


def evaluate(backbone, heads, mr_mean, mr_std, device, num_games, mode,
             bot_types, lam=0.1, verbose=True):
    backbone.eval()
    heads.eval()

    if mode == 'lookahead':
        select_fn = lambda s, l, c: select_move_lookahead(backbone, heads, device, s, l, c, lam)
    else:
        select_fn = lambda s, l, c: select_move_policy(backbone, device, s, l, c)

    available = bot_types or list(BOT_REGISTRY.keys())
    wins = 0
    total = 0
    per_bot = defaultdict(lambda: {'wins': 0, 'games': 0})
    t0 = time.time()

    for g in range(num_games):
        mp = random.choice([0, 2])
        op = 2 if mp == 0 else 0
        bot_type = random.choice(available)
        bot = get_bot(bot_type, player_id=op)
        state = ludo_cpp.create_initial_state_2p()
        consec = [0, 0, 0, 0]
        mc = 0

        while not state.is_terminal and mc < MAX_MOVES_PER_GAME:
            cp = state.current_player
            if not state.active_players[cp]:
                nxt = (cp + 1) % 4
                while not state.active_players[nxt]:
                    nxt = (nxt + 1) % 4
                state.current_player = nxt
                continue
            if state.current_dice_roll == 0:
                state.current_dice_roll = random.randint(1, 6)
                if state.current_dice_roll == 6:
                    consec[cp] += 1
                else:
                    consec[cp] = 0
                if consec[cp] >= 3:
                    nxt = (cp + 1) % 4
                    while not state.active_players[nxt]:
                        nxt = (nxt + 1) % 4
                    state.current_player = nxt
                    state.current_dice_roll = 0
                    consec[cp] = 0
                    continue
            legal = ludo_cpp.get_legal_moves(state)
            if not legal:
                nxt = (cp + 1) % 4
                while not state.active_players[nxt]:
                    nxt = (nxt + 1) % 4
                state.current_player = nxt
                state.current_dice_roll = 0
                continue
            if cp == mp:
                action = select_fn(state, legal, consec)
            else:
                action = bot.select_move(state, legal)
            state = ludo_cpp.apply_move(state, action)
            mc += 1

        winner = ludo_cpp.get_winner(state) if state.is_terminal else -1
        w = (winner == mp)
        wins += int(w)
        total += 1
        per_bot[bot_type]['games'] += 1
        if w:
            per_bot[bot_type]['wins'] += 1

        if verbose and (g + 1) % 25 == 0:
            wr = wins / total * 100
            dt = time.time() - t0
            gpm = (g + 1) / (dt / 60)
            print(f"  [{g+1}/{num_games}] WR: {wr:.1f}% ({wins}/{total}) | {gpm:.0f} GPM")

    dt = time.time() - t0
    print(f"\n=== mode={mode}, lambda={lam} ===")
    print(f"  Overall: {wins}/{total} = {wins/total*100:.1f}%")
    print(f"  Duration: {dt:.1f}s ({total/(dt/60):.0f} GPM)")
    for bt in sorted(per_bot.keys()):
        s = per_bot[bt]
        print(f"    vs {bt}: {s['wins']}/{s['games']} = {s['wins']/s['games']*100:.1f}%")
    return wins / total


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='checkpoints/ac_v6_3_capture/model_heads.pt')
    parser.add_argument('--games', type=int, default=200)
    parser.add_argument('--bots', nargs='+', default=['Expert'])
    parser.add_argument('--mode', choices=['policy', 'lookahead'], default='lookahead')
    parser.add_argument('--lam', type=float, default=0.1,
                        help='Weight on moves_remaining penalty')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    backbone, heads, mr_mean, mr_std = load_model_with_heads(args.model, device)
    print(f"Loaded {args.model}")
    print(f"mode={args.mode}, games={args.games}, bots={args.bots}, lam={args.lam}")
    evaluate(backbone, heads, mr_mean, mr_std, device, args.games, args.mode,
             args.bots, lam=args.lam)
