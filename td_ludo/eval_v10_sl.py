"""
V10 SL evaluation — policy win rate + win_prob calibration + moves_remaining MAE.

After joint SL training, we need to know three things:
  1. Pure-policy WR vs heuristic bots (is the policy head any good?)
  2. Is the win_prob head calibrated? (the thing that failed on V6.3's frozen backbone)
  3. Does moves_remaining track reality? (bonus signal that backbone learned planning features)

Reports Brier score + reliability buckets for calibration.
"""
import os, sys, random, time, argparse
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import td_ludo_cpp as ludo_cpp
from td_ludo.models.v10 import AlphaLudoV10
from src.heuristic_bot import get_bot, BOT_REGISTRY
from src.config import MAX_MOVES_PER_GAME


def eval_v10(model, device, num_games=200, verbose=False):
    """Run games, record policy result + per-decision win_prob + moves predictions."""
    model.eval()
    wins = 0
    total = 0
    game_lengths = []
    per_bot = defaultdict(lambda: {'wins': 0, 'games': 0})

    # Calibration collection — every decision point for the model:
    # (pred_win_prob, pred_moves_remaining, actual_won, actual_moves_remaining)
    per_game_records = []  # list of lists; each inner list is one game's decisions
    start = time.time()

    for game_idx in range(num_games):
        model_player = random.choice([0, 2])
        opp = 2 if model_player == 0 else 0
        bot_type = random.choice(list(BOT_REGISTRY.keys()))
        bot = get_bot(bot_type, player_id=opp)

        state = ludo_cpp.create_initial_state_2p()
        consec = [0, 0, 0, 0]
        move_count = 0
        game_records = []  # (pred_win, pred_moves, own_turn_idx)

        while not state.is_terminal and move_count < MAX_MOVES_PER_GAME:
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

            if cp == model_player:
                if len(legal) == 1:
                    action = legal[0]
                    # Still record heads for calibration
                    enc = np.array(ludo_cpp.encode_state_v10(state), dtype=np.float32)
                    mask = np.zeros(4, dtype=np.float32)
                    mask[legal[0]] = 1.0
                    with torch.no_grad():
                        s_t = torch.from_numpy(enc).unsqueeze(0).to(device)
                        m_t = torch.from_numpy(mask).unsqueeze(0).to(device)
                        _, win_p, moves_p = model(s_t, m_t)
                        game_records.append((win_p.item(), moves_p.item()))
                else:
                    enc = np.array(ludo_cpp.encode_state_v10(state), dtype=np.float32)
                    mask = np.zeros(4, dtype=np.float32)
                    for m in legal:
                        mask[m] = 1.0
                    with torch.no_grad():
                        s_t = torch.from_numpy(enc).unsqueeze(0).to(device)
                        m_t = torch.from_numpy(mask).unsqueeze(0).to(device)
                        pol, win_p, moves_p = model(s_t, m_t)
                        action = pol.argmax(dim=1).item()
                        game_records.append((win_p.item(), moves_p.item()))
                    if action not in legal:
                        action = random.choice(legal)
            else:
                action = bot.select_move(state, legal)

            state = ludo_cpp.apply_move(state, action)
            move_count += 1

        winner = ludo_cpp.get_winner(state) if state.is_terminal else -1
        model_won = int(winner == model_player)
        if model_won: wins += 1
        total += 1
        game_lengths.append(move_count)
        per_bot[bot_type]['games'] += 1
        if model_won: per_bot[bot_type]['wins'] += 1
        per_game_records.append((game_records, model_won))

        if verbose and (game_idx + 1) % 50 == 0:
            wr = wins / total * 100
            gpm = (game_idx + 1) / ((time.time() - start) / 60)
            print(f"  [{game_idx+1}/{num_games}] WR {wr:.1f}% | {gpm:.0f} gpm", flush=True)

    elapsed = time.time() - start
    wr = wins / total if total else 0

    # Calibration: bucket predicted win_probs and see if actual outcomes match
    all_preds = []
    all_outcomes = []
    all_moves_pred = []
    all_moves_true = []
    for game_records, won in per_game_records:
        n = len(game_records)
        for turn_idx, (win_p, moves_p) in enumerate(game_records):
            all_preds.append(win_p)
            all_outcomes.append(won)
            all_moves_pred.append(moves_p)
            all_moves_true.append(n - 1 - turn_idx)  # own-turns remaining from this point

    preds = np.array(all_preds)
    outcomes = np.array(all_outcomes)
    moves_pred = np.array(all_moves_pred)
    moves_true = np.array(all_moves_true)

    brier = float(np.mean((preds - outcomes) ** 2))
    moves_mae = float(np.mean(np.abs(moves_pred - moves_true)))

    # Reliability buckets
    buckets = []
    for lo, hi in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        mask = (preds >= lo) & (preds < hi)
        if mask.sum() > 0:
            buckets.append({
                'range': f'[{lo:.1f}, {hi:.1f})',
                'n': int(mask.sum()),
                'mean_pred': float(preds[mask].mean()),
                'mean_actual': float(outcomes[mask].mean()),
            })

    return {
        'win_rate': wr,
        'win_rate_pct': round(wr * 100, 1),
        'wins': wins,
        'total': total,
        'avg_game_length': float(np.mean(game_lengths)),
        'brier_score': round(brier, 4),
        'moves_mae': round(moves_mae, 2),
        'calibration_buckets': buckets,
        'elapsed_seconds': round(elapsed, 1),
        'per_bot': {
            bt: {
                'wr': round(s['wins'] / s['games'] * 100, 1),
                'games': s['games'],
            }
            for bt, s in sorted(per_bot.items())
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='checkpoints/ac_v10/model_sl.pt')
    parser.add_argument('--games', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[V10 Eval] Device: {device}")
    print(f"[V10 Eval] Loading: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    arch = ckpt.get('arch', {'num_res_blocks': 6, 'num_channels': 96, 'in_channels': 28})
    model = AlphaLudoV10(**arch).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"[V10 Eval] Model: {model.count_parameters():,} params")
    print(f"[V10 Eval] Running {args.games} games...")

    res = eval_v10(model, device, num_games=args.games, verbose=True)

    print(f"\n{'='*60}")
    print(f"  V10 SL Evaluation Results")
    print(f"{'='*60}")
    print(f"  Win rate:        {res['win_rate_pct']}% ({res['wins']}/{res['total']})")
    print(f"  Avg game len:    {res['avg_game_length']:.1f} moves")
    print(f"  Brier score:     {res['brier_score']}   (lower=better, baseline 0.25)")
    print(f"  Moves MAE:       {res['moves_mae']}    (avg own-turns-to-end error)")
    print(f"\n  Per-bot breakdown:")
    for bt, s in res['per_bot'].items():
        print(f"    {bt:14s}: {s['wr']}% ({s['games']} games)")
    print(f"\n  Win-prob calibration (is prediction == reality?):")
    print(f"    {'Range':<14} {'N':>7} {'Predicted':>12} {'Actual':>10}")
    for b in res['calibration_buckets']:
        print(f"    {b['range']:<14} {b['n']:>7} {b['mean_pred']:>12.3f} {b['mean_actual']:>10.3f}")
    print(f"{'='*60}")

    # Quick verdict
    if res['brier_score'] < 0.20:
        print(f"  ✓ win_prob calibration LOOKS GOOD (Brier < 0.20)")
    elif res['brier_score'] < 0.23:
        print(f"  ? win_prob calibration OK-ish (Brier {res['brier_score']})")
    else:
        print(f"  ✗ win_prob calibration WEAK (Brier {res['brier_score']}; random=0.25)")


if __name__ == '__main__':
    main()
