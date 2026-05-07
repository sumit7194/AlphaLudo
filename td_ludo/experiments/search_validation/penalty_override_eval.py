"""Penalty-override inference test (Option 1).

Test whether the bias-penalty signal is correctly directional WITHOUT
running an RL training cycle. Two play modes against bots:

  vanilla   : V12.2 model.argmax (current behavior)
  override  : at each multi-legal decision, simulate apply_move for each
              legal action, compute bias_penalties, then pick action with
              max (policy_prob + penalty_total). Penalty is ≤ 0, so this
              effectively requires alts to beat the model's pick by enough
              policy margin to overcome the penalty hit.

If override > vanilla → penalties are correctly directional → training will work
If override = vanilla → penalties fire on real biases but argmax wasn't wrong on net
If override < vanilla → penalty signal is wrong; fix before training

Run from td_ludo/:
    td_env/bin/python experiments/search_validation/penalty_override_eval.py \
        --games 500 --weights play/model_weights/v12_2/model_latest_pre_exp24_20260430_0128.pt
"""
import argparse
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TD_LUDO_DIR = HERE.parent.parent
sys.path.insert(0, str(TD_LUDO_DIR))

import td_ludo_cpp as ludo_cpp  # noqa: E402
from src.heuristic_bot import get_bot, BOT_REGISTRY  # noqa: E402
from src.config import MAX_MOVES_PER_GAME  # noqa: E402
from td_ludo.models.v12 import AlphaLudoV12  # noqa: E402
from td_ludo.game.bias_penalties import compute_bias_penalties  # noqa: E402

# Import compute_shaped_reward WITHOUT env flag set so it returns ONLY the
# base shaped reward (no bias-penalty addition). We call bias_penalties
# separately to keep the two signals decoupled in the override scoring.
os.environ.pop('LUDO_BIAS_PENALTIES', None)
from td_ludo.game.reward_shaping import compute_shaped_reward  # noqa: E402

DEFAULT_WEIGHTS = TD_LUDO_DIR / 'play' / 'model_weights' / 'v12_2' / 'model_latest_pre_exp24_20260430_0128.pt'


def load_v12_2(weights_path, device):
    model = AlphaLudoV12(
        num_res_blocks=3, num_channels=128,
        num_attn_layers=2, num_heads=4,
        ffn_ratio=4, dropout=0.0, in_channels=33,
    )
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and any(
        isinstance(k, str) and k.startswith('_orig_mod.') for k in ckpt.keys()
    ):
        ckpt = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()}
    if 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
        sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        model.load_state_dict(sd)
    else:
        model.load_state_dict(ckpt)
    model.eval()
    model.to(device)
    return model


def positions_dict(state):
    return {str(p): [int(x) for x in state.player_positions[p]] for p in [0, 2]}


def get_policy_probs(model, state, legal_moves, device):
    """Returns numpy [4] array of legal-masked softmax probs."""
    state_tensor = ludo_cpp.encode_state_v11(state)
    legal_mask = np.zeros(4, dtype=np.float32)
    for m in legal_moves:
        legal_mask[m] = 1.0
    with torch.no_grad():
        s_t = torch.from_numpy(np.asarray(state_tensor)).unsqueeze(0).to(device, dtype=torch.float32)
        m_t = torch.from_numpy(legal_mask).unsqueeze(0).to(device, dtype=torch.float32)
        logits = model.forward_policy_only(s_t, m_t)
        # Softmax over legal-masked logits → probabilities
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return probs


def reconstruct_state(positions, current_player, dice):
    """Build a fresh state with given positions + dice. Used for penalty
    simulation (apply_move mutates, so we need clones)."""
    g = ludo_cpp.create_initial_state_2p()
    pp = list(g.player_positions)
    for pstr, plist in positions.items():
        pp[int(pstr)] = list(int(x) for x in plist)
    g.player_positions = pp
    sc = list(g.scores)
    for pstr, plist in positions.items():
        sc[int(pstr)] = sum(1 for x in plist if int(x) == 99)
    g.scores = sc
    g.current_player = int(current_player)
    g.current_dice_roll = int(dice)
    return g


def select_action(model, state, legal_moves, device, mode, move_count, alpha=1.0):
    """Mode-aware action selection.

    Returns (action, policy_probs_or_None, optional_diagnostic_dict).

    Modes:
      vanilla         : V12.2 policy argmax (uses policy head — baseline)
      reward_search   : score = shaped_reward + bias_penalty (pure reward,
                        NO policy). Failed test — drops 17pp.
      policy_reward   : score = policy_prob + alpha * (shaped_reward + penalty).
                        Policy-informed search: policy as prior, rewards as
                        adjustment. Alpha controls how much reward can override.
      override_strict : minimal-intervention — override only when model has
                        penalty AND zero-penalty alt exists; pick highest-policy
                        zero-penalty alt.
    """
    if len(legal_moves) == 1:
        return legal_moves[0], None, None

    cp = int(state.current_player)
    dice = int(state.current_dice_roll)
    pos_pre = positions_dict(state)

    if mode == 'vanilla':
        probs = get_policy_probs(model, state, legal_moves, device)
        action = int(np.argmax(probs))
        if action not in legal_moves:
            action = random.choice(legal_moves)
        return action, probs, None

    # For reward modes: simulate apply_move per legal action,
    # compute (shaped_reward, bias_penalty) for each.
    scores = {}    # action -> total reward (shaped + penalty)
    diag = {}      # action -> dict for debug
    for a in legal_moves:
        sim_pre = reconstruct_state(pos_pre, cp, dice)
        sim_next = ludo_cpp.apply_move(sim_pre, int(a))
        sim_pre = reconstruct_state(pos_pre, cp, dice)  # reconstruct: apply_move mutated
        base_r = compute_shaped_reward(sim_pre, sim_next, cp)
        ctx = {
            'dice': dice,
            'legal_moves': list(legal_moves),
            'action': int(a),
            'move_count': int(move_count),
        }
        pen, bd = compute_bias_penalties(sim_pre, sim_next, cp, ctx)
        scores[int(a)] = base_r + pen
        diag[int(a)] = {'shaped_reward': base_r, 'penalty': pen, 'breakdown': bd}

    if mode == 'reward_search':
        # Pure: pick action with highest shaped_reward + penalty (no policy at all).
        best_score = max(scores.values())
        candidates = [a for a in legal_moves if abs(scores[a] - best_score) < 1e-9]
        action = candidates[0]
        return action, None, diag

    if mode.startswith('policy_reward'):
        # Policy-informed: score = policy_prob + alpha * (shaped_reward + penalty).
        # Mode name 'policy_reward_a1.0' overrides default alpha.
        if '_a' in mode:
            try:
                alpha = float(mode.split('_a', 1)[1])
            except (ValueError, IndexError):
                pass
        probs = get_policy_probs(model, state, legal_moves, device)
        blended = {a: probs[a] + alpha * scores[a] for a in legal_moves}
        action = max(blended, key=blended.get)
        return action, probs, diag

    if mode == 'override_strict':
        # Use policy as primary, penalty as override gate.
        probs = get_policy_probs(model, state, legal_moves, device)
        model_pick = int(np.argmax(probs))
        if model_pick not in legal_moves:
            model_pick = legal_moves[0]
        # Check if model_pick has any penalty
        if diag[model_pick]['penalty'] < -1e-9:
            # Look for zero-penalty alts
            zero_alts = [a for a in legal_moves if diag[a]['penalty'] >= -1e-9]
            if zero_alts:
                action = max(zero_alts, key=lambda a: probs[a])
                return action, probs, diag
        return model_pick, probs, diag

    raise ValueError(f'Unknown mode: {mode}')


def play_game(model, device, mode, bot_type, seed=None):
    """One game between V12.2 (in given mode) and a heuristic bot.
    Returns (model_won, n_moves, n_overrides_made, n_decisions)."""
    if seed is not None:
        random.seed(seed)
    model_player = random.choice([0, 2])
    opp_player = 2 if model_player == 0 else 0
    bot = get_bot(bot_type, player_id=opp_player)

    state = ludo_cpp.create_initial_state_2p()
    consecutive_sixes = [0, 0, 0, 0]
    move_count = 0
    overrides = 0
    decisions = 0

    while not state.is_terminal and move_count < MAX_MOVES_PER_GAME:
        cp = int(state.current_player)
        if not state.active_players[cp]:
            next_p = (cp + 1) % 4
            while not state.active_players[next_p]:
                next_p = (next_p + 1) % 4
            state.current_player = next_p
            continue

        if state.current_dice_roll == 0:
            state.current_dice_roll = random.randint(1, 6)
            if state.current_dice_roll == 6:
                consecutive_sixes[cp] += 1
            else:
                consecutive_sixes[cp] = 0
            if consecutive_sixes[cp] >= 3:
                next_p = (cp + 1) % 4
                while not state.active_players[next_p]:
                    next_p = (next_p + 1) % 4
                state.current_player = next_p
                state.current_dice_roll = 0
                consecutive_sixes[cp] = 0
                continue

        legal = ludo_cpp.get_legal_moves(state)
        if not legal:
            next_p = (cp + 1) % 4
            while not state.active_players[next_p]:
                next_p = (next_p + 1) % 4
            state.current_player = next_p
            state.current_dice_roll = 0
            continue

        if cp == model_player:
            action, probs, _diag = select_action(
                model, state, list(legal), device, mode, move_count,
            )
            # Track override frequency (any time we differ from model.argmax).
            # For reward_search we still want to know; compute model.argmax cheaply.
            if mode != 'vanilla' and len(legal) > 1:
                if probs is None:
                    probs = get_policy_probs(model, state, list(legal), device)
                model_pick = int(np.argmax(probs))
                if model_pick in legal and action != model_pick:
                    overrides += 1
                decisions += 1
        else:
            action = bot.select_move(state, list(legal))

        state = ludo_cpp.apply_move(state, int(action))
        move_count += 1

    winner = ludo_cpp.get_winner(state) if state.is_terminal else -1
    model_won = (winner == model_player)
    return model_won, move_count, overrides, decisions


def run_mode(model, device, mode, n_games, bots, base_seed=42, log_every=100):
    """Run n_games per bot in given mode. Returns dict of stats."""
    per_bot = defaultdict(lambda: {'wins': 0, 'games': 0, 'overrides': 0, 'decisions': 0})
    t0 = time.time()
    for bot_type in bots:
        for i in range(n_games):
            won, mc, ov, dec = play_game(model, device, mode, bot_type,
                                          seed=base_seed + i)
            per_bot[bot_type]['games'] += 1
            per_bot[bot_type]['overrides'] += ov
            per_bot[bot_type]['decisions'] += dec
            if won:
                per_bot[bot_type]['wins'] += 1
            if (i + 1) % log_every == 0:
                wr = 100 * per_bot[bot_type]['wins'] / per_bot[bot_type]['games']
                print(f'  [{mode:>15}] {bot_type:<12} {i+1}/{n_games} '
                      f'WR={wr:.1f}% ({time.time() - t0:.0f}s)')
        wr = 100 * per_bot[bot_type]['wins'] / per_bot[bot_type]['games']
        print(f'  [{mode:>15}] {bot_type:<12} DONE  WR={wr:.1f}%')
    elapsed = time.time() - t0
    return per_bot, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, default=str(DEFAULT_WEIGHTS),
                    help='Path to V12.2 weights')
    ap.add_argument('--games', type=int, default=500,
                    help='Games per bot per mode (default 500)')
    ap.add_argument('--bots', type=str, default='Expert,Heuristic,Aggressive,Defensive,Racing',
                    help='Comma-separated bot types')
    ap.add_argument('--device', type=str, default='cpu',
                    help='cpu or mps (default cpu — eval is sequential)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--modes', type=str, default='vanilla,reward_search,override_strict',
                    help='Modes to run. vanilla=policy argmax (baseline); '
                         'reward_search=pure shaped_reward+penalty (no policy/value); '
                         'override_strict=policy with penalty as override gate')
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f'Loading weights from {args.weights}')
    model = load_v12_2(args.weights, device)
    print(f'Loaded {sum(p.numel() for p in model.parameters()):,} params')

    bots = args.bots.split(',')
    modes = args.modes.split(',')
    n_games = args.games

    print(f'\nConfig: {n_games} games × {len(bots)} bots × {len(modes)} modes '
          f'= {n_games * len(bots) * len(modes):,} total games')
    print(f'Bots: {bots}')
    print(f'Modes: {modes}\n')

    results = {}
    for mode in modes:
        print(f'\n--- Running mode: {mode} ---')
        per_bot, elapsed = run_mode(
            model, device, mode, n_games, bots, base_seed=args.seed,
        )
        results[mode] = (per_bot, elapsed)

    # Summary
    print('\n' + '=' * 80)
    print('RESULTS SUMMARY')
    print('=' * 80)
    print(f'{"bot":<12} ' + ' '.join(f'{m:>15}' for m in modes) +
          (' ' + ' '.join(f'{m+"-vanilla":>14}' for m in modes if m != 'vanilla')))
    for bot in bots:
        wrs = []
        for mode in modes:
            stats = results[mode][0][bot]
            wr = 100 * stats['wins'] / max(1, stats['games'])
            wrs.append(wr)
        deltas = [wrs[i] - wrs[0] for i in range(1, len(wrs))]
        print(f'{bot:<12} ' + ' '.join(f'{w:>14.1f}%' for w in wrs) +
              ' ' + ' '.join(f'{d:>+13.1f}%' for d in deltas))

    # Aggregate
    print('-' * 80)
    agg = []
    for mode in modes:
        all_wins = sum(s['wins'] for s in results[mode][0].values())
        all_games = sum(s['games'] for s in results[mode][0].values())
        agg.append(100 * all_wins / max(1, all_games))
    deltas = [agg[i] - agg[0] for i in range(1, len(agg))]
    print(f'{"AGGREGATE":<12} ' + ' '.join(f'{w:>14.1f}%' for w in agg) +
          ' ' + ' '.join(f'{d:>+13.1f}%' for d in deltas))

    # Override rates
    print('\nOverride rates (per multi-legal decision):')
    for mode in modes:
        if mode == 'vanilla':
            continue
        for bot in bots:
            s = results[mode][0][bot]
            ov_rate = 100 * s['overrides'] / max(1, s['decisions'])
            print(f'  {mode:>15} vs {bot:<12} : '
                  f'{s["overrides"]:>5}/{s["decisions"]:>5} ({ov_rate:.2f}%)')

    # Timing
    print('\nWall time per mode:')
    for mode, (_, t) in results.items():
        print(f'  {mode:>15}: {t:.0f}s ({t/60:.1f} min)')


if __name__ == '__main__':
    main()
