"""
V10 expectimax evaluation — test whether search over V10's calibrated win_prob
breaks the reactive-CNN plateau.

Search levels:
  d0 (greedy): argmax of policy head (baseline, identical to evaluate_v10)
  d1 (1-ply with chance): for each my legal action, apply → branch over next
      actor's dice roll (6 outcomes) → evaluate each leaf with value head.
      Average value over the 6 dice, pick max-expected-value action.
  d2 (2-ply expectimax): same as d1 but continue one level deeper — after
      opp's dice roll, enumerate opp's legal actions weighted by their policy,
      apply each, then evaluate resulting state. True max-chance-max-chance
      expectimax tree.

Key properties:
- All forward passes are BATCHED per decision (one big model call on all leaves)
- Perspective flip: win_prob is from current_player's POV; flip (1 - p) when
  evaluating a state where opponent is about to act
- Bonus turn handled naturally: if dice was 6, current_player stays after
  apply_move and dice resets to 0 — we branch over next dice as normal
- C++ boundary crossings: ~60-200 per decision, each ~10μs → sub-millisecond
  overhead. Batched forward dominates (~2ms on MPS).

Usage:
  python evaluate_v10_expectimax.py --ckpt checkpoints/ac_v10/model_latest.pt \
    --games 500 --depth 0   # greedy baseline
  python evaluate_v10_expectimax.py --ckpt ... --games 500 --depth 1
  python evaluate_v10_expectimax.py --ckpt ... --games 500 --depth 2
"""
import os
import sys
import random
import time
import argparse
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import td_ludo_cpp as ludo_cpp
from td_ludo.models.v10 import AlphaLudoV10
from src.heuristic_bot import get_bot, BOT_REGISTRY
from src.config import MAX_MOVES_PER_GAME


def _clone_state(state):
    """Create a mutable copy of a GameState. Uses apply_move with -1 hack?
    No — use in-place attribute copying via create + set."""
    # Simplest approach: serialize + deserialize via explicit field copy.
    s = ludo_cpp.GameState()
    s.board = np.array(state.board, dtype=np.int8)
    s.player_positions = np.array(state.player_positions, dtype=np.int8)
    s.scores = np.array(state.scores, dtype=np.int8)
    s.active_players = np.array(state.active_players, dtype=bool)
    s.current_player = int(state.current_player)
    s.current_dice_roll = int(state.current_dice_roll)
    s.is_terminal = bool(state.is_terminal)
    return s


def _encode(state):
    return np.asarray(ludo_cpp.encode_state_v10(state), dtype=np.float32)


def _mask_from_legal(legal):
    m = np.zeros(4, dtype=np.float32)
    for a in legal:
        m[a] = 1.0
    return m


def _forward_batch(states_list, masks_list, model, device):
    """One batched forward pass. Returns (policy, win_prob) numpy arrays."""
    if not states_list:
        return np.zeros((0, 4)), np.zeros((0,))
    s = torch.from_numpy(np.stack(states_list)).to(device)
    m = torch.from_numpy(np.stack(masks_list)).to(device)
    with torch.no_grad():
        policy, win_prob, _moves = model(s, m)
    return policy.cpu().numpy(), win_prob.cpu().numpy()


def pick_action_greedy(state, my_pid, model, device):
    """d0: greedy argmax over policy head. No search."""
    legal = ludo_cpp.get_legal_moves(state)
    if len(legal) == 0:
        return -1
    if len(legal) == 1:
        return legal[0]
    t = _encode(state)
    m = _mask_from_legal(legal)
    policy, _ = _forward_batch([t], [m], model, device)
    a = int(policy[0].argmax())
    return a if a in legal else random.choice(legal)


def pick_action_depth1(state, my_pid, model, device):
    """d1: for each legal action, apply → branch over next actor's dice (6) →
    evaluate leaf. Pick argmax of expected value (averaged over the 6 dice).
    """
    legal = ludo_cpp.get_legal_moves(state)
    if len(legal) == 0:
        return -1
    if len(legal) == 1:
        return legal[0]

    # Phase 1: enumerate all leaves to evaluate
    leaf_tensors = []
    leaf_masks = []
    # leaf_info[i] = (action_idx, weight, perspective_is_mine)
    # weight = 1/6 for each dice branch (uniform prior)
    # perspective_is_mine = True if current_player in leaf == my_pid (e.g. bonus turn)
    leaf_info = []
    # Map action_idx → list of indices into leaves; or precomputed terminal values
    terminal_values = {}  # action_idx → float in [0, 1] from my perspective

    for action_idx, my_action in enumerate(legal):
        s_next = ludo_cpp.apply_move(state, my_action)
        if s_next.is_terminal:
            w = ludo_cpp.get_winner(s_next)
            terminal_values[action_idx] = 1.0 if w == my_pid else 0.0
            continue
        # Branch over next actor's dice
        for dice in range(1, 7):
            s_branch = _clone_state(s_next)
            s_branch.current_dice_roll = dice
            branch_legal = ludo_cpp.get_legal_moves(s_branch)
            if not branch_legal:
                # Empty — can't evaluate meaningfully; fall back to state as-is
                t = _encode(s_branch)
                m = np.ones(4, dtype=np.float32)  # permissive
            else:
                t = _encode(s_branch)
                m = _mask_from_legal(branch_legal)
            is_mine = (s_branch.current_player == my_pid)
            leaf_tensors.append(t)
            leaf_masks.append(m)
            leaf_info.append((action_idx, 1.0 / 6.0, is_mine))

    # Phase 2: batch forward
    _, win_prob = _forward_batch(leaf_tensors, leaf_masks, model, device)

    # Phase 3: aggregate per-action expected value
    action_value_sum = np.zeros(len(legal))
    action_weight_sum = np.zeros(len(legal))

    # Seed with terminal values (weight = 1.0 since deterministic)
    for a_idx, val in terminal_values.items():
        action_value_sum[a_idx] += val
        action_weight_sum[a_idx] += 1.0

    for i, (a_idx, w, is_mine) in enumerate(leaf_info):
        p = float(win_prob[i])
        # Perspective: win_prob is P(current_player_of_leaf wins).
        # If current_player IS me (bonus turn continuation): value = p
        # Else (opp about to act): value = 1 - p (from my perspective)
        my_value = p if is_mine else (1.0 - p)
        action_value_sum[a_idx] += w * my_value
        action_weight_sum[a_idx] += w

    expected_values = np.where(
        action_weight_sum > 0,
        action_value_sum / np.maximum(action_weight_sum, 1e-9),
        -1.0,
    )
    best_action_idx = int(np.argmax(expected_values))
    return legal[best_action_idx]


def pick_action_depth2(state, my_pid, model, device):
    """2-ply expectimax with two batched forward passes.
    Pass A: opp policies at (my_action, opp_dice) nodes.
    Pass B: win_prob at (my_action, opp_dice, opp_action) leaves.
    """
    legal = ludo_cpp.get_legal_moves(state)
    if len(legal) == 0:
        return -1
    if len(legal) == 1:
        return legal[0]

    # Accumulators indexed by action_idx
    num_a = len(legal)
    v_sum = np.zeros(num_a)
    w_sum = np.zeros(num_a)

    def add(ai, weight, value):
        v_sum[ai] += weight * value
        w_sum[ai] += weight

    # --- Step 1: enumerate my actions, classify branches ---
    # Three categories per (action_idx, dice):
    #   (a) Terminal after my action → known value, skip dice branching
    #   (b) Next actor has no legal moves OR it's my bonus turn → evaluate directly (value head)
    #   (c) Opp's turn with opp_legal → expand one more ply using opp policy
    #
    # We collect batched inputs for (b) and (c) nodes, then run forward once to get
    # both win_prob (for b) and policy (for c). Then enumerate c's children for pass B.

    direct_eval_tensors = []  # category (b)
    direct_eval_masks = []
    direct_eval_keys = []  # (action_idx, dice_weight, is_mine)

    expand_node_tensors = []  # category (c)
    expand_node_masks = []
    expand_node_keys = []  # (action_idx, dice_weight, opp_legal, parent_state)

    for ai, my_action in enumerate(legal):
        s_after_me = ludo_cpp.apply_move(state, my_action)
        if s_after_me.is_terminal:
            w = ludo_cpp.get_winner(s_after_me)
            add(ai, 1.0, 1.0 if w == my_pid else 0.0)
            continue
        for dice in range(1, 7):
            s_br = _clone_state(s_after_me)
            s_br.current_dice_roll = dice
            br_legal = ludo_cpp.get_legal_moves(s_br)
            is_opp = (s_br.current_player != my_pid)
            if not br_legal:
                # Evaluate as-is
                direct_eval_tensors.append(_encode(s_br))
                direct_eval_masks.append(np.ones(4, dtype=np.float32))
                direct_eval_keys.append((ai, 1.0 / 6.0, not is_opp))
            elif not is_opp:
                # Bonus turn — evaluate via value head (bounded compute)
                direct_eval_tensors.append(_encode(s_br))
                direct_eval_masks.append(_mask_from_legal(br_legal))
                direct_eval_keys.append((ai, 1.0 / 6.0, True))
            else:
                # Opp's turn → expand one more ply
                expand_node_tensors.append(_encode(s_br))
                expand_node_masks.append(_mask_from_legal(br_legal))
                expand_node_keys.append((ai, 1.0 / 6.0, br_legal, s_br))

    # --- Pass A (direct eval + opp policy nodes in ONE batch) ---
    all_tensors = direct_eval_tensors + expand_node_tensors
    all_masks = direct_eval_masks + expand_node_masks
    all_pol, all_win = _forward_batch(all_tensors, all_masks, model, device)

    # Split results back
    n_direct = len(direct_eval_tensors)
    direct_win = all_win[:n_direct]
    expand_pol = all_pol[n_direct:]

    # Accumulate direct-eval contributions
    for i, (ai, w, is_mine) in enumerate(direct_eval_keys):
        p = float(direct_win[i])
        my_value = p if is_mine else (1.0 - p)
        add(ai, w, my_value)

    # --- Build leaf list (pass B) from expand nodes ---
    leaf_tensors = []
    leaf_masks = []
    leaf_keys = []  # (action_idx, combined_weight, is_mine_at_leaf)

    for node_i, (ai, dice_w, opp_legal, parent) in enumerate(expand_node_keys):
        policy = expand_pol[node_i]
        legal_probs = np.array([policy[a] for a in opp_legal], dtype=np.float64)
        total = legal_probs.sum()
        if total <= 0:
            legal_probs = np.ones(len(opp_legal)) / len(opp_legal)
        else:
            legal_probs = legal_probs / total

        for j, opp_action in enumerate(opp_legal):
            p_opp = float(legal_probs[j])
            if p_opp < 1e-6:
                continue
            s_leaf = ludo_cpp.apply_move(parent, opp_action)
            combined_w = dice_w * p_opp
            if s_leaf.is_terminal:
                w = ludo_cpp.get_winner(s_leaf)
                add(ai, combined_w, 1.0 if w == my_pid else 0.0)
                continue
            # Non-terminal leaf — evaluate via value head
            leaf_legal = ludo_cpp.get_legal_moves(s_leaf)
            if not leaf_legal:
                mask = np.ones(4, dtype=np.float32)
            else:
                mask = _mask_from_legal(leaf_legal)
            leaf_tensors.append(_encode(s_leaf))
            leaf_masks.append(mask)
            is_mine_at_leaf = (s_leaf.current_player == my_pid)
            leaf_keys.append((ai, combined_w, is_mine_at_leaf))

    # --- Pass B (leaf win_prob) ---
    if leaf_tensors:
        _, leaf_win = _forward_batch(leaf_tensors, leaf_masks, model, device)
        for i, (ai, w, is_mine) in enumerate(leaf_keys):
            p = float(leaf_win[i])
            my_value = p if is_mine else (1.0 - p)
            add(ai, w, my_value)

    # --- Pick argmax of expected value ---
    expected = np.where(w_sum > 0, v_sum / np.maximum(w_sum, 1e-9), -1.0)
    return legal[int(np.argmax(expected))]


# Pick the action function based on depth
PICKERS = {
    0: pick_action_greedy,
    1: pick_action_depth1,
    2: pick_action_depth2,
}


def run_eval(model, device, num_games=200, depth=1, verbose=True, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model.eval()
    picker = PICKERS[depth]
    per_bot = defaultdict(lambda: {'wins': 0, 'games': 0})
    wins = 0
    total = 0
    game_lengths = []
    start = time.time()

    for game_idx in range(num_games):
        model_player = random.choice([0, 2])
        opp_player = 2 if model_player == 0 else 0
        bot_type = random.choice(list(BOT_REGISTRY.keys()))
        bot = get_bot(bot_type, player_id=opp_player)

        state = ludo_cpp.create_initial_state_2p()
        consec = [0, 0, 0, 0]
        moves = 0

        while not state.is_terminal and moves < MAX_MOVES_PER_GAME:
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
                action = picker(state, model_player, model, device)
            else:
                action = bot.select_move(state, legal)

            if action not in legal:
                action = random.choice(legal)
            state = ludo_cpp.apply_move(state, action)
            moves += 1

        winner = ludo_cpp.get_winner(state) if state.is_terminal else -1
        won = int(winner == model_player)
        if won:
            wins += 1
        total += 1
        game_lengths.append(moves)
        per_bot[bot_type]['games'] += 1
        per_bot[bot_type]['wins'] += won

        if verbose and (game_idx + 1) % 25 == 0:
            wr = wins / total * 100
            gpm = (game_idx + 1) / ((time.time() - start) / 60)
            print(f"  [{game_idx+1}/{num_games}] WR {wr:.1f}% | {gpm:.0f} gpm", flush=True)

    elapsed = time.time() - start
    wr = wins / total if total else 0
    return {
        'depth': depth,
        'win_rate': wr,
        'win_rate_pct': round(wr * 100, 1),
        'wins': wins,
        'total': total,
        'avg_game_length': float(np.mean(game_lengths)),
        'elapsed_seconds': round(elapsed, 1),
        'games_per_minute': round(total / (elapsed / 60), 1) if elapsed > 0 else 0,
        'per_bot': {
            bt: {'wr': round(s['wins'] / s['games'] * 100, 1), 'games': s['games']}
            for bt, s in sorted(per_bot.items())
        },
    }


def main():
    parser = argparse.ArgumentParser()
    # NOTE: default is model_sl.pt because RL training inverted the win_prob
    # head on model_latest.pt (game 297K) and model_best.pt (game 241K). SL
    # checkpoint has correctly calibrated win_prob — required for expectimax
    # to pick winning states. See V10_RL_VALUE_INVERSION finding in journal.
    parser.add_argument('--ckpt', default='checkpoints/ac_v10/model_sl.pt')
    parser.add_argument('--games', type=int, default=500)
    parser.add_argument('--depth', type=int, default=1, choices=[0, 1, 2])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"[V10 Expectimax] Device: {device}")
    print(f"[V10 Expectimax] Loading: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    arch = ckpt.get('arch', {'num_res_blocks': 6, 'num_channels': 96, 'in_channels': 28})
    model = AlphaLudoV10(**arch).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"[V10 Expectimax] Model: {model.count_parameters():,} params")
    print(f"[V10 Expectimax] Depth: {args.depth}  Games: {args.games}")

    res = run_eval(model, device, num_games=args.games, depth=args.depth, seed=args.seed)

    print(f"\n{'='*60}")
    print(f"  V10 Expectimax @ depth={args.depth} — {args.games} games")
    print(f"{'='*60}")
    print(f"  Win rate:        {res['win_rate_pct']}% ({res['wins']}/{res['total']})")
    print(f"  Avg game len:    {res['avg_game_length']:.1f} moves")
    print(f"  Throughput:      {res['games_per_minute']} games/min")
    print(f"  Wall time:       {res['elapsed_seconds']}s")
    print(f"\n  Per-bot breakdown:")
    for bt, s in res['per_bot'].items():
        print(f"    {bt:<14s}: {s['wr']}% ({s['games']} games)")


if __name__ == '__main__':
    main()
