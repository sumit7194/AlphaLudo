"""Step 1 — Search-augmented training-data generator (2-ply expectimax).

For each visited state during V13.2-vs-V13.2 self-play, runs 2-ply
expectimax search using V13.2's value head as the leaf evaluator:

  My move (4 actions)
    → Opp dice (chance node, 6 outcomes uniformly)
      → Opp move (best response — opp picks max over their actions)
        → V13.2.value at the leaf

  Q(s, a) = -E_dice[ max_opp_a' V(state_after(a, dice, opp_a')) ]
        // negate since opp's max is our loss

Outputs per state:
  search_action  = argmax_a Q(s, a)
  search_value   = max_a Q(s, a)
  search_policy  = softmax(Q / TEMP)   // soft target, TEMP=0.5

Stored as a parquet/npz buffer for the SL distillation trainer in
`train_search_distill.py`.

This script is the cheap proxy for full MCTS — if 2-ply expectimax
distillation produces a stronger model than V13.2, full MCTS
likely will too. If 2-ply gives nothing, MCTS is unlikely to help
in this codebase.

Usage:
    python -m experiments.mcts_v1.generate_search_data \\
        --teacher checkpoints/v132/model_latest.pt \\
        --target-states 1000000 \\
        --batch-size 200 \\
        --output runs/mcts_v1_search_buffer.npz
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import td_ludo_cpp as ludo_cpp
from td_ludo.game.encoder_v17 import encode_state_v17, V17_CHANNELS
from experiments.distillation_14ch.model_14ch import MinimalCNN14


SCORE_POSITION = 99
BASE_POSITION = -1
NUM_TOKENS = 4
DICE_VALUES = (1, 2, 3, 4, 5, 6)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", required=True, help="V13.2 checkpoint")
    p.add_argument("--target-states", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=200,
                   help="Parallel games per VectorGameState batch")
    p.add_argument("--policy-temperature", type=float, default=0.5,
                   help="Softmax temperature for search_policy soft target")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="runs/mcts_v1_search_buffer.npz")
    p.add_argument("--num-res-blocks", type=int, default=10)
    p.add_argument("--num-channels", type=int, default=128)
    p.add_argument("--max-turns-per-game", type=int, default=400)
    p.add_argument("--save-every", type=int, default=100_000,
                   help="Flush partial buffer to disk every N states (failsafe)")
    return p.parse_args()


def pick_device(name):
    if name in ("cuda", "cpu", "mps"):
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_v132(path, device, num_res_blocks, num_channels):
    print(f"[Gen] Loading V13.2 teacher from {path}...")
    model = MinimalCNN14(
        num_res_blocks=num_res_blocks,
        num_channels=num_channels,
        in_channels=V17_CHANNELS,
    )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[Gen] Loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    return model


def _advance_to_decision(game, rng):
    """Spin a game forward to its next decision state (or terminal).
    Returns True if we have a decision to make, False if game ended."""
    while not game.is_terminal:
        if game.current_dice_roll == 0:
            game.current_dice_roll = int(rng.integers(1, 7))
        legal = ludo_cpp.get_legal_moves(game)
        if legal:
            return True
        cp = int(game.current_player)
        nxt = (cp + 1) % 4
        while not game.active_players[nxt]:
            nxt = (nxt + 1) % 4
        game.current_player = nxt
        game.current_dice_roll = 0
    return False


def _expand_two_ply_leaves(game):
    """Enumerate all 2-ply leaf states from `game`.

    Returns:
      leaves:   list of GameState, one per leaf (own action × dice × opp action)
      meta:     list of dicts {own_action, dice, opp_action, opp_terminal}
                — opp_terminal=True means the game ended after opp's move,
                  so the leaf value is determined by the outcome (no V call needed)
      legal_own: list of int — own legal actions at the root
      structure_per_action: dict[own_action] -> list of (dice, [(opp_action, leaf_index)])
                indices into the `leaves` array. opp_action == -1 means
                "opp had no legal moves" (turn passes back to us).
    """
    legal_own = list(ludo_cpp.get_legal_moves(game))
    leaves = []
    meta = []
    structure_per_action = {a: {} for a in legal_own}

    for own_a in legal_own:
        # Apply our move with the CURRENT dice (already known at root)
        s_after_own = ludo_cpp.apply_move(game, int(own_a))
        for dice in DICE_VALUES:
            # If our move terminated the game, opp doesn't get a turn —
            # the leaf is the post-our-move state with known outcome.
            if s_after_own.is_terminal:
                leaf = s_after_own
                terminal_winner = int(ludo_cpp.get_winner(leaf))
                meta.append({
                    "own_action": int(own_a),
                    "dice": dice,
                    "opp_action": -1,
                    "terminal_winner": terminal_winner,
                })
                leaves.append(leaf)
                structure_per_action[own_a].setdefault(dice, []).append(
                    (-1, len(leaves) - 1)
                )
                continue

            # Opp's turn: set dice, find legal opp moves
            s_for_opp = s_after_own
            # Set dice for whoever's turn it is now (could be us if we rolled 6)
            s_for_opp.current_dice_roll = dice
            opp_legals = list(ludo_cpp.get_legal_moves(s_for_opp))

            if not opp_legals:
                # Opp has no legal moves — turn passes. The post-pass state
                # is the leaf (with the next player's dice unknown; V at this
                # state captures expected value over unknown dice).
                # In Ludo, "pass" means cycle to next active player, dice=0.
                cp = int(s_for_opp.current_player)
                nxt = (cp + 1) % 4
                while not s_for_opp.active_players[nxt]:
                    nxt = (nxt + 1) % 4
                # Construct the post-pass state
                leaf = s_for_opp
                leaf.current_player = nxt
                leaf.current_dice_roll = 0
                meta.append({
                    "own_action": int(own_a),
                    "dice": dice,
                    "opp_action": -1,
                    "terminal_winner": -1,
                })
                leaves.append(leaf)
                structure_per_action[own_a].setdefault(dice, []).append(
                    (-1, len(leaves) - 1)
                )
                continue

            # Opp has legal moves — try each, leaf = state-after-opp-move
            for opp_a in opp_legals:
                s_leaf = ludo_cpp.apply_move(s_for_opp, int(opp_a))
                terminal_winner = (
                    int(ludo_cpp.get_winner(s_leaf)) if s_leaf.is_terminal else -1
                )
                meta.append({
                    "own_action": int(own_a),
                    "dice": dice,
                    "opp_action": int(opp_a),
                    "terminal_winner": terminal_winner,
                })
                leaves.append(s_leaf)
                structure_per_action[own_a].setdefault(dice, []).append(
                    (int(opp_a), len(leaves) - 1)
                )

    return leaves, meta, legal_own, structure_per_action


def _aggregate_q_per_action(leaves, meta, leaf_values, legal_own, structure, root_player):
    """Aggregate leaf V values up to per-own-action Q values.

    Algorithm:
      For each own action a:
        For each dice d:
          opp_q = max over opp actions of V(leaf), where V is from
                  leaf state's current_player POV.
          But V at leaf is from the LEAF's current_player POV. We need
          "value from opponent's POV after THEIR move." If opp made the
          move, the leaf's current_player is the *next* player after opp
          (which could be us, or opp again on a 6, or another player in 4P).
          Sign convention: we want value "from root_player's POV".

      For sign-flipping: walking back from leaf, flip when crossing
      player boundaries. Standard MCTS perspective math.

    For 2-ply expectimax with known root:
      - At leaf: V_leaf is from leaf.current_player POV
      - Map to root POV: V_root = V_leaf if leaf.current_player == root_player
                                 else -V_leaf
      - Opp picks min over their actions (they want their V high → our V low)
      - We average over dice (1/6 each)
      - We pick max over our actions

    Returns:
      Q: dict[own_action] -> Q value from root_player POV
    """
    # Map leaf index → V_root (value from root_player's POV)
    leaf_v_root = np.zeros(len(leaves), dtype=np.float32)
    for idx, m in enumerate(meta):
        leaf = leaves[idx]
        # Terminal handling: if a winner exists, value is ±1 from root POV
        if m["terminal_winner"] >= 0:
            leaf_v_root[idx] = 1.0 if m["terminal_winner"] == root_player else -1.0
        else:
            v = float(leaf_values[idx])  # V(s) ∈ [0, 1] = win_prob from leaf.current_player POV
            v_signed = 2.0 * v - 1.0     # to [-1, +1]
            if int(leaf.current_player) == root_player:
                leaf_v_root[idx] = v_signed
            else:
                leaf_v_root[idx] = -v_signed

    Q = {}
    for own_a in legal_own:
        dice_to_pairs = structure[own_a]
        # Average over dice (1/6 each, even if some dice values weren't expanded — shouldn't happen)
        sum_over_dice = 0.0
        for dice in DICE_VALUES:
            pairs = dice_to_pairs.get(dice, [])
            if not pairs:
                continue
            # Among opp's choices, opp picks the one MAXIMIZING their value (our loss).
            # In root POV: opp picks the min V_root among their options.
            best_opp_v_root = min(leaf_v_root[idx] for (_, idx) in pairs)
            sum_over_dice += best_opp_v_root
        Q[own_a] = sum_over_dice / 6.0  # uniform expectation over dice

    return Q


def _q_to_search_targets(Q, legal_own, temperature):
    """Convert Q dict to (search_policy, search_value, search_action)."""
    # search_policy is over all 4 token slots (illegal = 0)
    policy = np.zeros(NUM_TOKENS, dtype=np.float32)
    if not Q:
        # No legal actions — shouldn't reach here, but be safe
        return policy, 0.0, -1

    q_legal = np.array([Q[a] for a in legal_own], dtype=np.float32)
    # Soft target via softmax(Q / temp)
    q_t = q_legal / max(temperature, 1e-6)
    q_t = q_t - q_t.max()
    e = np.exp(q_t)
    soft = e / e.sum()
    for k, a in enumerate(legal_own):
        policy[a] = soft[k]

    best_idx = int(np.argmax(q_legal))
    search_action = int(legal_own[best_idx])
    search_value = float(q_legal[best_idx])  # max Q from root POV ∈ [-1, +1]
    return policy, search_value, search_action


def _save_buffer(out_path, states, search_policies, search_values, search_actions, outcomes):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        states=np.array(states, dtype=np.float32),
        search_policies=np.array(search_policies, dtype=np.float32),
        search_values=np.array(search_values, dtype=np.float32),
        search_actions=np.array(search_actions, dtype=np.int8),
        outcomes=np.array(outcomes, dtype=np.float32),
    )
    print(f"[Gen] Saved buffer → {out_path} ({len(states):,} states)")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = pick_device(args.device)
    print(f"[Gen] Device: {device}")
    teacher = load_v132(args.teacher, device, args.num_res_blocks, args.num_channels)

    env = ludo_cpp.VectorGameState(batch_size=args.batch_size, two_player_mode=True)
    pending = [[] for _ in range(args.batch_size)]
    turn_count = np.zeros(args.batch_size, dtype=np.int32)

    out_states = []
    out_policies = []
    out_values = []
    out_actions = []
    out_outcomes = []
    finalized_states = 0

    t_start = time.time()
    last_log = t_start
    last_save = 0
    print(f"[Gen] Generating {args.target_states:,} states "
          f"(batch={args.batch_size}, policy_temp={args.policy_temperature})...")

    while finalized_states < args.target_states:
        # Spin every game forward to its next decision state
        decision_idxs = []
        decision_games = []
        for i in range(args.batch_size):
            game = env.get_game(i)
            if game.is_terminal or turn_count[i] >= args.max_turns_per_game:
                continue
            ok = _advance_to_decision(game, rng)
            if not ok:
                continue
            decision_idxs.append(i)
            decision_games.append(game)

        if not decision_idxs:
            # All games terminal/timeout → reset terminated games and continue
            for i in range(args.batch_size):
                env.reset_game(i)
                turn_count[i] = 0
            continue

        # For each decision game, expand 2-ply leaves
        per_game_leaves = []      # list of (leaves_list, meta_list, legal_own, structure)
        all_leaves_for_batch = []  # flat list of leaf states (for batched encoding+inference)
        leaf_offsets = []          # start index of each game's leaves in the flat list
        for game in decision_games:
            leaves, meta, legal_own, structure = _expand_two_ply_leaves(game)
            per_game_leaves.append((leaves, meta, legal_own, structure))
            leaf_offsets.append(len(all_leaves_for_batch))
            all_leaves_for_batch.extend(leaves)
        leaf_offsets.append(len(all_leaves_for_batch))  # sentinel

        # Encode all leaves in one shot, batched inference
        if all_leaves_for_batch:
            encoded = np.stack(
                [encode_state_v17(s) for s in all_leaves_for_batch], axis=0
            )
            x = torch.from_numpy(encoded).to(device, dtype=torch.float32)
            # No mask matters for value — but model expects one. Permissive mask works.
            mask = torch.ones(len(all_leaves_for_batch), 4, device=device)
            with torch.no_grad():
                _, win_prob_batch, _ = teacher(x, mask)
            leaf_values = win_prob_batch.cpu().numpy().reshape(-1)
        else:
            leaf_values = np.array([], dtype=np.float32)

        # Aggregate Q per game, derive search targets, pick action
        actions = [-1] * args.batch_size
        for k, i in enumerate(decision_idxs):
            game = decision_games[k]
            leaves, meta, legal_own, structure = per_game_leaves[k]
            start, end = leaf_offsets[k], leaf_offsets[k + 1]
            game_leaf_v = leaf_values[start:end]
            root_player = int(game.current_player)

            Q = _aggregate_q_per_action(
                leaves, meta, game_leaf_v, legal_own, structure, root_player,
            )
            policy, value, search_action = _q_to_search_targets(
                Q, legal_own, args.policy_temperature,
            )

            # Encode root state for the buffer (the one we're training on)
            state_tensor = encode_state_v17(game)

            # Sample action from soft policy for diverse self-play data
            # (so collected states aren't all argmax-driven greedy paths)
            try:
                action = int(rng.choice(NUM_TOKENS, p=policy))
            except ValueError:
                # If policy degenerate, fall back to search_action
                action = search_action
            if action not in legal_own:
                action = search_action

            # Buffer the search-target row, will get outcome label later
            pending[i].append({
                "state": state_tensor,
                "search_policy": policy,
                "search_value": value,
                "search_action": search_action,
                "current_player": root_player,
            })
            actions[i] = action

        # Step env
        _, _, _, infos = env.step(actions)
        for i in range(args.batch_size):
            if actions[i] >= 0:
                turn_count[i] += 1

        # Finalize terminated games + max-turns timeouts
        for i, info in enumerate(infos):
            timed_out = (turn_count[i] >= args.max_turns_per_game) and not info["is_terminal"]
            if info["is_terminal"] or timed_out:
                if info["is_terminal"]:
                    winner = int(info["winner"])
                    for rec in pending[i]:
                        outcome = 1.0 if rec["current_player"] == winner else 0.0
                        out_states.append(rec["state"])
                        out_policies.append(rec["search_policy"])
                        out_values.append(rec["search_value"])
                        out_actions.append(rec["search_action"])
                        out_outcomes.append(outcome)
                        finalized_states += 1
                # else: timed out, discard pending (no clean outcome)
                pending[i].clear()
                env.reset_game(i)
                turn_count[i] = 0
                if finalized_states >= args.target_states:
                    break

        # Progress + periodic save
        now = time.time()
        if now - last_log > 5.0:
            elapsed = now - t_start
            sps = finalized_states / max(1e-6, elapsed)
            pending_total = sum(len(p) for p in pending)
            print(f"  [{finalized_states:>7}/{args.target_states:,}]  "
                  f"pending={pending_total:>5}  rate={sps:.1f} states/sec  "
                  f"elapsed={elapsed:.0f}s  ETA={(args.target_states - finalized_states) / max(1, sps):.0f}s")
            last_log = now

        if finalized_states - last_save >= args.save_every:
            _save_buffer(args.output, out_states, out_policies, out_values,
                         out_actions, out_outcomes)
            last_save = finalized_states

    # Final save
    out_states = out_states[: args.target_states]
    out_policies = out_policies[: args.target_states]
    out_values = out_values[: args.target_states]
    out_actions = out_actions[: args.target_states]
    out_outcomes = out_outcomes[: args.target_states]
    _save_buffer(args.output, out_states, out_policies, out_values,
                 out_actions, out_outcomes)

    elapsed = time.time() - t_start
    print(f"[Gen] Done. {len(out_states):,} states in {elapsed:.0f}s "
          f"({len(out_states) / elapsed:.1f} states/sec).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
