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
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import td_ludo_cpp as ludo_cpp
from td_ludo.game.encoder_v18_production import (
    encode_state_v18_production, V18_PROD_CHANNELS,
)
from td_ludo.models.v13_5_production import V135ProductionAdapter


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


def load_v135(path, device, num_res_blocks, num_channels):
    """Load V13.5 production teacher (V18 production encoder, 21ch).
    V135ProductionAdapter is token-id-indexed (not rank-indexed) at I/O
    boundary, so the search aggregator can use the policy directly.
    Value head output is sigmoid win_prob (same shape as V13.2)."""
    print(f"[Gen] Loading V13.5 teacher from {path}...")
    model = V135ProductionAdapter(
        num_res_blocks=num_res_blocks,
        num_channels=num_channels,
    )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[Gen] Loaded V13.5 ({sum(p.numel() for p in model.parameters()):,} params)")
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


def _copy_state(state):
    """Return a fresh deep copy of a `GameState`.

    Required because `apply_move` returns a new state, but mutating fields
    on it (e.g. `current_dice_roll`, `current_player`) and then storing
    that reference in multiple places aliases all those references to the
    SAME mutable object — a fatal bug for the search expansion below.

    Implementation: construct a new GameState and copy each exposed field
    using numpy for the array-typed properties (which the pybind binding
    exposes as `np.ndarray` views).
    """
    s = ludo_cpp.GameState()
    # Array fields — copy via np.ndarray to detach from the source's buffer.
    s.player_positions = np.array(state.player_positions, dtype=np.int8).copy()
    s.scores = np.array(state.scores, dtype=np.int8).copy()
    s.active_players = np.array(state.active_players, dtype=bool).copy()
    s.idle_counter = np.array(state.idle_counter, dtype=np.int8).copy()
    s.last_moved_token = np.array(state.last_moved_token, dtype=np.int8).copy()
    s.streak = np.array(state.streak, dtype=np.int8).copy()
    # Scalar fields
    s.current_player = int(state.current_player)
    s.current_dice_roll = int(state.current_dice_roll)
    s.is_terminal = bool(state.is_terminal)
    return s


def _expand_two_ply_leaves(game):
    """Enumerate all 2-ply leaf states from `game`.

    Returns:
      leaves:   list of GameState, one per leaf (own action × dice × opp action)
      meta:     list of dicts {own_action, dice, opp_action, terminal_winner,
                               next_player_is_root}
                next_player_is_root: True iff after `own_a` the current_player
                is still the root_player (= a 6-roll bonus or scoring move that
                kept the turn). The Q aggregator branches on this flag because
                "opp picks min" is wrong when it's actually still us choosing.
      legal_own: list of int — own legal actions at the root
      structure_per_action: dict[own_action] -> dict[dice] -> list of
                (chosen_action, leaf_index). chosen_action == -1 means the
                player at that node had no legal moves (turn-pass).

    Bug-fix history:
      v2 (2026-05-06): _copy_state used everywhere mutation occurs, so each
        leaf is a distinct GameState object. Plus `next_player_is_root` flag
        plumbed into meta so the aggregator handles bonus-turn moves correctly.
    """
    legal_own = list(ludo_cpp.get_legal_moves(game))
    leaves = []
    meta = []
    structure_per_action = {a: {} for a in legal_own}

    root_player = int(game.current_player)

    for own_a in legal_own:
        s_after_own = ludo_cpp.apply_move(game, int(own_a))
        # Whether the turn stayed with the root player after own_a.
        # In Ludo this happens on dice == 6 OR when the move scored a token.
        # The aggregator MUST use max-over-actions (not min) for these,
        # because the actor at the next decision is still root_player.
        next_player_is_root = (
            (not s_after_own.is_terminal)
            and int(s_after_own.current_player) == root_player
        )

        for dice in DICE_VALUES:
            if s_after_own.is_terminal:
                # Game ended on our move. Single leaf, all dice values share it.
                # Take a fresh copy so mutations elsewhere can't corrupt it.
                leaf = _copy_state(s_after_own)
                terminal_winner = int(ludo_cpp.get_winner(leaf))
                meta.append({
                    "own_action": int(own_a),
                    "dice": dice,
                    "opp_action": -1,
                    "terminal_winner": terminal_winner,
                    "next_player_is_root": False,  # game over, no decision
                })
                leaves.append(leaf)
                structure_per_action[own_a].setdefault(dice, []).append(
                    (-1, len(leaves) - 1)
                )
                continue

            # Build a FRESH copy for this dice iteration so mutations to
            # current_dice_roll don't leak across dice loop iterations.
            s_for_opp = _copy_state(s_after_own)
            s_for_opp.current_dice_roll = dice
            opp_legals = list(ludo_cpp.get_legal_moves(s_for_opp))

            if not opp_legals:
                # No legal moves at this dice — pass turn back. We need the
                # post-pass state as the leaf. CRITICAL: this leaf must be a
                # standalone GameState (no aliasing to s_after_own), or every
                # subsequent iteration's leaf references will see whatever this
                # one mutated to. We use _copy_state again to be safe.
                leaf = _copy_state(s_for_opp)
                cp = int(leaf.current_player)
                nxt = (cp + 1) % 4
                while not leaf.active_players[nxt]:
                    nxt = (nxt + 1) % 4
                leaf.current_player = nxt
                leaf.current_dice_roll = 0
                meta.append({
                    "own_action": int(own_a),
                    "dice": dice,
                    "opp_action": -1,
                    "terminal_winner": -1,
                    "next_player_is_root": next_player_is_root,
                })
                leaves.append(leaf)
                structure_per_action[own_a].setdefault(dice, []).append(
                    (-1, len(leaves) - 1)
                )
                continue

            # Opp (or root, if bonus turn) has legal moves — try each.
            for opp_a in opp_legals:
                s_leaf = ludo_cpp.apply_move(s_for_opp, int(opp_a))
                # apply_move returns a fresh state, so no copy needed for
                # storage. (We're not mutating s_leaf after this point.)
                terminal_winner = (
                    int(ludo_cpp.get_winner(s_leaf)) if s_leaf.is_terminal else -1
                )
                meta.append({
                    "own_action": int(own_a),
                    "dice": dice,
                    "opp_action": int(opp_a),
                    "terminal_winner": terminal_winner,
                    "next_player_is_root": next_player_is_root,
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

    # `meta` carries `next_player_is_root`: True when, after own_a, the
    # next decision-maker is the root player (6-roll bonus or score). The
    # aggregation must use MAX over actions (we pick best for ourselves)
    # instead of MIN (which is correct only when opp is choosing).
    # We look up the flag once per (own_a, dice) bucket — every leaf in
    # the same bucket shares the same flag by construction.
    Q = {}
    for own_a in legal_own:
        dice_to_pairs = structure[own_a]
        sum_over_dice = 0.0
        for dice in DICE_VALUES:
            pairs = dice_to_pairs.get(dice, [])
            if not pairs:
                continue
            # Find flag for this bucket using the first leaf's meta entry.
            _, first_leaf_idx = pairs[0]
            next_is_root = bool(meta[first_leaf_idx]["next_player_is_root"])

            if next_is_root:
                # Bonus turn: WE are the actor at the next decision.
                # Pick max V_root over our second-move options.
                best_v_root = max(leaf_v_root[idx] for (_, idx) in pairs)
            else:
                # Standard: opp picks max-for-them = min V_root for us.
                best_v_root = min(leaf_v_root[idx] for (_, idx) in pairs)
            sum_over_dice += best_v_root
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
    """Atomically save the buffer with one-version-back failsafe.

    Three-step write:
      1. Write to <out_path>.tmp.npz
      2. If <out_path> exists, mv <out_path> -> <out_path>.prev.npz
         (overwriting any previous .prev backup)
      3. mv <out_path>.tmp.npz -> <out_path>
    Result: even if step 1 is interrupted (truncated tmp file), the
    previous valid <out_path> is intact and the corrupt tmp can be
    deleted. Even if a save WAS already complete, the immediate prior
    save is kept as <out_path>.prev.npz, giving us a 2-version rolling
    history. This prevents the failure mode where a network hiccup
    during the final write corrupts the entire file (which destroyed
    ~6 hours of work on 2026-05-06 morning).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp.npz")
    prev_path = out_path.with_name(out_path.stem + ".prev.npz")

    # 1. Write to temp first (this is the only step that can corrupt anything,
    #    and it's a separate file so it can't take down a valid out_path).
    np.savez_compressed(
        tmp_path,
        states=np.array(states, dtype=np.float32),
        search_policies=np.array(search_policies, dtype=np.float32),
        search_values=np.array(search_values, dtype=np.float32),
        search_actions=np.array(search_actions, dtype=np.int8),
        outcomes=np.array(outcomes, dtype=np.float32),
    )

    # 2. Rotate: out_path -> prev_path (if out_path exists).
    if out_path.exists():
        # os.replace is atomic on POSIX, so prev_path always reflects a valid
        # previously-good buffer (or doesn't exist if we never had one).
        os.replace(out_path, prev_path)

    # 3. Atomic rename tmp -> out_path. After this point the new buffer is
    #    durable.
    os.replace(tmp_path, out_path)
    print(f"[Gen] Saved buffer → {out_path} ({len(states):,} states) "
          f"[prev kept at {prev_path.name}]")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = pick_device(args.device)
    print(f"[Gen] Device: {device}")
    teacher = load_v135(args.teacher, device, args.num_res_blocks, args.num_channels)

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
                [encode_state_v18_production(s) for s in all_leaves_for_batch], axis=0
            )
            x = torch.from_numpy(encoded).to(device, dtype=torch.float32)
            # No mask matters for value — but model expects one. Permissive mask works.
            mask = torch.ones(len(all_leaves_for_batch), 4, device=device)
            with torch.no_grad():
                # V135ProductionAdapter returns 4-tuple
                # (policy, win_prob, moves, progress).
                out = teacher(x, mask)
                win_prob_batch = out[1]
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
            state_tensor = encode_state_v18_production(game)

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
            eta_sec = (args.target_states - finalized_states) / max(1, sps)
            print(f"  [{finalized_states:>7}/{args.target_states:,}]  "
                  f"pending={pending_total:>5}  rate={sps:.1f} states/sec  "
                  f"elapsed={elapsed:.0f}s  ETA={eta_sec:.0f}s", flush=True)
            last_log = now
            # Write stats JSON for dashboard
            try:
                stats_path = args.output + ".stats.json"
                with open(stats_path, "w") as f:
                    json.dump({
                        "stage": "gen", "phase": "running",
                        "finalized_states": int(finalized_states),
                        "target_states": int(args.target_states),
                        "fraction": float(finalized_states) / args.target_states,
                        "pending": int(pending_total),
                        "rate_states_per_sec": float(sps),
                        "elapsed_sec": float(elapsed),
                        "eta_sec": float(eta_sec),
                        "teacher": str(args.teacher),
                        "output": str(args.output),
                        "ts": int(now),
                    }, f)
            except Exception:
                pass

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
