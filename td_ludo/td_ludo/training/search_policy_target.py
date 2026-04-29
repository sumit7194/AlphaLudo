"""
Depth-1 expectimax search-during-training for V12.2 (Exp 24).

Used as auxiliary policy target alongside PPO: at a fraction of training
states we run depth-1 expectimax over (first action × next dice × second
action), pick the argmax-Q first action, and add a cross-entropy auxiliary
loss pushing the model toward that target.

Design choices (see journal Discussion 2026-04-29):
- Target shape: argmax-onehot with label smoothing (NOT softmax(Q/T)).
  Value-head outputs sit in ~[0.4, 0.7] so softmax over Q is near-uniform
  regardless of T. argmax matches AlphaZero's policy improvement form.
- Opponent assumption: opponent picks argmax(pi_model) on s' (NOT min).
  Bot-mix opponents aren't adversarially perfect; pi_model is the
  on-policy expectation and consistent with self-play.
- Leaf eval: V12.2 value head (win_prob), state encoded with
  current_player overridden to the search root. The encoder uses
  current_player to decide which channels are "own", so this gives
  win_prob from the root player's perspective regardless of whose turn
  the leaf state actually represents.

Cost: ~96 leaf forward passes per searched state (4 first × 6 dice × 4
second worst case), plus ~24 opp-policy queries. With
search_target_fraction=0.25 and BATCH_SIZE=512 in PROD, expect roughly
~12K leaf evaluations per simulator turn — single batched forward pass.

Triple-six rule: not modeled at this depth. Approximation acceptable —
triple-six only kicks in when consecutive_sixes_count is already ≥ 2,
which is rare. The bonus-turn rule (dice==6, home, cut) IS captured
exactly because it lives inside apply_move.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

import td_ludo_cpp as cpp


NUM_DICE_VALUES = 6
NUM_TOKENS = 4
NUM_PLAYERS = 4


@dataclass
class _LeafSpec:
    """One leaf to evaluate: (game_idx, first_action_idx, dice_value,
    leaf_state_to_encode, root_player_for_perspective)."""
    game_idx: int
    first_idx: int   # index into game's legal_first_actions list
    dice: int
    leaf_state: object  # td_ludo_cpp.GameState
    root_player: int


@dataclass
class _OppQuery:
    """One opponent-policy query at s' (post-first-move, pre-second-move):
    we forward pi_model on s' from opp's perspective and use argmax."""
    game_idx: int
    first_idx: int
    dice: int
    s_prime: object  # td_ludo_cpp.GameState with dice set
    legal_seconds: List[int]
    root_player: int


def _encode_with_perspective(state, root_player: int) -> np.ndarray:
    """Encode `state` as if `root_player` is the current player.

    The V11 encoder uses state.current_player to decide which channels are
    "own" vs opponent. We temporarily override to root_player so the
    win_prob output corresponds to P(root_player wins).
    """
    saved = state.current_player
    state.current_player = root_player
    enc = cpp.encode_state_v11(state).copy()
    state.current_player = saved
    return enc


def _make_legal_mask(legal_moves: List[int]) -> np.ndarray:
    mask = np.zeros(NUM_TOKENS, dtype=np.float32)
    for m in legal_moves:
        mask[m] = 1.0
    return mask


def compute_pi_search_batch(
    games: List[object],
    root_players: List[int],
    model,
    device,
    label_smoothing: float = 0.1,
):
    """Compute depth-1 expectimax search policy targets for a batch of games.

    Args:
        games: list of N GameState objects, each at a decision point
            (current_dice_roll already set, root player is current_player).
        root_players: list of N ints. For each game, the player from whose
            perspective to compute Q. Typically equals games[i].current_player.
        model: V12.2 model exposing forward(x, mask) → (policy, win_prob, moves)
            and forward_policy_only(x, mask) → policy_logits.
        device: torch device for forward passes.
        label_smoothing: smoothing applied to one-hot pi_search target. The
            argmax action gets weight (1 - smoothing); remaining smoothing is
            spread uniformly over OTHER LEGAL actions.

    Returns:
        pi_search: torch.FloatTensor (N, 4) on `device`. One-hot smoothed at
            argmax(Q) per game. Illegal action slots are 0; smoothing only
            distributes over legal alternatives.
        diagnostics: dict with keys:
            - leaf_count (int)
            - opp_query_count (int)
            - top_actions (list of int): argmax first action per game
            - q_values (list of dict): per-game {action: Q}
            - games_with_immediate_win (int)
    """
    N = len(games)
    assert len(root_players) == N

    pi_search = torch.zeros(N, NUM_TOKENS, dtype=torch.float32, device=device)

    legal_first_per_game: List[List[int]] = [
        list(cpp.get_legal_moves(g)) for g in games
    ]
    # Q values per (game, first_idx). We'll fill in over the loop.
    q_per_game: List[List[float]] = [
        [0.0 for _ in legals] for legals in legal_first_per_game
    ]

    leaf_specs: List[_LeafSpec] = []
    opp_queries: List[_OppQuery] = []

    games_with_immediate_win = 0

    # Per-(game, first_idx, dice) running aggregates of leaf values; once
    # all leaves for that (g, fi, d) are evaluated we collapse via
    # max (root's bonus turn) or take-the-opp-action (opp turn).
    # Use a dict keyed by (gi, fi, d) → list of (leaf_value_tensor_idx,
    # owner_is_root_bool). Filled in below after we know leaf_count.
    bucket: dict = {}  # key (gi, fi, d) → {'is_root_turn': bool, 'leaf_idxs': List[int]}

    # For terminal-on-first-move games, we record value directly.
    terminal_value: dict = {}  # key (gi, fi) → float in [0,1]

    # ── Phase 1: enumerate all (game, first_action, dice, second_action)
    # quadruples; collect leaf states; collect opp-policy queries.
    for gi, (g, legal_firsts, root_p) in enumerate(zip(
        games, legal_first_per_game, root_players,
    )):
        if not legal_firsts:
            # No legal first action — model would skip turn anyway; pi_search
            # stays all-zero (caller skips zero-row losses).
            continue

        for fi, a in enumerate(legal_firsts):
            s_prime = cpp.apply_move(g, a)

            winner = cpp.get_winner(s_prime)
            if winner >= 0:
                terminal_value[(gi, fi)] = 1.0 if winner == root_p else 0.0
                games_with_immediate_win += 1
                continue

            for d in range(1, NUM_DICE_VALUES + 1):
                # We need an INDEPENDENT s' for each dice value because
                # opp_queries store references to it and later iterations
                # would otherwise overwrite the dice. pybind11 GameState
                # has no .clone(); re-deriving via apply_move(g, a) is the
                # cheapest path (pure-C++ memcpy, no Python allocation).
                s_prime_d = cpp.apply_move(g, a)
                s_prime_d.current_dice_roll = d

                next_player = s_prime_d.current_player
                legal_seconds = list(cpp.get_legal_moves(s_prime_d))

                bucket_key = (gi, fi, d)

                if not legal_seconds:
                    # No legal move with this dice — treat the resulting
                    # "pass turn" state as the leaf for value evaluation
                    # from root's perspective. For opp turn this means opp
                    # passes; for root's bonus turn this is rare but
                    # possible (e.g., all tokens home except one stuck).
                    leaf_idx = len(leaf_specs)
                    leaf_specs.append(_LeafSpec(
                        game_idx=gi, first_idx=fi, dice=d,
                        leaf_state=s_prime_d, root_player=root_p,
                    ))
                    bucket[bucket_key] = {
                        'is_root_turn': (next_player == root_p),
                        'leaf_idxs': [leaf_idx],
                    }
                    continue

                if next_player == root_p:
                    # Bonus turn — enumerate own second actions, max over them.
                    leaf_idxs = []
                    for a2 in legal_seconds:
                        leaf = cpp.apply_move(s_prime_d, a2)
                        leaf_idxs.append(len(leaf_specs))
                        leaf_specs.append(_LeafSpec(
                            game_idx=gi, first_idx=fi, dice=d,
                            leaf_state=leaf, root_player=root_p,
                        ))
                    bucket[bucket_key] = {
                        'is_root_turn': True, 'leaf_idxs': leaf_idxs,
                    }
                else:
                    # Opp turn — pick argmax(pi_model) on s_prime_d from
                    # opp perspective, then evaluate the resulting leaf.
                    if len(legal_seconds) == 1:
                        leaf = cpp.apply_move(s_prime_d, legal_seconds[0])
                        leaf_idx = len(leaf_specs)
                        leaf_specs.append(_LeafSpec(
                            game_idx=gi, first_idx=fi, dice=d,
                            leaf_state=leaf, root_player=root_p,
                        ))
                        bucket[bucket_key] = {
                            'is_root_turn': False, 'leaf_idxs': [leaf_idx],
                        }
                    else:
                        # Defer: schedule opp policy query, fill leaves
                        # after we know which action opp picked.
                        opp_queries.append(_OppQuery(
                            game_idx=gi, first_idx=fi, dice=d,
                            s_prime=s_prime_d, legal_seconds=legal_seconds,
                            root_player=root_p,
                        ))
                        bucket[bucket_key] = {
                            'is_root_turn': False, 'leaf_idxs': None,
                        }

    # ── Phase 2: resolve opponent-policy queries with one batched forward.
    if opp_queries:
        opp_states = np.stack([
            cpp.encode_state_v11(q.s_prime) for q in opp_queries
        ])
        opp_masks = np.stack([
            _make_legal_mask(q.legal_seconds) for q in opp_queries
        ])
        opp_states_t = torch.from_numpy(opp_states).to(device, dtype=torch.float32)
        opp_masks_t = torch.from_numpy(opp_masks).to(device, dtype=torch.float32)
        with torch.no_grad():
            opp_logits = model.forward_policy_only(opp_states_t, opp_masks_t)
            opp_actions = opp_logits.argmax(dim=1).cpu().numpy()

        for k, q in enumerate(opp_queries):
            chosen = int(opp_actions[k])
            if chosen not in q.legal_seconds:
                # Defensive — illegal masking should prevent this. Fallback.
                chosen = q.legal_seconds[0]
            leaf = cpp.apply_move(q.s_prime, chosen)
            leaf_idx = len(leaf_specs)
            leaf_specs.append(_LeafSpec(
                game_idx=q.game_idx, first_idx=q.first_idx, dice=q.dice,
                leaf_state=leaf, root_player=q.root_player,
            ))
            bucket[(q.game_idx, q.first_idx, q.dice)]['leaf_idxs'] = [leaf_idx]

    # ── Phase 3: one batched forward pass over all leaves for win_prob.
    leaf_values: np.ndarray
    if leaf_specs:
        leaf_states = np.stack([
            _encode_with_perspective(spec.leaf_state, spec.root_player)
            for spec in leaf_specs
        ])
        leaf_states_t = torch.from_numpy(leaf_states).to(device, dtype=torch.float32)
        # No legal_mask needed for value-only forward; pass None.
        with torch.no_grad():
            _, win_prob, _ = model(leaf_states_t, None)
        leaf_values = win_prob.detach().cpu().numpy().astype(np.float64)
    else:
        leaf_values = np.zeros(0, dtype=np.float64)

    # ── Phase 4: aggregate into Q values.
    for (gi, fi, d), info in bucket.items():
        leaf_idxs = info['leaf_idxs']
        is_root_turn = info['is_root_turn']
        vals = [leaf_values[i] for i in leaf_idxs]
        if is_root_turn:
            v_d = max(vals)  # root picks best second action
        else:
            v_d = vals[0]    # opp's chosen action (single leaf)
        # Accumulate mean over dice: Q(a) = (1/6) * sum_d v(d).
        q_per_game[gi][fi] += v_d / NUM_DICE_VALUES

    # Add terminal-first-move values: contribute uniformly across dice (no
    # post-action enumeration needed — game ended on the first action).
    for (gi, fi), val in terminal_value.items():
        q_per_game[gi][fi] = val

    # ── Phase 5: argmax → smoothed one-hot per game.
    top_actions: List[int] = []
    q_diag: List[dict] = []
    for gi, (legal_firsts, q_list) in enumerate(zip(legal_first_per_game, q_per_game)):
        if not legal_firsts:
            top_actions.append(-1)
            q_diag.append({})
            continue

        best_local = int(np.argmax(q_list))
        best_action = legal_firsts[best_local]
        top_actions.append(best_action)
        q_diag.append({a: float(q_list[i]) for i, a in enumerate(legal_firsts)})

        # Build smoothed target only on legal positions.
        n_legal = len(legal_firsts)
        if n_legal == 1:
            pi_search[gi, best_action] = 1.0
        else:
            spread = label_smoothing / (n_legal - 1) if label_smoothing > 0 else 0.0
            for a in legal_firsts:
                pi_search[gi, a] = spread
            pi_search[gi, best_action] = 1.0 - label_smoothing

    diagnostics = {
        'leaf_count': len(leaf_specs),
        'opp_query_count': len(opp_queries),
        'top_actions': top_actions,
        'q_values': q_diag,
        'games_with_immediate_win': games_with_immediate_win,
    }
    return pi_search, diagnostics
