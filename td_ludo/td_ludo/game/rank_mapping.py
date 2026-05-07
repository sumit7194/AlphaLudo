"""Canonical rank mapping for V13.5's symmetric output head.

Concept
-------
V13.5 outputs 4 logits indexed by *canonical rank* of unique own-token
positions, not by token-ID. This makes the output head permutation-
equivariant under token-ID permutations: swap any two own token-IDs in
the input, and the canonical-rank output is unchanged.

Canonical rank ordering
-----------------------
Sort the unique current positions of the 4 own tokens in DESCENDING
order (most advanced first, with home positions sharing the bottom
rank slot):

    rank 0  → most-advanced unique position
    rank 1  → second-most-advanced unique position
    ...
    rank R-1 → least-advanced unique position (often "home")

`R` is the number of unique own-token current positions (1 ≤ R ≤ 4).
If R < 4, the unused rank slots are masked illegal in the policy.

Token-ID action recovery
------------------------
At play time, the engine receives a rank prediction. To execute the
move, it picks ANY token currently at the chosen unique position
(tokens at the same position are physically interchangeable in Ludo —
moving any of them produces the same next-state).

Permutation invariance
----------------------
The mapping `(state) → (rank → unique position)` depends only on the
multiset of current positions of own tokens, not on which token-ID is
at which position. So:
    rank_to_position(state) == rank_to_position(token_id_permuted(state))

This is verified by unit tests.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


# Position semantics (engine convention):
#   -1   = at home
#    0+  = on the board path (larger = more advanced)
HOME_POS = -1
MAX_RANK_SLOTS = 4   # Ludo: 4 own tokens => at most 4 unique positions


def _rank_key(pos: int) -> int:
    """Canonical rank key — larger = higher rank (more advanced).

    Position -1 (home) sorts to the bottom. Board positions sort by
    raw integer order (more advanced = higher).
    """
    if pos == HOME_POS:
        return -1_000_000  # well below any board position
    return pos


def state_to_rank_mapping(player_positions_row: np.ndarray) -> Tuple[List[int], List[List[int]]]:
    """Compute the canonical rank mapping for one player's tokens.

    Parameters
    ----------
    player_positions_row : (4,) int array
        Current positions of the 4 tokens of one player. -1 = at home,
        0+ = board path.

    Returns
    -------
    rank_positions : List[int]
        unique positions in canonical rank order (length R, 1 ≤ R ≤ 4).
        rank_positions[k] is the position assigned to rank k.
    rank_token_ids : List[List[int]]
        rank_token_ids[k] is the list of token-IDs (0..3) currently at
        rank_positions[k]. Token-IDs within a rank are sorted ascending
        for a deterministic tiebreak when picking which token to move.
    """
    pp = np.asarray(player_positions_row).flatten().tolist()
    assert len(pp) == 4, f"expected 4 token positions, got {len(pp)}"

    # Group token-IDs by current position
    pos_to_tokens: dict[int, List[int]] = {}
    for tok_id, pos in enumerate(pp):
        pos_to_tokens.setdefault(pos, []).append(tok_id)

    # Sort unique positions by rank key (descending → most-advanced first)
    unique_positions = sorted(pos_to_tokens.keys(), key=_rank_key, reverse=True)

    rank_positions: List[int] = list(unique_positions)
    rank_token_ids: List[List[int]] = [sorted(pos_to_tokens[p]) for p in unique_positions]
    return rank_positions, rank_token_ids


def aggregate_token_policy_to_ranks(
    token_probs: Sequence[float],
    rank_token_ids: Sequence[Sequence[int]],
) -> np.ndarray:
    """Aggregate a per-token-ID policy into a per-rank policy by summing
    probabilities of all token-IDs in the same rank group.

    Parameters
    ----------
    token_probs : (4,) sequence of floats
        Per-token-ID probabilities (sum should be ≈ 1; not enforced).
    rank_token_ids : output of state_to_rank_mapping (rank → token-IDs).

    Returns
    -------
    rank_probs : (MAX_RANK_SLOTS,) np.ndarray, dtype float32
        rank_probs[k] = sum of token_probs[t] for t in rank_token_ids[k].
        Slots k >= R (R = number of unique positions) are 0.0.
    """
    out = np.zeros(MAX_RANK_SLOTS, dtype=np.float32)
    for k, tokens in enumerate(rank_token_ids):
        if k >= MAX_RANK_SLOTS:
            break
        out[k] = float(sum(token_probs[t] for t in tokens))
    return out


def legal_mask_per_rank(
    legal_token_ids: Sequence[int],
    rank_token_ids: Sequence[Sequence[int]],
) -> np.ndarray:
    """Build a 4-element legal mask over canonical ranks.

    A rank is legal iff at least one token-ID in that rank-group has a
    legal move. (Two tokens at the same position have identical legal
    moves; the engine can pick either.)

    Returns
    -------
    mask : (MAX_RANK_SLOTS,) np.ndarray, dtype float32
        1.0 where the rank has at least one legal token-ID, 0.0 elsewhere.
    """
    legal_set = set(int(t) for t in legal_token_ids)
    out = np.zeros(MAX_RANK_SLOTS, dtype=np.float32)
    for k, tokens in enumerate(rank_token_ids):
        if k >= MAX_RANK_SLOTS:
            break
        if any(t in legal_set for t in tokens):
            out[k] = 1.0
    return out


def rank_to_token_id(
    rank: int,
    legal_token_ids: Sequence[int],
    rank_token_ids: Sequence[Sequence[int]],
) -> int:
    """Map a chosen canonical rank back to a concrete legal token-ID.

    Tokens at the same position are interchangeable; we pick the lowest
    token-ID for determinism. If the chosen rank has no legal token-ID
    (shouldn't happen when the legal_mask was respected), we fall back
    to any legal token-ID.
    """
    legal_set = set(int(t) for t in legal_token_ids)
    if 0 <= rank < len(rank_token_ids):
        candidates = [t for t in rank_token_ids[rank] if t in legal_set]
        if candidates:
            return min(candidates)
    # Safety fallback — should not be reached when callers respect mask.
    return int(legal_token_ids[0]) if len(legal_token_ids) else -1


def permute_own_tokens(
    state, perm: Sequence[int]
):
    """Return a NEW state where the current player's 4 token IDs are permuted.

    Specifically, after the permutation:
        new_pp[cp][i] = old_pp[cp][perm[i]]

    Used at distillation time to feed V13.2 a token-ID-permuted input
    so that we can sample from the orbit of V13.2's policy under
    permutations of own token-IDs.

    Note: only OWN tokens (current_player) are permuted. Opponent
    tokens are left untouched (V13.2's per-token output dimension is
    over OWN tokens, so opp permutation isn't needed for our purpose).
    """
    import td_ludo_cpp as _cpp
    # Construct a copy by going through the state's array. The C++
    # GameState exposes player_positions as a writable numpy array.
    new_state = _cpp.create_initial_state_2p()
    # Bulk-copy fields
    new_state.current_player = int(state.current_player)
    new_state.current_dice_roll = int(state.current_dice_roll)
    new_state.player_positions[:] = state.player_positions[:]
    new_state.active_players[:] = state.active_players[:]
    if hasattr(state, "scores"):
        new_state.scores[:] = state.scores[:]
    if hasattr(state, "idle_counter"):
        new_state.idle_counter[:] = state.idle_counter[:]
    if hasattr(state, "streak"):
        new_state.streak[:] = state.streak[:]
    if hasattr(state, "last_moved_token"):
        new_state.last_moved_token[:] = state.last_moved_token[:]
    if hasattr(state, "is_terminal"):
        new_state.is_terminal = state.is_terminal
    if hasattr(state, "board"):
        new_state.board[:] = state.board[:]

    cp = int(state.current_player)
    perm_arr = np.asarray(perm, dtype=int)
    assert perm_arr.shape == (4,), f"perm must be length 4, got {perm_arr.shape}"
    assert sorted(perm_arr.tolist()) == [0, 1, 2, 3], f"perm must be a permutation of {{0,1,2,3}}, got {perm_arr.tolist()}"

    # Permute current player's row only
    orig_row = state.player_positions[cp].copy()
    new_state.player_positions[cp] = orig_row[perm_arr]
    return new_state


__all__ = [
    "HOME_POS",
    "MAX_RANK_SLOTS",
    "state_to_rank_mapping",
    "aggregate_token_policy_to_ranks",
    "legal_mask_per_rank",
    "rank_to_token_id",
    "permute_own_tokens",
]
