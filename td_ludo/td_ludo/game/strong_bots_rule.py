"""Rule-based bot variants — diversifies decision rules beyond
expectimax/MCTS. Not necessarily stronger than expectimax, but
qualitatively different — the trained model can't pattern-match
"the expectimax decision rule" against these.

Bots:

  MaxCaptureBot
    Always captures if any legal move captures. Falls back to longest-
    distance move otherwise. Tests if "pure aggression" beats Expert
    (it usually doesn't, because it sometimes makes losing trades).

  TwoStackBot
    Prefers moves that create or preserve 2-token stacks on safe
    squares (blockades). Falls back to forward movement. Distinctive
    style — blockade-heavy play that the scripted bots don't do.

  HomeRushBot
    Always advances the most-advanced own token. Single-token strategy
    — race one token home at a time. Predictable and exploitable, but
    tests whether trained model can capitalize.

  StackHomeRushBot
    Hybrid: blockade when possible, otherwise rush the leader home.
    More varied than the pure versions.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

import td_ludo_cpp as ludo_cpp

from td_ludo.game.strong_bots import (
    _BASE_POS, _HOME_POS, _absolute_pos, _is_safe,
)


def _move_captures(state, action: int) -> bool:
    """True if `action` (token index) captures at least one opp token."""
    try:
        after = ludo_cpp.apply_move(state, int(action))
    except Exception:
        return False
    cp = int(state.current_player)
    # Count opp tokens at base before vs after; capture sends to base.
    before_base = sum(
        1 for p in range(4) if p != cp and state.active_players[p]
        for t in range(4) if int(state.player_positions[p][t]) == _BASE_POS
    )
    after_base = sum(
        1 for p in range(4) if p != cp and after.active_players[p]
        for t in range(4) if int(after.player_positions[p][t]) == _BASE_POS
    )
    return after_base > before_base


def _own_two_stacks(state, player: int) -> int:
    """Count own positions where ≥2 own tokens stack on a safe square."""
    positions = np.asarray(state.player_positions, dtype=np.int8)
    pos_counts = {}
    for t in range(4):
        p = int(positions[player][t])
        if p == _BASE_POS or p == _HOME_POS or p > 50:
            continue
        if not _is_safe(player, p):
            continue
        pos_counts[p] = pos_counts.get(p, 0) + 1
    return sum(1 for c in pos_counts.values() if c >= 2)


def _most_advanced_token(state, player: int) -> int:
    """Return token index of the most-advanced own token (highest pos,
    HOME counts as 99). Ties broken by lowest index."""
    positions = np.asarray(state.player_positions, dtype=np.int8)
    best = -1
    best_pos = -2  # base = -1, so -2 ensures any real pos wins
    for t in range(4):
        p = int(positions[player][t])
        if p > best_pos:
            best_pos = p
            best = t
    return best


def _move_target_pos(state, action: int, player: int) -> int:
    """Where token `action` would end up after the move (relative pos)."""
    try:
        after = ludo_cpp.apply_move(state, int(action))
        return int(after.player_positions[player][int(action)])
    except Exception:
        return _BASE_POS


# ─── MaxCaptureBot ────────────────────────────────────────────────────────


class MaxCaptureBot:
    """Captures whenever possible. Falls back to longest-distance move."""

    NAME = "MaxCapture"

    def __init__(self, player_id: Optional[int] = None):
        self.player_id = player_id

    def select_move(self, state, legal_moves: Sequence[int]) -> int:
        if not legal_moves:
            return -1
        # Find any capturing move
        for a in legal_moves:
            if _move_captures(state, int(a)):
                return int(a)
        # No captures — pick move that advances the most-distant token
        me = self.player_id if self.player_id is not None else int(state.current_player)
        best, best_advance = int(legal_moves[0]), -float("inf")
        for a in legal_moves:
            cur = int(state.player_positions[me][int(a)])
            new = _move_target_pos(state, int(a), me)
            advance = (new if new != _HOME_POS else 56) - (cur if cur != _BASE_POS else -1)
            if advance > best_advance:
                best_advance = advance
                best = int(a)
        return best


# ─── TwoStackBot ──────────────────────────────────────────────────────────


class TwoStackBot:
    """Prefers moves that create or maintain a 2-token stack on a safe
    square (i.e., a blockade)."""

    NAME = "TwoStack"

    def __init__(self, player_id: Optional[int] = None):
        self.player_id = player_id

    def select_move(self, state, legal_moves: Sequence[int]) -> int:
        if not legal_moves:
            return -1
        me = self.player_id if self.player_id is not None else int(state.current_player)
        # Pick the move that maximizes own_two_stacks AFTER applying it,
        # breaking ties by total forward progress.
        best, best_score = int(legal_moves[0]), (-1, -float("inf"))
        for a in legal_moves:
            try:
                after = ludo_cpp.apply_move(state, int(a))
            except Exception:
                continue
            n_stacks = _own_two_stacks(after, me)
            advance = 0
            for t in range(4):
                cur = int(state.player_positions[me][t])
                new = int(after.player_positions[me][t])
                if new == _HOME_POS:
                    advance += 56 - (cur if cur != _BASE_POS else -1)
                elif new != _BASE_POS and cur != _BASE_POS:
                    advance += new - cur
                elif new != _BASE_POS and cur == _BASE_POS:
                    advance += 0  # spawning
            score = (n_stacks, advance)
            if score > best_score:
                best_score = score
                best = int(a)
        return best


# ─── HomeRushBot ──────────────────────────────────────────────────────────


class HomeRushBot:
    """Always advances the most-advanced own token if it can move;
    otherwise falls back to first legal."""

    NAME = "HomeRush"

    def __init__(self, player_id: Optional[int] = None):
        self.player_id = player_id

    def select_move(self, state, legal_moves: Sequence[int]) -> int:
        if not legal_moves:
            return -1
        me = self.player_id if self.player_id is not None else int(state.current_player)
        leader = _most_advanced_token(state, me)
        if leader in legal_moves:
            return int(leader)
        # Leader can't move (e.g., overshoot home); pick the next-most-advanced.
        positions = np.asarray(state.player_positions, dtype=np.int8)
        candidates = sorted(
            legal_moves,
            key=lambda t: -int(positions[me][int(t)]),  # highest pos first
        )
        return int(candidates[0])


# ─── StackHomeRushBot ─────────────────────────────────────────────────────


class StackHomeRushBot:
    """Hybrid: if a blockade-creating move exists, take it (TwoStack
    logic). Otherwise rush the most-advanced token (HomeRush logic)."""

    NAME = "StackHomeRush"

    def __init__(self, player_id: Optional[int] = None):
        self.player_id = player_id
        self._twostack = TwoStackBot(player_id)
        self._homerush = HomeRushBot(player_id)

    def select_move(self, state, legal_moves: Sequence[int]) -> int:
        if not legal_moves:
            return -1
        me = self.player_id if self.player_id is not None else int(state.current_player)
        # Check if any move creates a NEW stack (count goes up)
        cur_stacks = _own_two_stacks(state, me)
        for a in legal_moves:
            try:
                after = ludo_cpp.apply_move(state, int(a))
            except Exception:
                continue
            if _own_two_stacks(after, me) > cur_stacks:
                return int(a)
        # Fall back to home rush
        self._homerush.player_id = me
        return self._homerush.select_move(state, legal_moves)


# ─── Registry ─────────────────────────────────────────────────────────────


RULE_BOT_REGISTRY = {
    "MaxCapture":     MaxCaptureBot,
    "TwoStack":       TwoStackBot,
    "HomeRush":       HomeRushBot,
    "StackHomeRush":  StackHomeRushBot,
}
