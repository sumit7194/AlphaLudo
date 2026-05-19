"""Expectimax personality variants — Phase 1 of STRONG_BOTS_PLAN.md.

Subclasses the depth-1 expectimax lookahead structure of `ExpectimaxBot`
(from `strong_bots.py`) but swaps the scoring function. Same compute
cost (~50ms/decision, 46 g/s); fundamentally different decision
preferences.

Why subclass rather than parametrize the existing bot: keeps the
production `ExpectimaxBot` untouched (which may already be feeding
training/eval code paths), and makes each personality self-documenting
as a distinct class.

Variants
--------

  AggressiveExpectimaxBot
    "Always punch." Heavy bonus on captures (own opp_progress gain by
    sending opp tokens to base). Reduced fear of own exposure. Should
    trade tokens aggressively where the math is even-ish.

  DefensiveExpectimaxBot
    "Never expose." Heavy penalty on own unsafe tokens. Will refuse to
    advance into reach-of-opp even at the cost of progress. Stalls.

  RacingExpectimaxBot
    "Just run." Only own_progress matters; opponent ignored entirely.
    Closest to a "single-agent" planner — interesting because the model
    has to learn to capture it (since it'll never defend).

  MinimaxExpectimaxBot
    "Worst-case opp." Picks the move maximizing the score AFTER opp's
    best response — same as base ExpectimaxBot — but the score is
    `-opp_score` (minimax over opp's POV) rather than `+own_score`.
    Subtle difference; produces more cautious play when own and opp
    objectives are not perfectly anti-symmetric.

All four expose a `select_move(state, legal_moves) -> int` interface
matching the BOT_REGISTRY contract (player_id optional via __init__).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

import td_ludo_cpp as ludo_cpp

# Reuse the scoring + state-cloning helpers from the base module.
from td_ludo.game.strong_bots import (
    _absolute_pos,
    _is_safe,
    _token_progress_score,
    _exposure_penalty,
    _score_position as _base_score_position,
    _clone_state,
    _next_active_player,
    _BASE_POS,
    _HOME_POS,
)


# ─── Scoring-function variants ────────────────────────────────────────────


def _score_aggressive(player: int, state) -> float:
    """Aggressive: ×3 the opp's exposure (encourages putting opp under
    threat), and ×0.5 own exposure (less afraid of retaliation).

    Mechanically: own_progress + own_scored×60 − 0.5·own_exposure
                  − 1.0·opp_progress + 3·opp_exposure_estimate

    The "opp exposure estimate" approximates how much trouble opp's
    tokens are in — proxied by counting opp tokens that are 1-6 squares
    AHEAD of any of our tokens on the main track. The own-exposure
    function happens to be symmetric (just swap player↔opp), so we can
    reuse it.
    """
    positions = np.asarray(state.player_positions, dtype=np.int8)
    active = np.asarray(state.active_players, dtype=bool)

    own_progress = _token_progress_score(player, positions)
    own_scored = int(state.scores[player]) * 60
    own_exposure = _exposure_penalty(player, positions, active)

    opp_progress = 0.0
    opp_exposure_total = 0.0
    for opp in range(4):
        if opp == player or not active[opp]:
            continue
        opp_progress += _token_progress_score(opp, positions)
        opp_progress += int(state.scores[opp]) * 60
        # opp's exposure FROM OUR perspective: opp tokens we could capture
        opp_exposure_total += _exposure_penalty(opp, positions, active)

    return (
        own_progress + own_scored - 0.5 * own_exposure
        - opp_progress + 3.0 * opp_exposure_total
    )


def _score_defensive(player: int, state) -> float:
    """Defensive: ×3 own exposure penalty (refuses to advance into
    threats), opp_progress weighted ×2 (caring more about opp's
    advancement than own).

    Net effect: prefers staying-put or moving safely over racing or
    capturing.
    """
    positions = np.asarray(state.player_positions, dtype=np.int8)
    active = np.asarray(state.active_players, dtype=bool)

    own_progress = _token_progress_score(player, positions)
    own_scored = int(state.scores[player]) * 60
    own_exposure = _exposure_penalty(player, positions, active)

    opp_progress = 0.0
    for opp in range(4):
        if opp == player or not active[opp]:
            continue
        opp_progress += _token_progress_score(opp, positions)
        opp_progress += int(state.scores[opp]) * 60

    return (
        own_progress + own_scored - 3.0 * own_exposure
        - 2.0 * opp_progress
    )


def _score_racing(player: int, state) -> float:
    """Racing: only own progress + own scored. Opp entirely ignored.

    Acts like a single-agent path-planner — won't capture, won't defend,
    just runs. The trained model has to figure out it can freely
    capture this bot's tokens.
    """
    positions = np.asarray(state.player_positions, dtype=np.int8)
    own_progress = _token_progress_score(player, positions)
    own_scored = int(state.scores[player]) * 60
    return own_progress + own_scored


def _score_blockade(player: int, state) -> float:
    """Blockade: bonus for own 2+ token stacks on safe squares
    (blockades), strong penalty for breaking existing stacks.

    Score = base + 25·(own_two_stacks) + own_progress − own_exposure
            − 1.5·opp_progress

    Style: tries to lock down chokepoints with paired tokens, slow but
    hard to capture. Different play texture from Aggressive/Defensive
    because blockades are about *position* not *count*.
    """
    positions = np.asarray(state.player_positions, dtype=np.int8)
    active = np.asarray(state.active_players, dtype=bool)

    own_progress = _token_progress_score(player, positions)
    own_scored = int(state.scores[player]) * 60
    own_exposure = _exposure_penalty(player, positions, active)

    # Count own 2+ token stacks on safe squares.
    pos_counts = {}
    for t in range(4):
        p = int(positions[player][t])
        if p == _BASE_POS or p == _HOME_POS or p > 50:
            continue
        if not _is_safe(player, p):
            continue
        pos_counts[p] = pos_counts.get(p, 0) + 1
    n_stacks = sum(1 for c in pos_counts.values() if c >= 2)

    opp_progress = 0.0
    for opp in range(4):
        if opp == player or not active[opp]:
            continue
        opp_progress += _token_progress_score(opp, positions)
        opp_progress += int(state.scores[opp]) * 60

    return (
        own_progress + own_scored + 25.0 * n_stacks
        - own_exposure - 1.5 * opp_progress
    )


def _score_minimax(player: int, state) -> float:
    """Minimax: prefer states where OPP's progress is minimized.

    Mathematically equivalent (up to constants) to maximizing own
    progress if zero-sum, but Ludo isn't zero-sum (both can lose tokens
    to base independently). So this differs subtly from base
    Expectimax: it'll prefer moves that hurt opp even at zero own gain,
    where the base bot would be indifferent.

    Score = own_scored×60 + own_progress − 3·opp_progress − 3·opp_scored×60
            − own_exposure
    """
    positions = np.asarray(state.player_positions, dtype=np.int8)
    active = np.asarray(state.active_players, dtype=bool)

    own_progress = _token_progress_score(player, positions)
    own_scored = int(state.scores[player]) * 60
    own_exposure = _exposure_penalty(player, positions, active)

    opp_progress = 0.0
    opp_scored = 0.0
    for opp in range(4):
        if opp == player or not active[opp]:
            continue
        opp_progress += _token_progress_score(opp, positions)
        opp_scored += int(state.scores[opp]) * 60

    return own_progress + own_scored - 3.0 * opp_progress - 3.0 * opp_scored - own_exposure


# ─── Generic depth-1 expectimax base ──────────────────────────────────────


class _BaseExpectimaxV2:
    """Depth-1 expectimax with pluggable scoring. Same lookahead
    structure as the original ExpectimaxBot; subclasses inject the
    `_score(player, state) -> float` callable via class attr or
    constructor arg.

    Implementation note: I duplicate the lookahead loop from the
    original ExpectimaxBot rather than inheriting, so that swapping
    scoring is the ONLY mechanism that differs. Keeps benchmark
    comparisons clean — any WR delta vs the original is attributable to
    the scoring change.
    """

    # Subclasses override:
    SCORE_FN = staticmethod(_base_score_position)
    NAME = "BaseExpectimaxV2"

    def __init__(self, player_id: Optional[int] = None):
        self.player_id = player_id

    def select_move(self, state, legal_moves: Sequence[int]) -> int:
        if not legal_moves:
            return -1
        if len(legal_moves) == 1:
            return int(legal_moves[0])

        me = self.player_id if self.player_id is not None else int(state.current_player)
        active = state.active_players
        opp = None
        for p in range(4):
            if p != me and active[p]:
                opp = p
                break
        if opp is None:
            return int(legal_moves[0])

        score_fn = self.SCORE_FN

        best_action = int(legal_moves[0])
        best_expected = -float("inf")

        for a in legal_moves:
            try:
                after_mine = ludo_cpp.apply_move(state, int(a))
            except Exception:
                continue

            if after_mine.is_terminal:
                winner = (ludo_cpp.get_winner(after_mine)
                          if hasattr(ludo_cpp, "get_winner") else
                          (me if state.scores[me] == 3 else -1))
                expected = 1e9 if winner == me else 0.0
            else:
                expected = 0.0
                for d in range(1, 7):
                    sim = _clone_state(after_mine)
                    sim.current_dice_roll = d
                    cp = int(sim.current_player)
                    if cp == me:
                        # Bonus-turn: score as-is from MY POV (skip
                        # recursing into my own continuation).
                        expected += score_fn(me, sim) / 6.0
                        continue
                    opp_legal = ludo_cpp.get_legal_moves(sim)
                    if not opp_legal:
                        expected += score_fn(me, sim) / 6.0
                        continue
                    # Opp picks greedily under OWN scoring (i.e. base
                    # scoring from opp's POV — we don't assume opp uses
                    # our personality). This keeps opp model neutral.
                    best_opp_action = opp_legal[0]
                    best_opp_score = -float("inf")
                    for oa in opp_legal:
                        try:
                            after_opp = ludo_cpp.apply_move(sim, int(oa))
                        except Exception:
                            continue
                        s_opp = _base_score_position(cp, after_opp)
                        if s_opp > best_opp_score:
                            best_opp_score = s_opp
                            best_opp_action = oa
                    try:
                        after_opp = ludo_cpp.apply_move(sim, int(best_opp_action))
                    except Exception:
                        after_opp = sim
                    expected += score_fn(me, after_opp) / 6.0

            if expected > best_expected:
                best_expected = expected
                best_action = int(a)

        return best_action


# ─── Personality subclasses ───────────────────────────────────────────────


class AggressiveExpectimaxBot(_BaseExpectimaxV2):
    SCORE_FN = staticmethod(_score_aggressive)
    NAME = "AggressiveExpectimax"


class DefensiveExpectimaxBot(_BaseExpectimaxV2):
    SCORE_FN = staticmethod(_score_defensive)
    NAME = "DefensiveExpectimax"


class RacingExpectimaxBot(_BaseExpectimaxV2):
    SCORE_FN = staticmethod(_score_racing)
    NAME = "RacingExpectimax"


class MinimaxExpectimaxBot(_BaseExpectimaxV2):
    SCORE_FN = staticmethod(_score_minimax)
    NAME = "MinimaxExpectimax"


class BlockadeExpectimaxBot(_BaseExpectimaxV2):
    """Bonus for own 2-token stacks on safe squares — chokepoint
    play. Was in the original Phase 6 plan; built as a bonus variant
    to add positional diversity beyond Aggressive/Defensive."""
    SCORE_FN = staticmethod(_score_blockade)
    NAME = "BlockadeExpectimax"


# ─── Registry ─────────────────────────────────────────────────────────────


EXPECTIMAX_V2_REGISTRY = {
    "AggressiveExpectimax": AggressiveExpectimaxBot,
    "DefensiveExpectimax":  DefensiveExpectimaxBot,
    "RacingExpectimax":     RacingExpectimaxBot,
    "MinimaxExpectimax":    MinimaxExpectimaxBot,
    "BlockadeExpectimax":   BlockadeExpectimaxBot,
}
