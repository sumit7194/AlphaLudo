"""Adaptive + ensemble expectimax variants — Phase 6 of STRONG_BOTS_PLAN.

Two new bot ideas distinct from the depth/prior axes:

  AdaptiveExpectimaxBot
    Switches scoring function based on game phase. Early game = race
    (most tokens still in base, value forward progress); midgame =
    aggressive (most tokens out, capture opportunities matter most);
    endgame = defensive (few tokens left, protect leaders).

    Phase classification by total tokens-still-in-base across both players:
      early   : ≥ 5 tokens in base (most haven't spawned)
      mid     : 2-4 in base
      endgame : ≤ 1 in base (everyone's out, racing home)

    Scoring per phase:
      early   → racing scoring (no opp consideration, just go)
      mid     → aggressive scoring (capture-oriented)
      endgame → defensive scoring (protect own leaders)

    This isn't strictly "stronger" than any single-personality variant
    but provides a DIFFERENT play style — opportunistic across the game,
    where each phase exploits different weaknesses in opponents.

  VoteExpectimaxBot
    Run all 4 personality scorers + base, vote (Borda count) on the
    preferred action. Resilient to individual scoring quirks; closer
    to an "ensemble" decision rule.

    Cost: 5× a single expectimax decision (~250ms). Slow but diverse
    opponent — the model can't easily learn "the" scoring function the
    bot uses because it's a mixture.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

import td_ludo_cpp as ludo_cpp

from td_ludo.game.strong_bots import (
    _score_position as _base_score_position,
    _clone_state,
    _BASE_POS,
    _HOME_POS,
)
from td_ludo.game.strong_bots_v2 import (
    _BaseExpectimaxV2,
    _score_aggressive, _score_defensive, _score_racing, _score_minimax,
)


# ─── Phase classification ─────────────────────────────────────────────────


def _game_phase(state) -> str:
    """Return 'early', 'mid', or 'endgame' based on token distribution."""
    positions = np.asarray(state.player_positions, dtype=np.int8)
    active = np.asarray(state.active_players, dtype=bool)
    in_base = 0
    for p in range(4):
        if not active[p]:
            continue
        for t in range(4):
            if int(positions[p][t]) == _BASE_POS:
                in_base += 1
    if in_base >= 5:
        return 'early'
    if in_base >= 2:
        return 'mid'
    return 'endgame'


def _score_adaptive(player: int, state) -> float:
    """Dispatch on game phase."""
    phase = _game_phase(state)
    if phase == 'early':
        return _score_racing(player, state)
    if phase == 'mid':
        return _score_aggressive(player, state)
    return _score_defensive(player, state)


# ─── AdaptiveExpectimaxBot ────────────────────────────────────────────────


class AdaptiveExpectimaxBot(_BaseExpectimaxV2):
    """Depth-1 expectimax with phase-dependent scoring. See module
    docstring for the phase-→-scorer mapping."""
    SCORE_FN = staticmethod(_score_adaptive)
    NAME = "AdaptiveExpectimax"


# ─── VoteExpectimaxBot ────────────────────────────────────────────────────


class VoteExpectimaxBot:
    """Ensemble: each of {base, aggressive, defensive, racing, minimax}
    scoring functions picks a preferred action via depth-1 expectimax,
    then we vote (Borda count). Ties broken by base scoring's ranking.

    Cost: 5 expectimax decisions per move (~250ms total, ~12 g/s).
    Provides a fundamentally noisy/mixed opponent — there's no single
    scoring function the model can pattern-match against.
    """

    NAME = "VoteExpectimax"
    SCORERS = [
        ("base",       _base_score_position),
        ("aggressive", _score_aggressive),
        ("defensive",  _score_defensive),
        ("racing",     _score_racing),
        ("minimax",    _score_minimax),
    ]

    def __init__(self, player_id: Optional[int] = None):
        self.player_id = player_id

    def select_move(self, state, legal_moves: Sequence[int]) -> int:
        if not legal_moves:
            return -1
        if len(legal_moves) == 1:
            return int(legal_moves[0])

        # For each scorer, compute expectimax Q values, rank actions.
        ranks_per_action = {a: [] for a in legal_moves}
        for _, score_fn in self.SCORERS:
            q_per_action = self._expectimax_q(state, legal_moves, score_fn)
            # Rank actions by Q (highest = rank 0, used as Borda points)
            sorted_actions = sorted(
                legal_moves, key=lambda x: q_per_action.get(x, -float("inf")),
                reverse=True,
            )
            for i, a in enumerate(sorted_actions):
                # Borda: action at rank i gets (n - i) points
                ranks_per_action[a].append(len(legal_moves) - i)

        # Sum points across scorers; pick action with highest total.
        # Ties broken by base scoring's preferred (which had been
        # appended first, so the first scorer's argmax wins).
        totals = {a: sum(pts) for a, pts in ranks_per_action.items()}
        best_a = max(totals, key=lambda a: (totals[a], ranks_per_action[a][0]))
        return int(best_a)

    def _expectimax_q(self, state, legal_moves, score_fn):
        """Same depth-1 expectimax as _BaseExpectimaxV2.select_move, but
        returns the full per-action Q dict instead of just the argmax."""
        me = self.player_id if self.player_id is not None else int(state.current_player)
        active = state.active_players
        opp = None
        for p in range(4):
            if p != me and active[p]:
                opp = p
                break
        if opp is None:
            return {a: 0.0 for a in legal_moves}

        q = {}
        for a in legal_moves:
            try:
                after_mine = ludo_cpp.apply_move(state, int(a))
            except Exception:
                q[a] = -float("inf")
                continue
            if after_mine.is_terminal:
                winner = (ludo_cpp.get_winner(after_mine)
                          if hasattr(ludo_cpp, "get_winner") else
                          (me if state.scores[me] == 3 else -1))
                q[a] = 1e9 if winner == me else 0.0
                continue
            expected = 0.0
            for d in range(1, 7):
                sim = _clone_state(after_mine)
                sim.current_dice_roll = d
                cp = int(sim.current_player)
                if cp == me:
                    expected += score_fn(me, sim) / 6.0
                    continue
                opp_legal = ludo_cpp.get_legal_moves(sim)
                if not opp_legal:
                    expected += score_fn(me, sim) / 6.0
                    continue
                # Opp greedy under base scoring
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
            q[a] = expected
        return q


# ─── Registry ─────────────────────────────────────────────────────────────


ADAPTIVE_REGISTRY = {
    "AdaptiveExpectimax": AdaptiveExpectimaxBot,
    "VoteExpectimax":     VoteExpectimaxBot,
}
