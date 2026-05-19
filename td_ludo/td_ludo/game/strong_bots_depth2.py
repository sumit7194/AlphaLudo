"""Depth-2 expectimax variants — Phase 3 of STRONG_BOTS_PLAN.

Extends the depth-1 lookahead of `ExpectimaxBot` / `_BaseExpectimaxV2`
by one ply: `my_move × opp_dice × opp_move × my_dice × my_move_again`,
score at the resulting leaf.

Cost analysis (worst case, 4 legal moves at each step):
  depth-1: 4 × 6 × 4         = 96 leaves
  depth-2: 4 × 6 × 4 × 6 × 4 = 2304 leaves
  → ~24× cost increase. ~50ms → ~1.2s/decision. ~0.5 g/s.

Architecture: `Depth2ExpectimaxBot` takes the same scoring function
slot as `_BaseExpectimaxV2`. Default scoring is the base `_score_position`
from strong_bots.py; subclasses can override to combine depth with a
personality (e.g., depth-2 + aggressive scoring).

For training-opponent use: only worth including if the depth meaningfully
strengthens vs depth-1. If the marginal gain is <5pp head-to-head,
depth-1 is the better cost/strength tradeoff for the training pool.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

import td_ludo_cpp as ludo_cpp

from td_ludo.game.strong_bots import (
    _score_position as _base_score_position,
    _clone_state,
)


class Depth2ExpectimaxBot:
    """Depth-2 expectimax with pluggable scoring.

    Tree:
        for each my_first_action a:
            s' = apply(a)
            for each opp_dice d_opp:
                s'_d = clone(s', dice=d_opp)
                opp_best = argmax over opp_legal of score_after_opp(opp_legal_action)
                s'' = apply(s'_d, opp_best)
                for each my_dice d_me:
                    s''_d = clone(s'', dice=d_me)
                    my_best_score = max over my_legal of score(apply(s''_d, my_legal_action))
                    inner_expected += my_best_score / 6
                outer_expected += inner_expected / 6
            Q(a) = outer_expected
        pick argmax_a Q(a)

    Same opp-greedy approximation as depth-1 (opp picks max-own-score,
    not adversarial — keeps cost bounded). The second "my" step uses
    full expectation over my dice and max over my actions (since I'm
    picking optimally).

    Note: I do NOT use the personality scoring functions from
    strong_bots_v2 at the inner my-greedy step; that uses base scoring.
    Only the outer (depth-2 leaf) eval uses the personality. This keeps
    the inner search consistent with what depth-1 would do.
    """

    SCORE_FN = staticmethod(_base_score_position)
    NAME = "Depth2Expectimax"

    def __init__(self, player_id: Optional[int] = None,
                 score_fn=None):
        self.player_id = player_id
        if score_fn is not None:
            self._score_fn = score_fn
        else:
            self._score_fn = self.SCORE_FN

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

        best_action = int(legal_moves[0])
        best_q = -float("inf")

        for a in legal_moves:
            try:
                after_mine = ludo_cpp.apply_move(state, int(a))
            except Exception:
                continue

            if after_mine.is_terminal:
                winner = (ludo_cpp.get_winner(after_mine)
                          if hasattr(ludo_cpp, "get_winner") else
                          (me if state.scores[me] == 3 else -1))
                q = 1e9 if winner == me else 0.0
            else:
                q = self._eval_after_my_move(after_mine, me)

            if q > best_q:
                best_q = q
                best_action = int(a)

        return best_action

    # ──────────────────────────────────────────────────────────────────

    def _eval_after_my_move(self, after_mine, me: int) -> float:
        """Expectation over opp_dice of: opp_plays_best, then expectation
        over my_dice of: I play best, scored from MY POV at depth-2 leaf."""
        outer = 0.0
        for d_opp in range(1, 7):
            opp_sim = _clone_state(after_mine)
            opp_sim.current_dice_roll = d_opp
            cp = int(opp_sim.current_player)
            if cp == me:
                # Bonus-turn: skip to my second action with this dice.
                inner = self._eval_my_second_dice(opp_sim, me, d_opp)
                outer += inner / 6.0
                continue
            opp_legal = ludo_cpp.get_legal_moves(opp_sim)
            if not opp_legal:
                # Opp passes — just evaluate from current state.
                outer += self._score_fn(me, opp_sim) / 6.0
                continue
            # Opp picks greedy under base scoring (POV: opp).
            best_opp_action = opp_legal[0]
            best_opp_score = -float("inf")
            for oa in opp_legal:
                try:
                    after_opp = ludo_cpp.apply_move(opp_sim, int(oa))
                except Exception:
                    continue
                s_opp = _base_score_position(cp, after_opp)
                if s_opp > best_opp_score:
                    best_opp_score = s_opp
                    best_opp_action = oa
            try:
                after_opp = ludo_cpp.apply_move(opp_sim, int(best_opp_action))
            except Exception:
                after_opp = opp_sim

            if after_opp.is_terminal:
                winner = (ludo_cpp.get_winner(after_opp)
                          if hasattr(ludo_cpp, "get_winner") else -1)
                outer += (1e9 if winner == me else 0.0) / 6.0
                continue

            inner = self._eval_after_opp_response(after_opp, me)
            outer += inner / 6.0

        return outer

    def _eval_after_opp_response(self, after_opp, me: int) -> float:
        """Expectation over my next dice of: I pick the best action,
        scored at the resulting leaf from MY POV."""
        cp = int(after_opp.current_player)
        if cp != me:
            # Opp's bonus turn (e.g., opp captured); treat as score-now.
            return self._score_fn(me, after_opp)
        inner = 0.0
        for d_me in range(1, 7):
            inner += self._eval_my_second_dice(after_opp, me, d_me) / 6.0
        return inner

    def _eval_my_second_dice(self, state, me: int, dice: int) -> float:
        """For my second action with given dice, pick the best move
        under self._score_fn and return its leaf score."""
        sim = _clone_state(state)
        sim.current_dice_roll = dice
        # If sim.current_player ended up != me (shouldn't normally), score now.
        if int(sim.current_player) != me:
            return self._score_fn(me, sim)
        legal = ludo_cpp.get_legal_moves(sim)
        if not legal:
            return self._score_fn(me, sim)
        best = -float("inf")
        for a in legal:
            try:
                after = ludo_cpp.apply_move(sim, int(a))
            except Exception:
                continue
            if after.is_terminal:
                winner = (ludo_cpp.get_winner(after)
                          if hasattr(ludo_cpp, "get_winner") else -1)
                s = 1e9 if winner == me else 0.0
            else:
                s = self._score_fn(me, after)
            if s > best:
                best = s
        return best


# ─── Personality combinations at depth-2 ──────────────────────────────────


from td_ludo.game.strong_bots_v2 import (
    _score_aggressive, _score_defensive, _score_racing, _score_minimax,
)


class Depth2AggressiveExpectimaxBot(Depth2ExpectimaxBot):
    SCORE_FN = staticmethod(_score_aggressive)
    NAME = "Depth2AggressiveExpectimax"


class Depth2DefensiveExpectimaxBot(Depth2ExpectimaxBot):
    SCORE_FN = staticmethod(_score_defensive)
    NAME = "Depth2DefensiveExpectimax"


DEPTH2_REGISTRY = {
    "Depth2Expectimax":           Depth2ExpectimaxBot,
    "Depth2AggressiveExpectimax": Depth2AggressiveExpectimaxBot,
    "Depth2DefensiveExpectimax":  Depth2DefensiveExpectimaxBot,
}
