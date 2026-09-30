"""Consequence-prediction targets for the V13.6 world-model heads.

Ground-truth supervised targets, computed EXACTLY from the board state by
enumerating the 6 dice — no search, no policy rollout. These teach the
shared trunk to encode *consequences/risk* (the understanding both play-test
flaws show it lacks; see discussion/WORLD_MODEL_ATTACK_ANGLE.md).

Per own token currently on the main track:
  - capture_prob_next_turn: P(captured on the opponent's next turn),
    assuming opponents capture when able. Geometric model consistent with
    bias_penalties._in_capture_range: an opp token exactly d∈{1..6} cells
    behind on the shared track captures with dice d. The probability is the
    fraction of dice values for which SOME opp token can land on this cell.
  - cells_at_risk: capture_prob × progress_lost_if_captured. Encodes "how
    much do I lose if this token dies" — the value-at-risk signal.

Targets are also exposed in canonical-RANK slots (matching V135's symmetric
rank-indexed output) via compute_ranked_targets().

Known v1 simplifications (intentional, matches bias_penalties):
  - Ignores opp tokens peeling into their home stretch before reaching the
    cell (slight over-estimate of risk for opp tokens near home entry).
  - Ignores blockades (2-stacks blocking passage).
Both are rare and refinable later.
"""
from __future__ import annotations

import numpy as np

from td_ludo.game.bias_penalties import (
    _abs_pos, _is_main_track, _is_safe_square,
    BASE_POSITION, SCORE_POSITION,
)
from td_ludo.game.rank_mapping import state_to_rank_mapping

# Max progress a token can lose (spawn at 0 .. last main-track cell 50, plus
# the spawn step). Used to normalize cells_at_risk into [0,1] for a stable
# regression target.
_MAX_PROGRESS = 51.0


def _own_count_at(state, player, pos) -> int:
    return sum(1 for p in state.player_positions[player] if int(p) == pos)


def capture_prob_next_turn(state, player: int, pos: int) -> float:
    """P(an own token at relative `pos` is captured on the opponent's next
    turn). 0 if off-track, on a safe square, or part of a 2+ own stack."""
    if not _is_main_track(pos):
        return 0.0
    if _is_safe_square(player, pos):
        return 0.0
    if _own_count_at(state, player, pos) >= 2:   # stacked → uncapturable
        return 0.0

    my_abs = _abs_pos(player, pos)
    capturing_dice = set()
    for opp in range(4):
        if opp == player or not state.active_players[opp]:
            continue
        for ot in range(4):
            op = int(state.player_positions[opp][ot])
            if not _is_main_track(op):
                continue
            dist = (my_abs - _abs_pos(opp, op)) % 52
            if 1 <= dist <= 6:
                capturing_dice.add(dist)
    return len(capturing_dice) / 6.0


def progress_lost_if_captured(pos: int) -> float:
    """Cells of progress a token loses if captured (returns to base).
    On-track tokens lose `pos + 1` (spawn cell counts); off-track = 0."""
    if not _is_main_track(pos):
        return 0.0
    return float(pos + 1)


def compute_per_token_targets(state, player: int):
    """Returns (capture_prob[4], cells_at_risk_norm[4], valid_mask[4]) indexed
    by TOKEN-ID. valid=1 for own tokens currently on the main track (the only
    ones that can be captured / have meaningful risk)."""
    cap = np.zeros(4, dtype=np.float32)
    car = np.zeros(4, dtype=np.float32)
    valid = np.zeros(4, dtype=np.float32)
    for t in range(4):
        pos = int(state.player_positions[player][t])
        if not _is_main_track(pos):
            continue
        valid[t] = 1.0
        p = capture_prob_next_turn(state, player, pos)
        cap[t] = p
        car[t] = (p * progress_lost_if_captured(pos)) / _MAX_PROGRESS
    return cap, car, valid


def compute_ranked_targets(state, player: int):
    """Same targets but in canonical-RANK slots (matching V135's rank-indexed
    output head). rank 0 = most-advanced unique position … rank R-1 = least.
    Returns (capture_prob[4], cells_at_risk_norm[4], valid_mask[4]) where
    valid=1 for rank slots whose unique position is on the main track."""
    rank_positions, _rank_token_ids = state_to_rank_mapping(
        state.player_positions[player]
    )
    cap = np.zeros(4, dtype=np.float32)
    car = np.zeros(4, dtype=np.float32)
    valid = np.zeros(4, dtype=np.float32)
    for r, pos in enumerate(rank_positions):
        if r >= 4:
            break
        pos = int(pos)
        if not _is_main_track(pos):
            continue
        valid[r] = 1.0
        p = capture_prob_next_turn(state, player, pos)
        cap[r] = p
        car[r] = (p * progress_lost_if_captured(pos)) / _MAX_PROGRESS
    return cap, car, valid
