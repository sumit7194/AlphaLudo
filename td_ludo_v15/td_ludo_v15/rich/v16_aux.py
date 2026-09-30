"""V16 auxiliary targets — the sparse tactical signals that drive property `b`.

The owner's idea names these directly: "the PHASE could change with OTHER
SPARSE SIGNALS like cuts, or home, or others". We use the two that are both
sparse and tactically real in 2-player Ludo:

    capture_available  can any of MY legal moves land on an opponent token?
    in_danger          can any opponent token reach one of mine on a 1-6?

Both are computed from the RAW GAME STATE at the decision point, not from the
encoded frame — the frame is a POV tensor and cannot be inverted back to token
positions. That is why this is called during rollout rather than at update time.

WHY THESE, AND WHY IT IS NOT CIRCULAR
-------------------------------------
`experiments/cogmap/CLS_RESULTS.md` measured that the v15.2 champion's CLS
summary encodes every counting/positional aggregate at R^2 0.83-0.99 but
capture-risk at only ~0.35. Danger is the thing this architecture demonstrably
fails to represent, so it is the honest place to give a second property its own
parameters. The aux head reads the CLS and never feeds the trunk, so it cannot
shortcut the policy.

Base rates are ~3% capture-available and ~14% in-danger; see
`V16RichTrainer.aux_pos_weight` for why that sparsity has to be reweighted
rather than left alone.
"""
from __future__ import annotations

from typing import List, Sequence, Set

from ..game.cells import cell_to_index, position_to_cell_in_pov

# Track length: positions 0..GOAL-1 are on-board; GOAL and beyond are home.
GOAL = 56
AUX_NAMES = ("capture_available", "in_danger")
N_AUX = len(AUX_NAMES)


def _cells_of(state, player: int, pov: int, lo: int = 0, hi: int = GOAL) -> Set[int]:
    """POV-relative cell indices of `player`'s tokens with lo <= pos < hi."""
    out: Set[int] = set()
    for t in range(4):
        p = int(state.player_positions[player][t])
        if lo <= p < hi:
            out.add(cell_to_index(*position_to_cell_in_pov(p, player, pov)))
    return out


def aux_targets(state, cp: int, legal: Sequence[int]) -> List[float]:
    """[capture_available, in_danger] for player `cp` at this decision point."""
    opp = (cp + 2) % 4
    dice = int(state.current_dice_roll)

    # in_danger — any opponent token that could land on one of mine with 1..6
    reach: Set[int] = set()
    for t in range(4):
        p = int(state.player_positions[opp][t])
        if 0 <= p < GOAL:
            for step in range(1, 7):
                if p + step < GOAL:
                    reach.add(cell_to_index(
                        *position_to_cell_in_pov(p + step, opp, cp)))
    in_danger = float(bool(_cells_of(state, cp, cp) & reach))

    # capture_available — any of MY legal moves landing on an opponent cell
    opp_cells = _cells_of(state, opp, cp)
    landing: Set[int] = set()
    for tok in legal:
        p = int(state.player_positions[cp][tok])
        if 0 <= p < GOAL and p + dice < GOAL:
            landing.add(cell_to_index(
                *position_to_cell_in_pov(p + dice, cp, cp)))
    capture_available = float(bool(landing & opp_cells))

    return [capture_available, in_danger]
