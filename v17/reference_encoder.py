"""Independent pure-Python implementation of the unified (V17) encoding.

Used only by tests to check the Rust encoder (ludo_core::unified). The spec is
documented in crates/ludo_core/src/unified.rs.
"""
from __future__ import annotations

import numpy as np

from . import geometry as g

F = 5


def encode(positions, current_player: int, dice: int, active, consecutive_sixes=(0, 0, 0, 0)):
    """positions: 4x4 (seat, token) with -1 base, 0..55 track, 99 home.
    Returns (frame int8 (225*5,), legal mask float32 (225,)) in the mover's frame."""
    pov = current_player
    x = np.full((g.N_CELLS, F), -1, dtype=np.int8)
    scores = [sum(1 for p in positions[s] if p == g.HOME_POS) for s in range(4)]
    for k in range(4):
        seat = (pov + k) % 4
        if not active[seat]:
            continue
        for pos in range(56):
            x[g.track_cell(pos, seat, pov), k] = 0
        for (r, c) in g.BASE_CELLS[k]:
            x[g.idx(r, c), k] = 0
        n_base = 0
        for pos in positions[seat]:
            if pos == g.BASE_POS:
                n_base += 1
            elif pos != g.HOME_POS:
                x[g.track_cell(pos, seat, pov), k] += 1
        for i, (r, c) in enumerate(g.BASE_CELLS[k]):
            x[g.idx(r, c), k] = 1 if i < n_base else 0
    for pos in range(56):
        x[g.track_cell(pos, pov, pov), 4] = 1 if g.is_safe_rel(pos, pov) else 0
    for (r, c) in g.BASE_CELLS[0]:
        x[g.idx(r, c), 4] = 1
    for k in range(4):
        seat = (pov + k) % 4
        if not active[seat]:
            continue
        d = g.idx(*g.DICE_CELLS[k])
        x[d, k] = 1 + consecutive_sixes[seat]
        x[d, 4] = dice if seat == current_player else 0
        x[g.idx(*g.SCORED_CELLS[k]), k] = scores[seat]

    mask = np.zeros(g.N_CELLS, dtype=np.float32)
    if dice:
        for t in range(4):
            pos = positions[pov][t]
            if pos == g.BASE_POS:
                if dice == 6:
                    mask[g.idx(*g.BASE_CELLS[0][0])] = 1
            elif pos != g.HOME_POS and pos + dice <= 56:
                mask[g.track_cell(pos, pov, pov)] = 1
    return x.reshape(-1), mask
