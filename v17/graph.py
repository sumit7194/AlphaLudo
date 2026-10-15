"""Static board graph for the V17 GraphTransformer (226 nodes = 225 cells + CLS).

Each ordered node pair gets an edge TYPE; the model learns one attention bias per
(type, head). Everything is in the mover's frame, for all four relative seats, so
the same graph serves 2-player and 4-player games.

Edge types
    1..6   STEP_d  : a token on the source cell reaches the target with a roll of d
                     (along any seat's route, including into its home run and onto
                     its scored cell — "home" — with the exact roll)
    7..12  BACK_d  : reverse of STEP_d (lets a cell see what can reach it)
    13     UNLOCK  : base cells <-> their seat's start square (a 6 releases a token)
    14     SEAT    : a seat's dice corner <-> its base cells and its scored cell
    15     GLOBAL  : the 8 special cells (dice corners, scored cells) <-> every node
    0      none

Changes vs the September 4P graph: every home-run cell (and the last loop cells)
links to the scored cell with the exact roll needed; all four base cells link to
the start square in both directions; per-seat special cells are associated.
"""
from __future__ import annotations

import numpy as np

from . import geometry as g

NONE, UNLOCK, SEAT, GLOBAL = 0, 13, 14, 15
N_EDGE_TYPES = 16


def step(d: int) -> int:
    return d


def back(d: int) -> int:
    return 6 + d


def build_edge_types() -> np.ndarray:
    m = np.zeros((g.N_NODES, g.N_NODES), dtype=np.int64)

    def put(a: int, b: int, t: int):
        prev = m[a, b]
        assert prev in (0, t) or t in (SEAT,), f"edge type clash at ({a},{b}): {prev} vs {t}"
        m[a, b] = t

    # Routes of all four relative seats (seat k == owner k seen from pov 0).
    for k in range(4):
        cells = [g.track_cell(p, k, 0) for p in range(56)] + [g.idx(*g.SCORED_CELLS[k])]  # index 56 = home
        for i in range(56):
            for d in range(1, 7):
                j = i + d
                if j > 56:
                    break
                put(cells[i], cells[j], step(d))
                put(cells[j], cells[i], back(d))
        start = cells[0]
        for (r, c) in g.BASE_CELLS[k]:
            put(g.idx(r, c), start, UNLOCK)
            put(start, g.idx(r, c), UNLOCK)

    specials = [g.idx(*c) for c in g.DICE_CELLS + g.SCORED_CELLS]
    for s in specials:
        for o in range(g.N_NODES):
            if o != s:
                if m[s, o] == NONE:
                    m[s, o] = GLOBAL
                if m[o, s] == NONE:
                    m[o, s] = GLOBAL

    for k in range(4):
        dice = g.idx(*g.DICE_CELLS[k])
        linked = [g.idx(*g.SCORED_CELLS[k])] + [g.idx(r, c) for (r, c) in g.BASE_CELLS[k]]
        for o in linked:
            m[dice, o] = SEAT
            m[o, dice] = SEAT
    return m


EDGE_TYPES = build_edge_types()
