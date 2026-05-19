"""V15 static graph topology.

The Graph Transformer attends over 226 nodes (225 board cells + 1 CLS readout).
Edges encode Ludo's path-graph structure; each edge has a learned type
embedding that's added to attention logits as a bias term.

Edge categories (15 distinct typed edges, plus type-0 = "no edge"):
    PATH_STEP_1..6    : forward path-step edges (dice value = type-rank)
    PATH_BACK_1..6    : reverse path-step edges (lets a cell see "what's behind")
    HOME_UNLOCK       : my home counter (2,2) → spawn cell (6,1), dice=6 unlock
    STRETCH_TO_SCORE  : my last home-stretch cell (7,5) → MS scored slot (7,6)
    GLOBAL_BROADCAST  : MD/OD/MS/OS bidirectionally connected to all cells

All edges defined in the **current player's POV** — same canonical layout
regardless of which player is current (the V18 encoder + the cpp engine's
rotation handles board orientation). The graph is built ONCE at import time
and is a static numpy array forever.
"""
from __future__ import annotations

import enum
from typing import List, Tuple

import numpy as np

import td_ludo_v15_cpp as _cpp
from .cells import (
    BOARD_SIZE,
    CLS_INDEX,
    HOME_BASE_COUNTER,
    HOME_STRETCH_CELLS_P0,
    MD_CELL,
    MS_CELL,
    NUM_BOARD_CELLS,
    NUM_NODES,
    OD_CELL,
    OS_CELL,
    cell_to_index,
)


class EdgeType(enum.IntEnum):
    """Edge type IDs used in the (226, 226) bias-lookup matrix.

    Type 0 = NO_EDGE (default for unconnected node pairs). Subsequent IDs
    are the typed-edge categories. Total 16 IDs.
    """
    NO_EDGE = 0
    PATH_STEP_1 = 1
    PATH_STEP_2 = 2
    PATH_STEP_3 = 3
    PATH_STEP_4 = 4
    PATH_STEP_5 = 5
    PATH_STEP_6 = 6
    PATH_BACK_1 = 7
    PATH_BACK_2 = 8
    PATH_BACK_3 = 9
    PATH_BACK_4 = 10
    PATH_BACK_5 = 11
    PATH_BACK_6 = 12
    HOME_UNLOCK = 13
    STRETCH_TO_SCORE = 14
    GLOBAL_BROADCAST = 15


NUM_EDGE_TYPES = len(EdgeType)  # 16 (including NO_EDGE)


# ─── Canonical position → cell index (P0 POV) ─────────────────────────────
# Path positions 0..50 map to specific (r,c) cells. We use the cpp helper
# at module-load to get these and convert to linear indices.

def _pos_to_index(pos: int) -> int:
    """Convert a path position (in current-player POV) to a board node index.

    Special cases:
        BASE_POS → home counter cell (2, 2) → index 33
        HOME_POS → home center (7, 6) which is MS slot. We don't use a
                   dedicated node for HOME; tokens scored count goes to MS
                   via the encoder. Graph nodes never reference HOME directly.
    """
    if pos == _cpp.BASE_POS:
        return cell_to_index(*HOME_BASE_COUNTER)
    if pos == _cpp.HOME_POS:
        return cell_to_index(*MS_CELL)
    r, c = _cpp.position_to_cell(pos, 0)  # P0-canonical = current-player POV
    return cell_to_index(r, c)


def build_edges() -> List[Tuple[int, int, int]]:
    """Build the static edge list. Returns `[(src_idx, dst_idx, type_id), ...]`.

    Edges are directed. The bias matrix `edge_type_matrix()` gives O(1)
    lookup of "what type of edge connects src → dst" for the attention layer.
    """
    edges: List[Tuple[int, int, int]] = []

    # ── PATH_STEP_d (forward) and PATH_BACK_d (reverse), d ∈ {1..6} ────
    # For each path position p ∈ [0, 50] and each dice d, edge p → p+d if
    # p+d is on the path (≤ 50) OR on home stretch (51..55) OR scored (56→HOME).
    # The reverse PATH_BACK_d goes the other direction.
    #
    # Note: path positions [51..55] are the home stretch. Tokens reaching
    # position 56 are scored (go to HOME). We add the home stretch edges
    # specifically — there's no "position 56" node, it collapses to MS.
    for p in range(51):  # main path
        src = _pos_to_index(p)
        for d in range(1, 7):
            target = p + d
            if target <= 50:
                # Within main path
                dst = _pos_to_index(target)
                edges.append((src, dst, EdgeType.PATH_STEP_1 + d - 1))
                edges.append((dst, src, EdgeType.PATH_BACK_1 + d - 1))
            elif target <= 55:
                # Onto home stretch — but only YOUR stretch from YOUR path;
                # there's no path → opp_stretch edge.
                dst = _pos_to_index(target)
                edges.append((src, dst, EdgeType.PATH_STEP_1 + d - 1))
                edges.append((dst, src, EdgeType.PATH_BACK_1 + d - 1))
            elif target == 56:
                # Scored — edge to MS slot. STRETCH_TO_SCORE handles the
                # canonical case (pos 55 → MS) separately below; from
                # mid-path going +d directly to 56 only happens from pos 50.
                # We DON'T add a generic PATH_STEP edge to MS here — that's
                # the role of STRETCH_TO_SCORE for symmetry.
                pass

    # ── Home stretch internal edges (positions 51..55) ──────────────────
    # Already covered by the loop above for paths from main → stretch.
    # But we also want stretch → stretch+d transitions.
    for p in range(51, 56):  # 51, 52, 53, 54, 55
        src = _pos_to_index(p)
        for d in range(1, 7):
            target = p + d
            if target <= 55:
                dst = _pos_to_index(target)
                edges.append((src, dst, EdgeType.PATH_STEP_1 + d - 1))
                edges.append((dst, src, EdgeType.PATH_BACK_1 + d - 1))
            # target = 56 is the STRETCH_TO_SCORE edge, handled below.

    # ── HOME_UNLOCK: home counter → spawn cell, gated by dice=6 ─────────
    home_idx = cell_to_index(*HOME_BASE_COUNTER)
    spawn_idx = _pos_to_index(0)
    edges.append((home_idx, spawn_idx, EdgeType.HOME_UNLOCK))

    # ── STRETCH_TO_SCORE: every stretch-end → MS scored slot ────────────
    # Specifically, pos 55 (the last stretch cell) → MS. Each stretch cell
    # X with X + d == 56 also goes to scored, but those use PATH_STEP_d.
    # We add a dedicated STRETCH_TO_SCORE edge from pos=55 only (cleanest
    # interpretation: "the final stretch cell is one move away from scoring").
    stretch_last_idx = _pos_to_index(55)
    ms_idx = cell_to_index(*MS_CELL)
    edges.append((stretch_last_idx, ms_idx, EdgeType.STRETCH_TO_SCORE))

    # ── GLOBAL_BROADCAST: each global (MD, OD, MS, OS) bidirectionally
    #    connected to every other node (excluding self-loops and avoiding
    #    double-emission for global↔global pairs) ───────────────────────
    global_indices_list = [
        cell_to_index(*MD_CELL),
        cell_to_index(*OD_CELL),
        cell_to_index(*MS_CELL),
        cell_to_index(*OS_CELL),
    ]
    global_indices_set = set(global_indices_list)
    for g_idx in global_indices_list:
        for other_idx in range(NUM_NODES):
            if other_idx == g_idx:
                continue
            # Always emit g_idx → other_idx
            edges.append((g_idx, other_idx, EdgeType.GLOBAL_BROADCAST))
            # Emit reverse other_idx → g_idx ONLY if other is not a global
            # (otherwise the other global's own iteration will emit it,
            # producing a duplicate edge).
            if other_idx not in global_indices_set:
                edges.append((other_idx, g_idx, EdgeType.GLOBAL_BROADCAST))

    return edges


def build_edge_type_matrix() -> np.ndarray:
    """Returns a `(NUM_NODES, NUM_NODES)` int8 matrix where entry [src, dst]
    is the EdgeType ID of the edge from src to dst (0 if no edge).

    For pairs with multiple edge types (e.g., a path cell that's also
    reachable from MD via GLOBAL_BROADCAST), the LATER edge in the build
    order wins. In practice this means GLOBAL_BROADCAST overrides path
    edges from a global node — which is fine because global cells aren't
    path cells.
    """
    edges = build_edges()
    mat = np.zeros((NUM_NODES, NUM_NODES), dtype=np.int8)
    for src, dst, t in edges:
        mat[src, dst] = int(t)
    return mat


# Precompute at module load
EDGES: List[Tuple[int, int, int]] = build_edges()
EDGE_TYPE_MATRIX: np.ndarray = build_edge_type_matrix()


# ─── Debugging / visualization ────────────────────────────────────────────


def edge_count_by_type() -> dict:
    """Returns a dict EdgeType.name → count."""
    counts = {t.name: 0 for t in EdgeType}
    for _, _, t in EDGES:
        counts[EdgeType(t).name] += 1
    return counts


def out_edges_from(row: int, col: int) -> List[Tuple[Tuple[int, int], str]]:
    """List edges originating from cell (row, col).

    Returns list of `((dst_row, dst_col), edge_type_name)` tuples.
    """
    from .cells import index_to_cell
    src_idx = cell_to_index(row, col)
    out = []
    for s, d, t in EDGES:
        if s == src_idx:
            if d == CLS_INDEX:
                out.append(("CLS", EdgeType(t).name))
            else:
                out.append((index_to_cell(d), EdgeType(t).name))
    return out


def print_topology(row: int, col: int) -> str:
    """ASCII-render the edges leaving cell (row, col) as a 15×15 overlay.

    Each cell `(r, c)` in the overlay shows the edge-type id that connects
    (row, col) → (r, c), or `.` if no edge.
    """
    src_idx = cell_to_index(row, col)
    overlay = np.full((BOARD_SIZE, BOARD_SIZE), -1, dtype=np.int8)
    for s, d, t in EDGES:
        if s == src_idx and d < NUM_BOARD_CELLS:
            r = d // BOARD_SIZE
            c = d % BOARD_SIZE
            overlay[r, c] = int(t)
    lines = [f"Out-edges from cell ({row}, {col}) — source = '*':"]
    fmt = "{:>3}"
    header = "    " + "".join(fmt.format(c) for c in range(BOARD_SIZE))
    lines.append(header)
    for r in range(BOARD_SIZE):
        row_repr = []
        for c in range(BOARD_SIZE):
            if r == row and c == col:
                row_repr.append(fmt.format("*"))
            elif overlay[r, c] < 0:
                row_repr.append(fmt.format("."))
            else:
                row_repr.append(fmt.format(int(overlay[r, c])))
        lines.append(f"{r:>2}  {''.join(row_repr)}")
    lines.append("")
    lines.append("Legend (edge type IDs):")
    for t in EdgeType:
        if t == EdgeType.NO_EDGE:
            continue
        lines.append(f"  {t.value:>2}  {t.name}")
    return "\n".join(lines)
