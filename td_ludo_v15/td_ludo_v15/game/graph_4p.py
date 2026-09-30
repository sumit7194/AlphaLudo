"""V15 4-Player static graph topology.

The Graph Transformer attends over 226 nodes (225 board cells + 1 CLS readout).
Edges encode Ludo's path-graph structure in 4-player mode; each edge has a learned
type embedding that's added to attention logits as a bias term.

Edge categories (15 distinct typed edges, plus type-0 = "no edge"):
    PATH_STEP_1..6    : forward path-step edges (dice value = type-rank)
    PATH_BACK_1..6    : reverse path-step edges (lets a cell see "what's behind")
    HOME_UNLOCK       : home base counters -> spawn cells, gated by dice=6
    STRETCH_TO_SCORE  : last home-stretch cell -> scored slot
    GLOBAL_BROADCAST  : 4 dice corners + 4 scored centers bidirectionally connected to all cells

All edges defined in the **current player's POV (P0)** — 4-fold rotational symmetry.
The graph is built ONCE at import time and is a static numpy array.
"""
from __future__ import annotations

import enum
from typing import List, Tuple

import numpy as np

import td_ludo_v15_cpp as _cpp
from .cells import (
    BOARD_SIZE,
    CLS_INDEX,
    DICE_CELLS_4P,
    HOME_BASE_CELLS_4P,
    HOME_STRETCH_CELLS_4P,
    NUM_BOARD_CELLS,
    NUM_NODES,
    SCORED_CELLS_4P,
    SPECIAL_CELLS_4P,
    cell_to_index,
)


class EdgeType(enum.IntEnum):
    """Edge type IDs used in the (226, 226) bias-lookup matrix.

    Type 0 = NO_EDGE (default for unconnected node pairs). Total 16 IDs.
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


NUM_EDGE_TYPES = len(EdgeType)  # 16


def _pos_to_index_p0(pos: int) -> int:
    """Convert a path position in P0's POV to a board node index."""
    if pos == _cpp.BASE_POS:
        return cell_to_index(*HOME_BASE_CELLS_4P[0][0])
    if pos == _cpp.HOME_POS:
        return cell_to_index(*SCORED_CELLS_4P[0])
    r, c = _cpp.position_to_cell(pos, 0)
    return cell_to_index(r, c)


def build_edges_4p() -> List[Tuple[int, int, int]]:
    """Build the static edge list for 4-player Ludo in current mover's POV.
    
    Returns [(src_idx, dst_idx, type_id), ...].
    """
    edges: List[Tuple[int, int, int]] = []

    # ── 1. GLOBAL_BROADCAST: 4 dice corners + 4 scored cells ────────────────
    # Bidirectionally connected to every node on the board + CLS as baseline
    global_indices_list = [cell_to_index(r, c) for r, c in SPECIAL_CELLS_4P]
    global_indices_set = set(global_indices_list)

    for g_idx in global_indices_list:
        for other_idx in range(NUM_NODES):
            if other_idx == g_idx:
                continue
            edges.append((g_idx, other_idx, EdgeType.GLOBAL_BROADCAST))
            if other_idx not in global_indices_set:
                edges.append((other_idx, g_idx, EdgeType.GLOBAL_BROADCAST))

    # ── 2. Shared Outer Loop: 52-cell closed ring with PATH_STEP / BACK ──────
    ring_cells = [_cpp.position_to_cell(p, 0) for p in range(51)] + [(6, 0)]
    ring_indices = [cell_to_index(r, c) for r, c in ring_cells]

    for i in range(52):
        src = ring_indices[i]
        for d in range(1, 7):
            dst = ring_indices[(i + d) % 52]
            edges.append((src, dst, EdgeType.PATH_STEP_1 + d - 1))
            edges.append((dst, src, EdgeType.PATH_BACK_1 + d - 1))

    # ── 3. Home Stretch Entries for all 4 players ───────────────────────────
    # Player 0 turns off ring at index 50 into P0 stretch
    # Player 1 turns off ring at index 11 into P1 stretch
    # Player 2 turns off ring at index 24 into P2 stretch
    # Player 3 turns off ring at index 37 into P3 stretch
    turn_indices = [50, 11, 24, 37]
    for p_id in range(4):
        turn_ring_idx = turn_indices[p_id]
        stretch_cells = HOME_STRETCH_CELLS_4P[p_id]
        for k in range(6):  # steps before the turn
            ring_pos = (turn_ring_idx - k) % 52
            src = ring_indices[ring_pos]
            for d in range(k + 1, min(7, k + 1 + len(stretch_cells))):
                s_idx = d - k - 1
                dst = cell_to_index(*stretch_cells[s_idx])
                edges.append((src, dst, EdgeType.PATH_STEP_1 + d - 1))
                edges.append((dst, src, EdgeType.PATH_BACK_1 + d - 1))

    # ── 4. All 4 Players' Home Stretches Internal Edges ──
    # For each player, the 5 stretch cells have internal forward/backward steps.
    for p_id in range(4):
        stretch_cells = HOME_STRETCH_CELLS_4P[p_id]
        for s_idx in range(len(stretch_cells)):
            src = cell_to_index(*stretch_cells[s_idx])
            for d in range(1, 7):
                t_idx = s_idx + d
                if t_idx < len(stretch_cells):
                    dst = cell_to_index(*stretch_cells[t_idx])
                    edges.append((src, dst, EdgeType.PATH_STEP_1 + d - 1))
                    edges.append((dst, src, EdgeType.PATH_BACK_1 + d - 1))

    # ── 4. HOME_UNLOCK for all 4 bases: base counter -> spawn cell ─────────
    spawn_positions = [0, 13, 26, 39]
    for p_id in range(4):
        base_counter_idx = cell_to_index(*HOME_BASE_CELLS_4P[p_id][0])
        spawn_r, spawn_c = _cpp.position_to_cell(spawn_positions[p_id], 0)
        spawn_idx = cell_to_index(spawn_r, spawn_c)
        edges.append((base_counter_idx, spawn_idx, EdgeType.HOME_UNLOCK))

    # ── 5. STRETCH_TO_SCORE for all 4 stretches: stretch-end -> scored slot ──
    for p_id in range(4):
        last_stretch_cell = HOME_STRETCH_CELLS_4P[p_id][-1]
        last_stretch_idx = cell_to_index(*last_stretch_cell)
        scored_idx = cell_to_index(*SCORED_CELLS_4P[p_id])
        edges.append((last_stretch_idx, scored_idx, EdgeType.STRETCH_TO_SCORE))

    return edges


def build_edge_type_matrix_4p() -> np.ndarray:
    """Returns a (NUM_NODES, NUM_NODES) int8 matrix where entry [src, dst]
    is the EdgeType ID of the edge from src to dst (0 if no edge).
    """
    edges = build_edges_4p()
    mat = np.zeros((NUM_NODES, NUM_NODES), dtype=np.int8)
    for src, dst, t in edges:
        mat[src, dst] = int(t)
    return mat


# Precompute at module load
EDGES_4P: List[Tuple[int, int, int]] = build_edges_4p()
EDGE_TYPE_MATRIX_4P: np.ndarray = build_edge_type_matrix_4p()
