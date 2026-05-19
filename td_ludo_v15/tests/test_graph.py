"""Tests for V15 static graph topology."""
from __future__ import annotations

import pytest

from td_ludo_v15.game import graph
from td_ludo_v15.game.cells import (
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
from td_ludo_v15.game.graph import (
    EDGE_TYPE_MATRIX,
    EDGES,
    EdgeType,
    NUM_EDGE_TYPES,
    edge_count_by_type,
    out_edges_from,
    print_topology,
)


def test_num_edge_types():
    """16 edge types: NO_EDGE + 6 path_step + 6 path_back + 3 specials."""
    assert NUM_EDGE_TYPES == 16


def test_edge_type_matrix_shape():
    assert EDGE_TYPE_MATRIX.shape == (NUM_NODES, NUM_NODES)
    assert EDGE_TYPE_MATRIX.dtype.name == "int8"


def test_path_step_edges_count():
    """For each main-path position p in [0, 50] and each dice d in [1, 6],
    edge p → (p+d) exists if (p+d) ≤ 55 (i.e., target is on path or stretch).

    Total expected: sum over p in [0,50] of count of d such that p+d ≤ 55.
    For p ≤ 49: all 6 dice land within or at stretch (≤ 55) ✓
    For p = 50: d=1..5 land in stretch (51..55), d=6 → 56 (scored, no PATH_STEP edge).

    Plus stretch internal edges: for p ∈ [51..55], d ∈ [1..6] with target ≤ 55.
        p=51: d=1..4 → ok (52..55), d=5,6 → off
        p=52: d=1..3 → ok (53..55), d=4,5,6 → off
        ...
        p=55: no in-range d.

    Counts:
      main → path/stretch: 50 × 6 + 5 = 305 (positions 0..49 contribute 6 each, position 50 contributes 5)
      stretch → stretch:   4 + 3 + 2 + 1 + 0 = 10
      total PATH_STEP_*:   305 + 10 = 315
    Each PATH_STEP has a matching PATH_BACK reverse → 315 of those too.
    """
    counts = edge_count_by_type()
    total_step = sum(counts[f"PATH_STEP_{d}"] for d in range(1, 7))
    total_back = sum(counts[f"PATH_BACK_{d}"] for d in range(1, 7))
    assert total_step == total_back, "forward and backward edge counts must match"
    assert total_step == 315, f"expected 315 PATH_STEP_* edges, got {total_step}"


def test_no_duplicate_edges():
    """No (src, dst, type) tuple appears twice."""
    s = set(EDGES)
    assert len(s) == len(EDGES), (
        f"duplicate edges found: {len(EDGES) - len(s)} dupes"
    )


def test_home_unlock_edge():
    """Exactly one HOME_UNLOCK edge: from home counter (2,2) to spawn (6,1)."""
    home_idx = cell_to_index(*HOME_BASE_COUNTER)
    spawn_idx = cell_to_index(6, 1)  # P0 path position 0
    unlock_edges = [(s, d) for s, d, t in EDGES if t == EdgeType.HOME_UNLOCK]
    assert len(unlock_edges) == 1
    assert unlock_edges[0] == (home_idx, spawn_idx)


def test_stretch_to_score_edge():
    """One STRETCH_TO_SCORE edge: from last stretch cell (7,5) to MS (7,6)."""
    src_idx = cell_to_index(*HOME_STRETCH_CELLS_P0[-1])  # (7, 5)
    ms_idx = cell_to_index(*MS_CELL)
    stretch_edges = [(s, d) for s, d, t in EDGES if t == EdgeType.STRETCH_TO_SCORE]
    assert len(stretch_edges) == 1
    assert stretch_edges[0] == (src_idx, ms_idx)


def test_global_broadcast_edges_count():
    """4 global cells bidirectionally connected to every other node.

    Each global iteration emits:
      - g → other (for all other nodes): NUM_NODES - 1 forward edges
      - other → g (reverse, ONLY for non-global others to avoid duplicating
        global↔global pairs): NUM_NODES - 4 reverse edges
    Per global: (NUM_NODES - 1) + (NUM_NODES - 4) = 446 edges
    Total: 4 × 446 = 1784.
    """
    counts = edge_count_by_type()
    expected = 4 * ((NUM_NODES - 1) + (NUM_NODES - 4))
    assert counts["GLOBAL_BROADCAST"] == expected, (
        f"expected {expected} broadcast edges, got {counts['GLOBAL_BROADCAST']}"
    )


def test_global_broadcast_covers_all_nodes():
    """Each global must connect to every other node via GLOBAL_BROADCAST."""
    md_idx = cell_to_index(*MD_CELL)
    reachable_via_broadcast = set()
    for s, d, t in EDGES:
        if s == md_idx and t == EdgeType.GLOBAL_BROADCAST:
            reachable_via_broadcast.add(d)
    # MD should broadcast to every other node
    assert len(reachable_via_broadcast) == NUM_NODES - 1
    assert md_idx not in reachable_via_broadcast


def test_path_step_at_spawn_cell():
    """From cell (6, 1) — P0's spawn — PATH_STEP_1 through PATH_STEP_6 should
    all exist (positions 1..6 are all on the path)."""
    spawn_idx = cell_to_index(6, 1)
    types_from_spawn = {}
    for s, d, t in EDGES:
        if s == spawn_idx and t in (EdgeType.PATH_STEP_1, EdgeType.PATH_STEP_2,
                                    EdgeType.PATH_STEP_3, EdgeType.PATH_STEP_4,
                                    EdgeType.PATH_STEP_5, EdgeType.PATH_STEP_6):
            types_from_spawn.setdefault(EdgeType(t).name, []).append(d)
    for d in range(1, 7):
        assert f"PATH_STEP_{d}" in types_from_spawn, f"missing PATH_STEP_{d} from spawn"


def test_edge_type_matrix_consistent_with_edges():
    """EDGE_TYPE_MATRIX[s, d] should equal the LAST edge type of (s, d) in EDGES.

    Some pairs may have multiple edges across types (e.g., a path cell that's
    also a global-broadcast target). The matrix takes the later one.
    """
    last_seen = {}
    for s, d, t in EDGES:
        last_seen[(s, d)] = t
    for (s, d), t in last_seen.items():
        assert int(EDGE_TYPE_MATRIX[s, d]) == int(t), (
            f"matrix[{s},{d}] = {EDGE_TYPE_MATRIX[s, d]} but edges say {t}"
        )


def test_out_edges_from_spawn():
    """Eyeball test — spawn cell (6,1) should connect to positions 1..6 + globals."""
    out = out_edges_from(6, 1)
    # Should have at least 6 PATH_STEP edges + 4 (to globals, GLOBAL_BROADCAST
    # reverse direction). Plus 1 PATH_BACK from spawn back to ... wait, no:
    # path back from spawn means "what was here before pos 0" which doesn't exist
    # except via HOME_UNLOCK from home counter (incoming, not outgoing).
    # So out edges from (6,1):
    #   6 forward path-step edges (pos 1..6)
    #   1 incoming HOME_UNLOCK reverse? No — HOME_UNLOCK is one-directional.
    #   4 broadcast reverse edges to MD/OD/MS/OS
    edge_types = [t for _, t in out]
    # 6 PATH_STEP_* edges
    step_count = sum(1 for t in edge_types if t.startswith("PATH_STEP"))
    assert step_count == 6, f"expected 6 PATH_STEP_* edges from spawn, got {step_count}"
    # 4 GLOBAL_BROADCAST edges (spawn → each global)
    broadcast_count = sum(1 for t in edge_types if t == "GLOBAL_BROADCAST")
    assert broadcast_count == 4, (
        f"expected 4 broadcast edges from spawn to globals, got {broadcast_count}"
    )


def test_print_topology_runs():
    """Smoke test: print_topology produces a non-empty string."""
    out = print_topology(6, 1)
    assert "Out-edges" in out
    assert "PATH_STEP_1" in out
    assert "Legend" in out
