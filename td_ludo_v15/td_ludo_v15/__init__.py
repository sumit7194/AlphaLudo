"""td_ludo_v15 — V15 (Graph Transformer over per-cell triplet) pipeline.

Public API (populated as phases land):
    game.cells: cell↔index, special cells, rotation helpers
    game.state: 8-frame history wrapper around the cpp engine
    game.encoder: per-cell triplet encoder
    game.graph: static graph topology
    models.v15: V15GraphTransformer
    viz.board_viewer: side-by-side board + encoding dump
"""
__version__ = "0.0.1"
